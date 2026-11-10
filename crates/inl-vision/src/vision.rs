//! The Gemma 4 vision tower (`vision_tower.*` + `embed_vision.*`) used by
//! EmbeddingGemma 2, written against `transformers` `Gemma4VisionModel`
//! (Apache-2.0):
//!
//! * patch embedder: `2 * (x - 0.5)`, a bias-free input projection, and a
//!   2-D position-embedding table lookup summed over `(x, y)` (padding zeroed);
//! * 16 bidirectional encoder layers with axial 2-D RoPE, query/key RMSNorm
//!   (with weight) and a weight-free value RMSNorm, attention scaling `1.0`;
//! * a per-layer MLP `gelu_pytorch_tanh(gate) * up -> down`;
//! * `3 x 3` spatial average pooling in patch-position order, scaled by
//!   `sqrt(hidden_size)` in float32, then the padding rows stripped;
//! * `embed_vision`: a weight-free RMSNorm and a `768 -> text_hidden` projection
//!   into the text tower's space.
//!
//! The checkpoint stores the projection weights wrapped in `Gemma4ClippableLinear`
//! (`.linear.weight`); with `use_clipped_linears = false` the clamps are the
//! identity, so plain bias-free linears are equivalent.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, Activation, Linear, VarBuilder};
use inl_core::{InlError, Result as InlResult};
use serde::Deserialize;

use crate::preprocess::{PATCH_SIZE, POOLING_KERNEL};

/// Additive mask value for a padding key. Finite so a fully-masked row stays
/// finite; padding query rows are discarded downstream anyway.
const MASKED: f32 = -1e9;

/// Vision tensor rows processed per attention block (bounds the score matrix).
const ATTENTION_TILE: usize = 256;

#[derive(Deserialize, Debug, Clone)]
struct RopeParameters {
    rope_theta: f64,
    rope_type: String,
}

/// The checkpoint's `vision_config`.
#[derive(Deserialize, Debug, Clone)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    hidden_activation: Activation,
    rms_norm_eps: f64,
    #[serde(default)]
    attention_bias: bool,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    rope_parameters: RopeParameters,
    #[serde(default)]
    standardize: bool,
    #[serde(default)]
    use_clipped_linears: bool,
}

/// Only the vision half of the checkpoint config.
#[derive(Deserialize)]
struct CheckpointConfig {
    vision_config: VisionConfig,
}

impl VisionConfig {
    /// Parse the `vision_config` block of a checkpoint `config.json`.
    pub fn parse(bytes: &[u8]) -> InlResult<Self> {
        let checkpoint: CheckpointConfig = serde_json::from_slice(bytes).map_err(|error| {
            InlError::Inference(format!("config.json has no vision_config: {error}"))
        })?;
        let config = checkpoint.vision_config;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> InlResult<()> {
        let invalid = |what: String| {
            Err(InlError::Inference(format!(
                "unsupported vision config: {what}"
            )))
        };
        if self.patch_size != PATCH_SIZE {
            return invalid(format!("patch_size {}", self.patch_size));
        }
        if self.pooling_kernel_size != POOLING_KERNEL {
            return invalid(format!("pooling_kernel_size {}", self.pooling_kernel_size));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return invalid(format!("head_dim {}", self.head_dim));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return invalid("attention head counts".to_string());
        }
        if self.hidden_activation != Activation::GeluPytorchTanh {
            return invalid(format!("hidden_activation {:?}", self.hidden_activation));
        }
        if self.attention_bias {
            return invalid("attention_bias".to_string());
        }
        if self.rope_parameters.rope_type != "axial" {
            return invalid(format!("rope_type {:?}", self.rope_parameters.rope_type));
        }
        if self.standardize {
            return invalid("standardize".to_string());
        }
        if self.use_clipped_linears {
            return invalid("use_clipped_linears".to_string());
        }
        Ok(())
    }
}

/// RMSNorm with an optional weight (the value norm has none), matching
/// `Gemma4RMSNorm`.
struct RmsNorm {
    weight: Option<Tensor>,
    dim: usize,
    eps: f32,
    device: Device,
    dtype: DType,
}

impl RmsNorm {
    fn full(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: Some(vb.get(cfg.hidden_size, "weight")?),
            dim: cfg.hidden_size,
            eps: cfg.rms_norm_eps as f32,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn head_dim(cfg: &VisionConfig, vb: VarBuilder, with_scale: bool) -> Result<Self> {
        let weight = if with_scale {
            Some(vb.get(cfg.head_dim, "weight")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            dim: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn weightless(dim: usize, eps: f32, device: &Device, dtype: DType) -> Self {
        Self {
            weight: None,
            dim,
            eps,
            device: device.clone(),
            dtype,
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Some(weight) => candle_nn::ops::rms_norm(&xs.contiguous()?, weight, self.eps),
            None => {
                let ones = Tensor::ones(self.dim, self.dtype, &self.device)?;
                candle_nn::ops::rms_norm(&xs.contiguous()?, &ones, self.eps)
            }
        }
    }
}

/// How the vision tower computes attention scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionImpl {
    /// Query-tiled `softmax(q k^T + mask) v`: the score matrix never exists at once.
    Tiled,
    /// Metal's fused scaled-dot-product kernel.
    MetalSdpa,
}

impl AttentionImpl {
    /// `Auto` takes the fused Metal kernel where there is one; `Eager` and
    /// `Tiled` both mean the tiled reference path; `Fused` is an error where no
    /// fused kernel exists.
    pub fn resolve(
        request: inl_core::attention::AttentionRequest,
        device: &Device,
        head_dim: usize,
    ) -> inl_core::Result<Self> {
        use inl_core::attention::AttentionRequest;
        let metal_ok = device.is_metal() && [32, 64, 72, 80, 96, 128, 256, 512].contains(&head_dim);
        match request {
            AttentionRequest::Eager | AttentionRequest::Tiled => Ok(Self::Tiled),
            AttentionRequest::Auto => Ok(if metal_ok { Self::MetalSdpa } else { Self::Tiled }),
            AttentionRequest::Fused if metal_ok => Ok(Self::MetalSdpa),
            AttentionRequest::Fused => Err(inl_core::InlError::Config(format!(
                "attention = \"fused\" is unavailable for the vision tower on {:?} with head_dim {head_dim}",
                device.location()
            ))),
        }
    }
}

/// `softmax(q k^T + mask) v` with scaling 1.0 over `q, k, v: [B, H, T, D]`.
///
/// The additive key `mask` (`[B, 1, 1, T]`) keeps padding keys out of every
/// real row; rows of padding queries are unspecified (the pooler strips them).
fn masked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    kind: AttentionImpl,
) -> Result<Tensor> {
    let (_, heads, seq, _) = q.dims4()?;
    match kind {
        AttentionImpl::MetalSdpa => {
            let mask = mask.expand((q.dim(0)?, heads, seq, seq))?;
            candle_nn::ops::sdpa(q, k, v, Some(&mask), false, 1.0, 1.0)
        }
        AttentionImpl::Tiled => {
            let kt = k.transpose(2, 3)?.contiguous()?;
            let mut blocks = Vec::with_capacity(seq.div_ceil(ATTENTION_TILE));
            let mut start = 0;
            while start < seq {
                let rows = ATTENTION_TILE.min(seq - start);
                let q_tile = q.narrow(2, start, rows)?.contiguous()?;
                let scores = q_tile.matmul(&kt)?;
                let weights = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
                blocks.push(weights.matmul(v)?);
                start += rows;
            }
            if blocks.len() == 1 {
                Ok(blocks.pop().unwrap())
            } else {
                Tensor::cat(&blocks, 2)
            }
        }
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj.linear"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj.linear"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj.linear"))?,
            o_proj: linear_no_bias(q_out, cfg.hidden_size, vb.pp("o_proj.linear"))?,
            q_norm: RmsNorm::head_dim(cfg, vb.pp("q_norm"), true)?,
            k_norm: RmsNorm::head_dim(cfg, vb.pp("k_norm"), true)?,
            v_norm: RmsNorm::head_dim(cfg, vb.pp("v_norm"), false)?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rope: (&Tensor, &Tensor),
        mask: &Tensor,
        kind: AttentionImpl,
    ) -> Result<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let dim = self.head_dim;
        let q = hidden
            .apply(&self.q_proj)?
            .reshape((batch, seq, self.heads, dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = apply_multidimensional_rope(&q, rope.0, rope.1)?.transpose(1, 2)?;
        let k = hidden
            .apply(&self.k_proj)?
            .reshape((batch, seq, self.kv_heads, dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = apply_multidimensional_rope(&k, rope.0, rope.1)?.transpose(1, 2)?;
        let v = hidden
            .apply(&self.v_proj)?
            .reshape((batch, seq, self.kv_heads, dim))?;
        let v = self.v_norm.forward(&v)?.transpose(1, 2)?;

        let groups = self.heads / self.kv_heads;
        let repeat = |x: &Tensor| -> Result<Tensor> {
            if groups == 1 {
                return Ok(x.clone());
            }
            let x = x.unsqueeze(2)?;
            Tensor::cat(&vec![&x; groups], 2)?.reshape((batch, self.heads, seq, dim))
        };
        let (k, v) = (repeat(&k)?.contiguous()?, repeat(&v)?.contiguous()?);

        let attn = masked_attention(&q, &k, &v, mask, kind)?;
        attn.transpose(1, 2)?
            .reshape((batch, seq, self.heads * dim))?
            .apply(&self.o_proj)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj.linear"),
            )?,
            up_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj.linear"),
            )?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj.linear"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.apply(&self.gate_proj)?.gelu()?;
        (gate * xs.apply(&self.up_proj)?)?.apply(&self.down_proj)
    }
}

struct EncoderLayer {
    attention: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl EncoderLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::full(cfg, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::full(cfg, vb.pp("post_attention_layernorm"))?,
            pre_feedforward_layernorm: RmsNorm::full(cfg, vb.pp("pre_feedforward_layernorm"))?,
            post_feedforward_layernorm: RmsNorm::full(cfg, vb.pp("post_feedforward_layernorm"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        rope: (&Tensor, &Tensor),
        mask: &Tensor,
        kind: AttentionImpl,
    ) -> Result<Tensor> {
        let residual = hidden;
        let h = self.input_layernorm.forward(hidden)?;
        let h = self.attention.forward(&h, rope, mask, kind)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let hidden = (residual + h)?;
        let h = self.pre_feedforward_layernorm.forward(&hidden)?;
        let h = self.mlp.forward(&h)?;
        let h = self.post_feedforward_layernorm.forward(&h)?;
        hidden + h
    }
}

/// The pooled, `sqrt(hidden)`-scaled, padding-stripped vision features and how
/// many of them belong to each batch row.
pub struct VisionOutput {
    /// `[total_soft_tokens, hidden_size]` (the `vision_tower` output, before
    /// `embed_vision`).
    pub hidden: Tensor,
    /// Soft tokens per batch row (image or frame).
    pub tokens_per_row: Vec<usize>,
}

/// One image/frame's padded inputs.
pub struct VisionBatch<'a> {
    /// `[batch, max_patches, 3 * 16 * 16]`, padded rows zero.
    pub pixel_values: &'a Tensor,
    /// `[batch, max_patches, 2]` `(x, y)`, padded rows `(-1, -1)`.
    pub position_ids: &'a Tensor,
    /// `[batch, max_patches]`, 1 for a padding patch.
    pub padding: &'a Tensor,
    /// Soft tokens the pooler should produce per image (the token budget).
    pub output_length: usize,
}

pub struct VisionTower {
    patch_input_proj: Linear,
    position_table: Tensor,
    layers: Vec<EncoderLayer>,
    embed_projection: Linear,
    embed_norm: RmsNorm,
    root_hidden_size: f64,
    cfg: VisionConfig,
    device: Device,
    dtype: DType,
    attention: AttentionImpl,
}

impl VisionTower {
    /// Build from a `VarBuilder` rooted at the checkpoint root (it reads
    /// `vision_tower.*` and `embed_vision.*`). `text_hidden` is the width of a
    /// row of the text tower's `inputs_embeds` (the projection output size).
    pub fn new(cfg: VisionConfig, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let vision = vb.pp("vision_tower");
        let patch_input_proj = linear_no_bias(
            cfg.patch_size * cfg.patch_size * 3,
            cfg.hidden_size,
            vision.pp("patch_embedder.input_proj"),
        )?;
        let position_table = vision.pp("patch_embedder").get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| EncoderLayer::new(&cfg, vision.pp(format!("encoder.layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let embed_projection = linear_no_bias(
            cfg.hidden_size,
            text_hidden,
            vb.pp("embed_vision.embedding_projection"),
        )?;
        let embed_norm =
            RmsNorm::weightless(cfg.hidden_size, cfg.rms_norm_eps as f32, &device, dtype);
        Ok(Self {
            patch_input_proj,
            position_table,
            layers,
            embed_projection,
            embed_norm,
            root_hidden_size: (cfg.hidden_size as f64).sqrt(),
            attention: AttentionImpl::Tiled,
            cfg,
            device,
            dtype,
        })
    }

    /// Choose the attention implementation. A tower starts on the tiled
    /// reference path; the engine passes the caller's request (`Auto` takes the
    /// fused kernel where one exists).
    pub fn with_attention(
        mut self,
        request: inl_core::attention::AttentionRequest,
    ) -> inl_core::Result<Self> {
        self.attention = AttentionImpl::resolve(request, &self.device, self.cfg.head_dim)?;
        Ok(self)
    }

    pub fn attention_impl(&self) -> AttentionImpl {
        self.attention
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// `x` and `y` position tables summed per patch, zero on padding rows.
    /// `positions` is `position_ids` as f32; `keep` is `1 - padding`,
    /// `[batch, patches, 1]` in the model dtype.
    fn position_embeddings(&self, positions: &Tensor, keep: &Tensor) -> Result<Tensor> {
        let (batch, patches, _) = positions.dims3()?;
        // Padding rows are (-1, -1): clamp them onto row 0; `keep` zeroes them.
        let index = |axis: usize| -> Result<Tensor> {
            positions
                .narrow(2, axis, 1)?
                .flatten_all()?
                .relu()?
                .to_dtype(DType::U32)
        };
        let x_table = self.position_table.narrow(0, 0, 1)?.squeeze(0)?;
        let y_table = self.position_table.narrow(0, 1, 1)?.squeeze(0)?;
        let x_emb = x_table.index_select(&index(0)?, 0)?;
        let y_emb = y_table.index_select(&index(1)?, 0)?;
        (x_emb + y_emb)?
            .reshape((batch, patches, self.cfg.hidden_size))?
            .broadcast_mul(keep)
    }

    /// Axial RoPE tables `[batch, seq, head_dim]` from f32 `positions`, laid
    /// out `[freq_x, freq_x, freq_y, freq_y]`.
    fn rope(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let spatial = self.cfg.head_dim / 2;
        let theta = self.cfg.rope_parameters.rope_theta as f32;
        let inv_freq: Vec<f32> = (0..spatial / 2)
            .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / spatial as f32))
            .collect();
        let freq = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, 1, freq), positions.device())?;
        let x = positions.narrow(2, 0, 1)?.broadcast_mul(&inv_freq)?;
        let y = positions.narrow(2, 1, 1)?.broadcast_mul(&inv_freq)?;
        let (x_cos, x_sin, y_cos, y_sin) = (x.cos()?, x.sin()?, y.cos()?, y.sin()?);
        let cos = Tensor::cat(&[&x_cos, &x_cos, &y_cos, &y_cos], 2)?.to_dtype(self.dtype)?;
        let sin = Tensor::cat(&[&x_sin, &x_sin, &y_sin, &y_sin], 2)?.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }

    /// Additive key mask `[batch, 1, 1, seq]`: `MASKED` on padding keys.
    fn attention_mask(&self, padding: &Tensor) -> Result<Tensor> {
        let (batch, seq) = padding.dims2()?;
        padding
            .to_dtype(DType::F32)?
            .affine(MASKED as f64, 0.0)?
            .reshape((batch, 1, 1, seq))?
            .to_dtype(self.dtype)
    }

    /// Run the encoder and pooler, then `embed_vision`.
    pub fn forward(&self, batch: VisionBatch<'_>) -> Result<VisionOutput> {
        let (batch_size, patches, _) = batch.pixel_values.dims3()?;
        let positions = batch.position_ids.to_dtype(DType::F32)?;
        let keep = batch
            .padding
            .to_dtype(DType::F32)?
            .affine(-1.0, 1.0)?
            .unsqueeze(2)?
            .to_dtype(self.dtype)?;
        let scaled = batch.pixel_values.affine(2.0, -1.0)?;
        let mut hidden = scaled.apply(&self.patch_input_proj)?;
        hidden = (hidden + self.position_embeddings(&positions, &keep)?)?;

        let (cos, sin) = self.rope(&positions)?;
        let mask = self.attention_mask(batch.padding)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, (&cos, &sin), &mask, self.attention)?;
        }

        let pooled = self.pool(
            &hidden,
            batch.position_ids,
            batch.padding,
            batch.output_length,
            batch_size,
            patches,
        )?;
        Ok(VisionOutput {
            hidden: pooled.tensor,
            tokens_per_row: pooled.tokens_per_row,
        })
    }

    /// `embed_vision`: a weight-free RMSNorm followed by the projection into the
    /// text tower's space. `hidden` is `[n, hidden_size]` (a `vision_tower`
    /// output).
    pub fn embed_vision(&self, hidden: &Tensor) -> Result<Tensor> {
        let normalized = self.embed_norm.forward(&hidden.to_dtype(self.dtype)?)?;
        normalized.apply(&self.embed_projection)
    }

    /// Average each `k x k` block of patches into one soft token, scale by
    /// `sqrt(hidden_size)` and keep the tokens that received patches. The slot
    /// layout comes from the (small) position ids on the host; the hidden
    /// states stay on the device (`index_add` + `index_select`).
    fn pool(
        &self,
        hidden: &Tensor,
        position_ids: &Tensor,
        padding: &Tensor,
        output_length: usize,
        batch: usize,
        patches: usize,
    ) -> Result<PooledOutput> {
        let dim = self.cfg.hidden_size;
        let k = self.cfg.pooling_kernel_size;

        let pad_host: Vec<f32> = padding.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let pos_host: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;

        // Target slot of every patch; padding and out-of-budget patches go to
        // a trailing discard slot.
        let discard = (batch * output_length) as u32;
        let mut slots = Vec::with_capacity(batch * patches);
        let mut counts = vec![0u32; batch * output_length];
        for row in 0..batch {
            let mut max_x = 0i64;
            for p in 0..patches {
                max_x = max_x.max(pos_host[(row * patches + p) * 2].max(0));
            }
            let ncols = ((max_x + 1) / k as i64) as usize;
            for p in 0..patches {
                let x = pos_host[(row * patches + p) * 2].max(0) as usize;
                let y = pos_host[(row * patches + p) * 2 + 1].max(0) as usize;
                let idx = (x / k) + ncols * (y / k);
                if idx >= output_length {
                    slots.push(discard);
                    continue;
                }
                counts[row * output_length + idx] += 1;
                slots.push(if pad_host[row * patches + p] != 0.0 {
                    discard
                } else {
                    (row * output_length + idx) as u32
                });
            }
        }
        let mut selected = Vec::new();
        let mut tokens_per_row = Vec::with_capacity(batch);
        for row in 0..batch {
            let before = selected.len();
            for idx in 0..output_length {
                if counts[row * output_length + idx] > 0 {
                    selected.push((row * output_length + idx) as u32);
                }
            }
            tokens_per_row.push(selected.len() - before);
        }

        let device = hidden.device();
        let slots = Tensor::from_vec(slots, batch * patches, device)?;
        let selected_len = selected.len();
        let selected = Tensor::from_vec(selected, selected_len, device)?;
        let sums = Tensor::zeros((batch * output_length + 1, dim), DType::F32, device)?.index_add(
            &slots,
            &hidden
                .to_dtype(DType::F32)?
                .reshape((batch * patches, dim))?,
            0,
        )?;
        let scale = self.root_hidden_size / (k * k) as f64;
        let tensor = sums.index_select(&selected, 0)?.affine(scale, 0.0)?;
        Ok(PooledOutput {
            tensor,
            tokens_per_row,
        })
    }
}

struct PooledOutput {
    tensor: Tensor,
    tokens_per_row: Vec<usize>,
}

/// Apply the axial 2-D RoPE: split the head into two halves (x-part, y-part),
/// rotate each with its own frequencies, and concatenate.
fn apply_multidimensional_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let part = dim / 2;
    let x0 = x.narrow(D::Minus1, 0, part)?;
    let x1 = x.narrow(D::Minus1, part, part)?;
    let c0 = cos.narrow(D::Minus1, 0, part)?.unsqueeze(2)?;
    let s0 = sin.narrow(D::Minus1, 0, part)?.unsqueeze(2)?;
    let c1 = cos.narrow(D::Minus1, part, part)?.unsqueeze(2)?;
    let s1 = sin.narrow(D::Minus1, part, part)?.unsqueeze(2)?;
    let y0 = apply_rotary(&x0, &c0, &s0)?;
    let y1 = apply_rotary(&x1, &c1, &s1)?;
    Tensor::cat(&[y0, y1], D::Minus1)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_config_validates_the_checkpoint() {
        let bytes = br#"{"vision_config": {
            "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16,
            "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64,
            "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
            "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 10240,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"},
            "standardize": false, "use_clipped_linears": false
        }}"#;
        let cfg = VisionConfig::parse(bytes).unwrap();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 16);
    }

    /// Random q/k/v over `seq` tokens whose last `seq - real` keys are padding.
    fn attention_case(
        device: &Device,
        seq: usize,
        real: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let (heads, dim) = (4, 64);
        let mk = || Tensor::randn(0f32, 1f32, (1, heads, seq, dim), device).unwrap();
        let mut mask = vec![0f32; seq];
        mask[real..].fill(MASKED);
        let mask = Tensor::from_vec(mask, (1, 1, 1, seq), device).unwrap();
        (mk(), mk(), mk(), mask)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn tiled_attention_ignores_padding_keys() {
        let device = Device::Cpu;
        let (q, k, v, mask) = attention_case(&device, 40, 31);
        let tiled = masked_attention(&q, &k, &v, &mask, AttentionImpl::Tiled).unwrap();
        // Same result when the padding keys are physically removed.
        let kr = k.narrow(2, 0, 31).unwrap().contiguous().unwrap();
        let vr = v.narrow(2, 0, 31).unwrap().contiguous().unwrap();
        let scores = q.matmul(&kr.transpose(2, 3).unwrap()).unwrap();
        let want = candle_nn::ops::softmax_last_dim(&scores)
            .unwrap()
            .matmul(&vr)
            .unwrap();
        assert!(
            max_diff(
                &tiled.narrow(2, 0, 31).unwrap(),
                &want.narrow(2, 0, 31).unwrap()
            ) < 1e-4
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_metal_attention_matches_tiled_on_real_rows() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let (q, k, v, mask) = attention_case(&device, 300, 211);
        let tiled = masked_attention(&q, &k, &v, &mask, AttentionImpl::Tiled).unwrap();
        let fused = masked_attention(&q, &k, &v, &mask, AttentionImpl::MetalSdpa).unwrap();
        let diff = max_diff(
            &tiled.narrow(2, 0, 211).unwrap(),
            &fused.narrow(2, 0, 211).unwrap(),
        );
        assert!(diff < 1e-3, "fused attention drifted from tiled by {diff}");
    }

    #[test]
    fn attention_impl_follows_the_device_and_request() {
        use inl_core::attention::AttentionRequest;
        let cpu = Device::Cpu;
        assert_eq!(
            AttentionImpl::resolve(AttentionRequest::Auto, &cpu, 64).unwrap(),
            AttentionImpl::Tiled
        );
        assert_eq!(
            AttentionImpl::resolve(AttentionRequest::Eager, &cpu, 64).unwrap(),
            AttentionImpl::Tiled,
            "eager keeps the reference tiled path"
        );
        assert!(AttentionImpl::resolve(AttentionRequest::Fused, &cpu, 64).is_err());
    }
}
