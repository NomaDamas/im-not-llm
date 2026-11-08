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

    fn forward(&self, hidden: &Tensor, rope: (&Tensor, &Tensor), mask: &Tensor) -> Result<Tensor> {
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
        let kt = k.transpose(2, 3)?.contiguous()?;

        // Scaling is 1.0 in the reference; scores are tiled over queries so the
        // [batch, heads, seq, seq] matrix never exists at once.
        let mut blocks = Vec::with_capacity(seq.div_ceil(ATTENTION_TILE));
        let mut start = 0;
        while start < seq {
            let rows = ATTENTION_TILE.min(seq - start);
            let q_tile = q.narrow(2, start, rows)?.contiguous()?;
            let scores = q_tile.matmul(&kt)?;
            let mask_tile = mask.narrow(2, start, rows)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask_tile)?)?;
            blocks.push(weights.matmul(&v)?);
            start += rows;
        }
        let attn = if blocks.len() == 1 {
            blocks.pop().unwrap()
        } else {
            Tensor::cat(&blocks, 2)?
        };
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

    fn forward(&self, hidden: &Tensor, rope: (&Tensor, &Tensor), mask: &Tensor) -> Result<Tensor> {
        let residual = hidden;
        let h = self.input_layernorm.forward(hidden)?;
        let h = self.attention.forward(&h, rope, mask)?;
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
            cfg,
            device,
            dtype,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    fn position_embeddings(&self, position_ids: &Tensor, padding: &Tensor) -> Result<Tensor> {
        let (batch, patches, _) = position_ids.dims3()?;
        let ids: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let mut x_idx = Vec::with_capacity(batch * patches);
        let mut y_idx = Vec::with_capacity(batch * patches);
        for row in 0..batch * patches {
            x_idx.push(ids[row * 2].max(0) as u32);
            y_idx.push(ids[row * 2 + 1].max(0) as u32);
        }
        let x = Tensor::from_vec(x_idx, batch * patches, &self.device)?;
        let y = Tensor::from_vec(y_idx, batch * patches, &self.device)?;
        let x_table = self.position_table.narrow(0, 0, 1)?.squeeze(0)?;
        let y_table = self.position_table.narrow(0, 1, 1)?.squeeze(0)?;
        let x_emb = x_table
            .index_select(&x, 0)?
            .reshape((batch, patches, self.cfg.hidden_size))?;
        let y_emb = y_table
            .index_select(&y, 0)?
            .reshape((batch, patches, self.cfg.hidden_size))?;
        let positions = (x_emb + y_emb)?;

        let pad: Vec<f32> = padding.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let keep: Vec<f32> = pad.iter().map(|&p| 1.0 - p).collect();
        let keep = Tensor::from_vec(keep, (batch, patches, 1), &self.device)?
            .to_dtype(self.dtype)?
            .broadcast_as((batch, patches, self.cfg.hidden_size))?;
        positions.broadcast_mul(&keep)
    }

    fn rope(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dim = self.cfg.head_dim;
        let spatial = dim / 2;
        let theta = self.cfg.rope_parameters.rope_theta as f32;
        let inv_freq: Vec<f32> = (0..spatial / 2)
            .map(|i| {
                let exponent = (2 * i) as f32 / spatial as f32;
                1.0f32 / theta.powf(exponent)
            })
            .collect();
        let freq = inv_freq.len();

        let (batch, seq, _) = position_ids.dims3()?;
        let ids: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;
        let mut cos = vec![0f32; batch * seq * dim];
        let mut sin = vec![0f32; batch * seq * dim];
        for row in 0..batch * seq {
            let x = ids[row * 2] as f32;
            let y = ids[row * 2 + 1] as f32;
            let base = row * dim;
            for j in 0..freq {
                let cx = x * inv_freq[j];
                let cy = y * inv_freq[j];
                // [freq_h, freq_h, freq_w, freq_w] recomposition.
                cos[base + j] = cx.cos();
                cos[base + freq + j] = cx.cos();
                cos[base + 2 * freq + j] = cy.cos();
                cos[base + 3 * freq + j] = cy.cos();
                sin[base + j] = cx.sin();
                sin[base + freq + j] = cx.sin();
                sin[base + 2 * freq + j] = cy.sin();
                sin[base + 3 * freq + j] = cy.sin();
            }
        }
        let cos = Tensor::from_vec(cos, (batch, seq, dim), &self.device)?.to_dtype(self.dtype)?;
        let sin = Tensor::from_vec(sin, (batch, seq, dim), &self.device)?.to_dtype(self.dtype)?;
        Ok((cos, sin))
    }

    fn attention_mask(&self, padding: &Tensor) -> Result<Tensor> {
        let (batch, seq) = padding.dims2()?;
        let pad: Vec<f32> = padding.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut mask = vec![0f32; batch * seq * seq];
        for row in 0..batch {
            for key in 0..seq {
                let value = if pad[row * seq + key] != 0.0 {
                    MASKED
                } else {
                    0.0
                };
                let base = row * seq * seq + key;
                for query in 0..seq {
                    mask[base + query * seq] = value;
                }
            }
        }
        Tensor::from_vec(mask, (batch, 1, seq, seq), &self.device)?.to_dtype(self.dtype)
    }

    /// Run the encoder and pooler, then `embed_vision`.
    pub fn forward(&self, batch: VisionBatch<'_>) -> Result<VisionOutput> {
        let (batch_size, patches, _) = batch.pixel_values.dims3()?;
        let scaled = batch.pixel_values.affine(2.0, -1.0)?;
        let mut hidden = scaled.apply(&self.patch_input_proj)?;
        hidden = (hidden + self.position_embeddings(batch.position_ids, batch.padding)?)?;

        let (cos, sin) = self.rope(batch.position_ids)?;
        let mask = self.attention_mask(batch.padding)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, (&cos, &sin), &mask)?;
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

        let hidden_host: Vec<f32> = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let pad_host: Vec<f32> = padding.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let pos_host: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;

        let mut pooled = vec![0f32; batch * output_length * dim];
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
                    continue;
                }
                counts[row * output_length + idx] += 1;
                // Padding rows carry zero hidden state already; skip their add.
                if pad_host[row * patches + p] != 0.0 {
                    continue;
                }
                let src = (row * patches + p) * dim;
                let dst = (row * output_length + idx) * dim;
                for c in 0..dim {
                    pooled[dst + c] += hidden_host[src + c] / (k * k) as f32;
                }
            }
        }

        // Scale by sqrt(hidden_size) in float32, then strip per-row by mask.
        let scale = self.root_hidden_size as f32;
        let mut rows = Vec::with_capacity(batch);
        let mut tokens_per_row = Vec::with_capacity(batch);
        for row in 0..batch {
            let mut tokens = 0usize;
            for idx in 0..output_length {
                if counts[row * output_length + idx] > 0 {
                    tokens += 1;
                }
            }
            let mut selected = Vec::with_capacity(tokens * dim);
            for idx in 0..output_length {
                if counts[row * output_length + idx] > 0 {
                    let base = (row * output_length + idx) * dim;
                    selected.extend(pooled[base..base + dim].iter().map(|v| v * scale));
                }
            }
            tokens_per_row.push(tokens);
            rows.push(Tensor::from_vec(selected, (tokens, dim), &self.device)?);
        }
        let stacked = if rows.len() == 1 {
            rows.pop().unwrap()
        } else {
            Tensor::cat(&rows, 0)?
        };
        Ok(PooledOutput {
            tensor: stacked,
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
}
