//! The Qwen3.5 text tower (`Qwen3_5TextModel` in `transformers`) on candle:
//! hybrid gated-delta-net (linear attention) and gated full-attention layers.
//!
//! Only what an embedder needs: one forward pass over right-padded rows, no
//! cache. Right padding keeps the real tokens exact in both layer kinds: the
//! gated-delta rule and its convolution are causal, so trailing pad tokens
//! never reach an earlier position, and full attention masks pad keys.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use inl_core::{InlError, Result as InlResult};
use serde::Deserialize;

/// Additive mask for a key that must not be attended (finite, so a fully
/// masked pad row stays finite).
const MASK_VALUE: f64 = -1e4;

/// Query rows per block of the tiled attention.
const QUERY_TILE: usize = 256;

/// Head sizes candle's Metal SDPA kernel is instantiated for.
const SDPA_HEAD_DIMS: [usize; 8] = [32, 64, 72, 80, 96, 128, 256, 512];

/// Largest f32 head size whose SDPA tile fits Metal's 32 KiB of threadgroup
/// memory: the f32 kernel for 256 is instantiated but its pipeline cannot be
/// created ("Failed to create pipeline"), and 512 is refused outright.
const SDPA_MAX_F32_HEAD_DIM: usize = 128;

/// Sequence block of the chunked gated-delta rule (the reference's `chunk_size`).
const CHUNK: usize = 64;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}

fn default_rope_type() -> String {
    "default".to_string()
}

fn one() -> f64 {
    1.0
}

fn yes() -> bool {
    true
}

/// `model_type` of the checkpoint's `config.json`.
pub const MODEL_TYPE: &str = "qwen3_5";

/// `text_config` of a Qwen3.5 checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub layer_types: Vec<String>,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    #[serde(default = "yes")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub attention_bias: bool,
    /// Late-interaction checkpoints run full attention bidirectionally.
    #[serde(default = "yes")]
    pub is_causal: bool,
    pub rope_parameters: RopeParameters,
    pub max_position_embeddings: usize,
}

#[derive(Deserialize)]
struct Outer {
    text_config: Config,
}

impl Config {
    /// A `config.json` with a nested `text_config` (`qwen3_5`) or a flat
    /// text config (`qwen3_5_text`).
    pub fn parse(bytes: &[u8]) -> InlResult<Self> {
        let config = serde_json::from_slice::<Outer>(bytes)
            .map(|outer| outer.text_config)
            .or_else(|_| serde_json::from_slice::<Config>(bytes))
            .map_err(|error| {
                InlError::Inference(format!(
                    "native late model config.json is not a Qwen3.5 config: {error}"
                ))
            })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> InlResult<()> {
        let fail = |what: String| {
            Err(InlError::Inference(format!(
                "native late model config.json: {what}"
            )))
        };
        if self.layer_types.len() != self.num_hidden_layers {
            return fail(format!(
                "{} layer_types for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if let Some(kind) = self
            .layer_types
            .iter()
            .find(|kind| *kind != "linear_attention" && *kind != "full_attention")
        {
            return fail(format!("unsupported layer type {kind}"));
        }
        if self.hidden_act != "silu" {
            return fail(format!("unsupported hidden_act {}", self.hidden_act));
        }
        if !self.attn_output_gate || self.attention_bias {
            return fail("only gated attention without bias is supported".to_string());
        }
        if self.rope_parameters.rope_type != "default" {
            return fail(format!(
                "unsupported rope_type {}",
                self.rope_parameters.rope_type
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            return fail("head counts do not divide".to_string());
        }
        if self.rotary_dim() == 0 || !self.rotary_dim().is_multiple_of(2) {
            return fail(format!(
                "rotary dimension {} is not even",
                self.rotary_dim()
            ));
        }
        Ok(())
    }

    fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
}

/// How full attention computes scores; both compute the same function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// candle's fused SDPA kernel (Metal); the score matrix never exists.
    Sdpa,
    /// Matmul / mask / softmax / matmul over blocks of query rows.
    Tiled,
}

impl AttentionBackend {
    pub fn resolve(device: &Device, dtype: DType, head_dim: usize) -> Self {
        let fits = dtype != DType::F32 || head_dim <= SDPA_MAX_F32_HEAD_DIM;
        if device.is_metal() && SDPA_HEAD_DIMS.contains(&head_dim) && fits {
            Self::Sdpa
        } else {
            Self::Tiled
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Sdpa => "fused",
            Self::Tiled => "tiled",
        }
    }
}

/// Zero-centred RMSNorm: `x / rms(x) * (1 + w)`. `1 + w` is folded at load.
struct RmsNorm {
    scale: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints_dtype(dim, "weight", Default::default(), DType::F32)?;
        Ok(Self {
            scale: (weight + 1.0)?.to_dtype(vb.dtype())?,
            eps: eps as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&xs.contiguous()?, &self.scale, self.eps)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down: candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(xs)?.silu()? * self.up.forward(xs)?)?;
        self.down.forward(&gated)
    }
}

/// Per-row facts of one forward pass shared by every layer.
pub struct Batch {
    /// Additive key mask, `[B, 1, 1, T]` (bidirectional) or `[B, 1, T, T]`
    /// (causal), in the model dtype.
    mask: Tensor,
    cos: Tensor,
    sin: Tensor,
}

struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    backend: AttentionBackend,
}

impl FullAttention {
    fn load(cfg: &Config, vb: VarBuilder, backend: AttentionBackend) -> Result<Self> {
        let (heads, kv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(cfg.hidden_size, heads * hd * 2, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(cfg.hidden_size, kv * hd, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(cfg.hidden_size, kv * hd, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(heads * hd, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: RmsNorm::load(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads,
            kv_heads: kv,
            head_dim: hd,
            rotary_dim: cfg.rotary_dim(),
            backend,
        })
    }

    /// Rotary embedding on the first `rotary_dim` features of every head
    /// (the half-split rotation of `rotate_half`); the rest pass through.
    fn rope(&self, x: &Tensor, batch: &Batch) -> Result<Tensor> {
        let rotated = candle_nn::rotary_emb::rope(
            &x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?,
            &batch.cos,
            &batch.sin,
        )?;
        if self.rotary_dim == self.head_dim {
            return Ok(rotated);
        }
        let pass = x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
        Tensor::cat(&[&rotated, &pass], D::Minus1)
    }

    fn forward(&self, xs: &Tensor, batch: &Batch) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let hd = self.head_dim;
        let projected = self
            .q_proj
            .forward(xs)?
            .reshape((b, t, self.heads, 2 * hd))?;
        let q = projected.narrow(3, 0, hd)?;
        let gate = projected
            .narrow(3, hd, hd)?
            .reshape((b, t, self.heads * hd))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(xs)?
                    .reshape((b, t, self.kv_heads, hd))?,
            )?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b, t, self.kv_heads, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.rope(&q, batch)?;
        let k = self.rope(&k, batch)?;
        let scale = (hd as f64).powf(-0.5);
        let n_rep = self.heads / self.kv_heads;
        let out = match self.backend {
            AttentionBackend::Sdpa if t > 1 => {
                let mask = batch.mask.broadcast_as((b, self.heads, t, t))?;
                candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, scale as f32, 1.0)?
            }
            _ => tiled(&q, &k, &v, &batch.mask, scale, n_rep)?,
        };
        let out = out.transpose(1, 2)?.reshape((b, t, self.heads * hd))?;
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv, n_rep, t, d))?
        .reshape((b, kv * n_rep, t, d))
}

/// Attention over blocks of query rows: scratch is bounded by
/// `B x heads x QUERY_TILE x T`.
fn tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    n_rep: usize,
) -> Result<Tensor> {
    let t = q.dim(2)?;
    let k = repeat_kv(k, n_rep)?;
    let v = repeat_kv(v, n_rep)?;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let q = (q * scale)?;
    let causal = mask.dim(2)? > 1;
    let mut blocks = Vec::with_capacity(t.div_ceil(QUERY_TILE));
    let mut start = 0;
    while start < t {
        let rows = QUERY_TILE.min(t - start);
        let scores = q.narrow(2, start, rows)?.matmul(&kt)?;
        let rows_mask = if causal {
            mask.narrow(2, start, rows)?
        } else {
            mask.clone()
        };
        let scores = scores.broadcast_add(&rows_mask)?;
        blocks.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?);
        start += rows;
    }
    if blocks.len() == 1 {
        Ok(blocks.swap_remove(0))
    } else {
        Tensor::cat(&blocks, 2)
    }
}

/// Gated DeltaNet: causal depthwise convolution, then the gated delta rule
/// over the sequence, a gated RMSNorm and the output projection.
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    /// Depthwise kernel, `[channels, kernel]`.
    conv: Tensor,
    /// `-exp(A_log)` per value head, f32.
    neg_a: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    eps: f64,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl GatedDeltaNet {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (nk, nv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads);
        let (dk, dv) = (cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let conv_dim = 2 * nk * dk + nv * dv;
        let kernel = cfg.linear_conv_kernel_dim;
        let f32_param = |name: &str, len: usize| {
            vb.get_with_hints_dtype(len, name, Default::default(), DType::F32)
        };
        Ok(Self {
            in_proj_qkv: candle_nn::linear_no_bias(
                cfg.hidden_size,
                conv_dim,
                vb.pp("in_proj_qkv"),
            )?,
            in_proj_z: candle_nn::linear_no_bias(cfg.hidden_size, nv * dv, vb.pp("in_proj_z"))?,
            in_proj_b: candle_nn::linear_no_bias(cfg.hidden_size, nv, vb.pp("in_proj_b"))?,
            in_proj_a: candle_nn::linear_no_bias(cfg.hidden_size, nv, vb.pp("in_proj_a"))?,
            out_proj: candle_nn::linear_no_bias(nv * dv, cfg.hidden_size, vb.pp("out_proj"))?,
            conv: vb
                .get((conv_dim, 1, kernel), "conv1d.weight")?
                .reshape((conv_dim, kernel))?,
            neg_a: f32_param("A_log", nv)?.exp()?.neg()?,
            dt_bias: f32_param("dt_bias", nv)?,
            norm_weight: vb.get(dv, "norm.weight")?,
            eps: cfg.rms_norm_eps,
            key_heads: nk,
            value_heads: nv,
            key_dim: dk,
            value_dim: dv,
        })
    }

    /// `silu(conv1d(x))` with left zero padding, cut to the input length;
    /// `x` is `[B, T, C]`.
    fn causal_conv(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, t, channels) = xs.dims3()?;
        let kernel = self.conv.dim(1)?;
        let padded = xs.pad_with_zeros(1, kernel - 1, 0)?;
        let mut out: Option<Tensor> = None;
        for tap in 0..kernel {
            let weight = self.conv.narrow(1, tap, 1)?.reshape(channels)?;
            let term = padded.narrow(1, tap, t)?.broadcast_mul(&weight)?;
            out = Some(match out {
                Some(sum) => (sum + term)?,
                None => term,
            });
        }
        out.expect("a kernel has at least one tap").silu()
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let (nk, nv, dk, dv) = (
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
        );
        let mixed = self.causal_conv(&self.in_proj_qkv.forward(xs)?)?;
        let q = mixed.narrow(2, 0, nk * dk)?.reshape((b, t, nk, dk))?;
        let k = mixed.narrow(2, nk * dk, nk * dk)?.reshape((b, t, nk, dk))?;
        let v = mixed
            .narrow(2, 2 * nk * dk, nv * dv)?
            .reshape((b, t, nv, dv))?;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(xs)?.to_dtype(DType::F32)?)?;
        let a = self
            .in_proj_a
            .forward(xs)?
            .to_dtype(DType::F32)?
            .broadcast_add(&self.dt_bias)?;
        let g = softplus(&a)?.broadcast_mul(&self.neg_a)?;
        let (q, k) = if nv > nk {
            let rep = nv / nk;
            let interleave = |x: Tensor| -> Result<Tensor> {
                x.unsqueeze(3)?
                    .broadcast_as((b, t, nk, rep, dk))?
                    .reshape((b, t, nv, dk))
            };
            (interleave(q)?, interleave(k)?)
        } else {
            (q, k)
        };
        let core = chunk_gated_delta_rule(&q, &k, &v, &g, &beta)?;
        // Gated RMSNorm per value head: normalize in f32, scale in the model
        // dtype, gate with silu(z) in f32.
        let dtype = xs.dtype();
        let z = self.in_proj_z.forward(xs)?.reshape((b, t, nv, dv))?;
        let core = core.to_dtype(dtype)?.to_dtype(DType::F32)?;
        let variance = core.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = core.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let normed = normed.to_dtype(dtype)?.broadcast_mul(&self.norm_weight)?;
        let gated = (normed.to_dtype(DType::F32)? * z.to_dtype(DType::F32)?.silu()?)?;
        let out = gated.to_dtype(dtype)?.reshape((b, t, nv * dv))?;
        self.out_proj.forward(&out)
    }
}

/// `log(1 + exp(x))`, stable for large `|x|`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let tail = (x.abs()?.neg()?.exp()? + 1.0)?.log()?;
    x.relu()? + tail
}

/// `x / sqrt(sum(x^2) + 1e-6)` over the last dimension (the FLA l2norm).
fn l2norm(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&norm)
}

/// The chunked gated delta rule of the reference
/// (`torch_chunk_gated_delta_rule`, no initial state), in f32.
///
/// `q, k: [B, T, H, dk]`, `v: [B, T, H, dv]`, `g, beta: [B, T, H]` (f32).
/// Returns `[B, T, H, dv]` in f32.
///
/// Within a chunk the reference solves the unit lower-triangular system
/// `(I + A) X = Y` with `solve_triangular`; here `(I + A)^-1` is built by
/// block doubling (invert the diagonal blocks, then join pairs with
/// `-X22 L21 X11`), which is the same back substitution, done for every chunk
/// at once with batched matmuls.
fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let (b, t, h, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let device = q.device();
    let heads_first =
        |x: &Tensor| -> Result<Tensor> { x.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous() };
    let q = (l2norm(&heads_first(q)?)? * (dk as f64).powf(-0.5))?;
    let k = l2norm(&heads_first(k)?)?;
    let v = heads_first(v)?;
    let g = heads_first(g)?;
    let beta = heads_first(beta)?;

    let pad = (CHUNK - t % CHUNK) % CHUNK;
    let padded = t + pad;
    let chunks = padded / CHUNK;
    let pad_seq = |x: Tensor| -> Result<Tensor> {
        if pad == 0 {
            Ok(x)
        } else {
            x.pad_with_zeros(2, 0, pad)
        }
    };
    let (q, k, v, g, beta) = (
        pad_seq(q)?,
        pad_seq(k)?,
        pad_seq(v)?,
        pad_seq(g)?,
        pad_seq(beta)?,
    );
    let beta = beta.unsqueeze(D::Minus1)?;
    let v_beta = v.broadcast_mul(&beta)?;
    let k_beta = k.broadcast_mul(&beta)?;
    // Batch and heads share one leading dimension: `[B * H, chunks, C, d]`.
    let n = b * h;
    let blocks = |x: Tensor, d: usize| x.reshape((n, chunks, CHUNK, d));
    let q = blocks(q, dk)?;
    let k = blocks(k, dk)?;
    let k_beta = blocks(k_beta, dk)?;
    let v_beta = blocks(v_beta, dv)?;
    let cum = g.reshape((n, chunks, CHUNK))?.cumsum(2)?;

    let positions = Tensor::arange(0u32, CHUNK as u32, device)?;
    let rows = positions.reshape((CHUNK, 1))?;
    let cols = positions.reshape((1, CHUNK))?;
    let lower = rows.broadcast_ge(&cols)?.to_dtype(DType::F32)?;
    let strictly_lower = rows.broadcast_gt(&cols)?.to_dtype(DType::F32)?;
    // exp(cum_i - cum_j) on and below the diagonal, zero above. `cum` falls
    // along a chunk (`g <= 0`), so the kept differences are <= 0; clamping the
    // rest keeps exp from overflowing before the mask zeroes it.
    let diff = cum.unsqueeze(3)?.broadcast_sub(&cum.unsqueeze(2)?)?;
    let decay = diff.clamp(f64::MIN, 0.0)?.exp()?.broadcast_mul(&lower)?;

    let kt = k.transpose(2, 3)?.contiguous()?;
    let system = k_beta
        .matmul(&kt)?
        .mul(&decay)?
        .broadcast_mul(&strictly_lower)?;
    let inverse = unit_lower_inverse(&system.reshape((n * chunks, CHUNK, CHUNK))?)?
        .reshape((n, chunks, CHUNK, CHUNK))?;
    let cum_exp = cum.exp()?.unsqueeze(3)?;
    let new_values = inverse.matmul(&v_beta)?;
    let k_cumdecay = inverse.matmul(&k_beta.broadcast_mul(&cum_exp)?)?;
    let intra = q.matmul(&kt)?.mul(&decay)?;

    let last = cum.narrow(2, CHUNK - 1, 1)?;
    let q_decayed = q.broadcast_mul(&cum_exp)?;
    let k_decayed = k
        .broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(3)?)?
        .transpose(2, 3)?
        .contiguous()?;
    let chunk_decay = last.exp()?.unsqueeze(3)?;

    let mut state: Option<Tensor> = None;
    let mut outputs = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        let at = |x: &Tensor| x.narrow(1, chunk, 1)?.squeeze(1)?.contiguous();
        let values = at(&new_values)?;
        let (v_new, out) = match &state {
            None => {
                let out = at(&intra)?.matmul(&values)?;
                (values, out)
            }
            Some(state) => {
                let v_new = (values - at(&k_cumdecay)?.matmul(state)?)?;
                let out = (at(&q_decayed)?.matmul(state)? + at(&intra)?.matmul(&v_new)?)?;
                (v_new, out)
            }
        };
        outputs.push(out);
        if chunk + 1 < chunks {
            let update = at(&k_decayed)?.matmul(&v_new)?;
            let decay = at(&chunk_decay)?;
            state = Some(match &state {
                None => update,
                Some(state) => (state.broadcast_mul(&decay)? + update)?,
            });
        }
    }
    let out = if outputs.len() == 1 {
        outputs.swap_remove(0)
    } else {
        Tensor::cat(&outputs, 1)?
    };
    out.reshape((b, h, padded, dv))?
        .narrow(2, 0, t)?
        .transpose(1, 2)?
        .contiguous()
}

/// `(I + A)^-1` for strictly lower-triangular `A: [N, S, S]` (`S` a power of
/// two), by doubling the inverted diagonal blocks:
/// `[[X11, 0], [L21, X22]]^-1 = [[X11^-1, 0], [-X22^-1 L21 X11^-1, X22^-1]]`.
fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    let (n, s, _) = a.dims3()?;
    let device = a.device();
    // Inverted diagonal blocks of size `width`: `[N, S / width, width, width]`.
    let mut inverse = Tensor::ones((n, s, 1, 1), DType::F32, device)?;
    let mut width = 1;
    while width < s {
        let count = s / width;
        // Every `width x width` block of `A`: `[N, count * count, width, width]`.
        let all = a
            .reshape((n, count, width, count, width))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((n, count * count, width, width))?;
        let pairs = count / 2;
        let below: Vec<u32> = (0..pairs)
            .map(|pair| ((2 * pair + 1) * count + 2 * pair) as u32)
            .collect();
        let below = Tensor::from_vec(below, pairs, device)?;
        let l21 = all.index_select(&below, 1)?;
        let joined = inverse.reshape((n, pairs, 2, width, width))?;
        let x11 = joined.narrow(2, 0, 1)?.squeeze(2)?;
        let x22 = joined.narrow(2, 1, 1)?.squeeze(2)?;
        let off = x22.matmul(&l21)?.matmul(&x11)?.neg()?;
        let zeros = Tensor::zeros((n, pairs, width, width), DType::F32, device)?;
        let top = Tensor::cat(&[&x11, &zeros], 3)?;
        let bottom = Tensor::cat(&[&off, &x22], 3)?;
        inverse = Tensor::cat(&[&top, &bottom], 2)?;
        width *= 2;
    }
    inverse.reshape((n, s, s))
}

enum Mixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

struct Layer {
    mixer: Mixer,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl Layer {
    fn forward(&self, xs: &Tensor, batch: &Batch) -> Result<Tensor> {
        let normed = self.input_norm.forward(xs)?;
        let mixed = match &self.mixer {
            Mixer::Linear(net) => net.forward(&normed)?,
            Mixer::Full(attention) => attention.forward(&normed, batch)?,
        };
        let xs = (xs + mixed)?;
        let mlp = self.mlp.forward(&self.post_norm.forward(&xs)?)?;
        xs + mlp
    }
}

/// The text tower up to its final norm (`last_hidden_state`).
pub struct Qwen35Text {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// Rotary tables, `[positions, rotary_dim / 2]`, model dtype.
    cos: Tensor,
    sin: Tensor,
    causal: bool,
    backend: AttentionBackend,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl Qwen35Text {
    /// `vb` sits at the text model's root (`embed_tokens`, `layers`, `norm`).
    /// `positions` bounds the rotary table.
    pub fn load(cfg: &Config, vb: VarBuilder, positions: usize) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let backend = AttentionBackend::resolve(&device, dtype, cfg.head_dim);
        let embed = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let layers = cfg
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let vb = vb.pp(format!("layers.{index}"));
                let mixer = if kind == "linear_attention" {
                    Mixer::Linear(GatedDeltaNet::load(cfg, vb.pp("linear_attn"))?)
                } else {
                    Mixer::Full(FullAttention::load(cfg, vb.pp("self_attn"), backend)?)
                };
                Ok(Layer {
                    mixer,
                    mlp: Mlp::load(cfg, vb.pp("mlp"))?,
                    input_norm: RmsNorm::load(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("input_layernorm"),
                    )?,
                    post_norm: RmsNorm::load(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_attention_layernorm"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let (cos, sin) = rotary_tables(cfg, positions.max(1), &device, dtype)?;
        Ok(Self {
            embed,
            layers,
            norm,
            cos,
            sin,
            causal: cfg.is_causal,
            backend,
            device,
            dtype,
            hidden_size: cfg.hidden_size,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn positions(&self) -> usize {
        self.cos.dim(0).unwrap_or(0)
    }

    pub fn backend(&self) -> AttentionBackend {
        self.backend
    }

    /// `last_hidden_state` for right-padded `ids: [B, T]` whose rows hold
    /// `lengths` real tokens.
    pub fn forward(&self, ids: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (_, t) = ids.dims2()?;
        let batch = Batch {
            mask: self.mask(lengths, t)?,
            cos: self.cos.narrow(0, 0, t)?,
            sin: self.sin.narrow(0, 0, t)?,
        };
        let mut xs = self.embed.forward(ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &batch)?;
        }
        self.norm.forward(&xs)
    }

    /// Additive mask over keys past each row's length, plus the future when
    /// the checkpoint is causal.
    fn mask(&self, lengths: &[usize], t: usize) -> Result<Tensor> {
        let keys = Tensor::arange(0u32, t as u32, &self.device)?.reshape((1, 1, 1, t))?;
        let limits = Tensor::from_vec(
            lengths.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            (lengths.len(), 1, 1, 1),
            &self.device,
        )?;
        let padding = keys
            .broadcast_ge(&limits)?
            .to_dtype(self.dtype)?
            .affine(MASK_VALUE, 0.0)?;
        if !self.causal {
            return Ok(padding);
        }
        let queries = Tensor::arange(0u32, t as u32, &self.device)?.reshape((1, 1, t, 1))?;
        let future = keys
            .broadcast_gt(&queries)?
            .to_dtype(self.dtype)?
            .affine(MASK_VALUE, 0.0)?;
        future.broadcast_add(&padding)
    }
}

/// `cos` / `sin` of `position * inv_freq`, computed in f32 as the reference
/// does. Text inputs give all three M-RoPE axes the same position, so the
/// interleaved sections collapse to plain partial RoPE.
fn rotary_tables(
    cfg: &Config,
    positions: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let theta = cfg.rope_parameters.rope_theta as f32;
    let inv_freq: Vec<f32> = (0..dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?;
    let steps = Tensor::arange(0u32, positions as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((positions, 1))?;
    let freqs = steps.broadcast_mul(&inv_freq)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// The per-token recurrence of the reference
    /// (`torch_recurrent_gated_delta_rule`), the definition the chunked form
    /// must reproduce.
    #[allow(clippy::too_many_arguments)] // the reference's own signature
    fn recurrent(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        t: usize,
        dk: usize,
        dv: usize,
    ) -> Vec<f32> {
        let norm = |x: &[f32]| {
            let n = (x.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
            x.iter().map(|a| a / n).collect::<Vec<_>>()
        };
        let mut state = vec![0f32; dk * dv];
        let mut out = vec![0f32; t * dv];
        for i in 0..t {
            let qi: Vec<f32> = norm(&q[i * dk..(i + 1) * dk])
                .iter()
                .map(|a| a / (dk as f32).sqrt())
                .collect();
            let ki = norm(&k[i * dk..(i + 1) * dk]);
            let decay = g[i].exp();
            state.iter_mut().for_each(|s| *s *= decay);
            for c in 0..dv {
                let mem: f32 = (0..dk).map(|r| state[r * dv + c] * ki[r]).sum();
                let delta = (v[i * dv + c] - mem) * beta[i];
                for r in 0..dk {
                    state[r * dv + c] += ki[r] * delta;
                }
                out[i * dv + c] = (0..dk).map(|r| state[r * dv + c] * qi[r]).sum();
            }
        }
        out
    }

    #[test]
    fn chunked_delta_rule_matches_the_per_token_recurrence_across_chunks() {
        let device = Device::Cpu;
        let (t, dk, dv) = (CHUNK * 2 + 17, 8, 6);
        let q = pseudo(t * dk, 1);
        let k = pseudo(t * dk, 2);
        let v = pseudo(t * dv, 3);
        // Decays in (-0.3, 0]; betas in (0, 1).
        let g: Vec<f32> = pseudo(t, 4).iter().map(|x| -0.3 * (x + 0.5)).collect();
        let beta: Vec<f32> = pseudo(t, 5).iter().map(|x| x + 0.5).collect();
        let tensor = |data: &[f32], shape: &[usize]| {
            Tensor::from_vec(data.to_vec(), shape, &device).unwrap()
        };
        let out = chunk_gated_delta_rule(
            &tensor(&q, &[1, t, 1, dk]),
            &tensor(&k, &[1, t, 1, dk]),
            &tensor(&v, &[1, t, 1, dv]),
            &tensor(&g, &[1, t, 1]),
            &tensor(&beta, &[1, t, 1]),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let expected = recurrent(&q, &k, &v, &g, &beta, t, dk, dv);
        let worst = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-4,
            "chunked rule differs from the recurrence by {worst}"
        );
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        let device = Device::Cpu;
        let s = 16;
        let mut values = pseudo(2 * s * s, 9);
        for n in 0..2 {
            for i in 0..s {
                for j in i..s {
                    values[n * s * s + i * s + j] = 0.0;
                }
            }
        }
        let a = Tensor::from_vec(values, (2, s, s), &device).unwrap();
        let eye = Tensor::eye(s, DType::F32, &device).unwrap();
        let system = a.broadcast_add(&eye).unwrap();
        let product = system.matmul(&unit_lower_inverse(&a).unwrap()).unwrap();
        let worst = product
            .broadcast_sub(&eye)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(worst < 1e-4, "(I + A) X differs from I by {worst}");
    }
}
