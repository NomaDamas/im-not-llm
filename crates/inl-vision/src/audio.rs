//! The Gemma 4 audio path used by EmbeddingGemma 2, written against
//! `transformers` `Gemma4AudioFeatureExtractor` and `Gemma4AudioModel`
//! (Apache-2.0):
//!
//! * decoding: any container `ffmpeg` reads, resampled to 16 kHz mono f32;
//! * features: semicausal-padded 20 ms Hann frames every 10 ms, a 512-point
//!   FFT magnitude, 128 HTK mel bins and `ln(mel + 1e-3)`; at most 30 s;
//! * `audio_tower`: two stride-2 `3 x 3` conv + LayerNorm + ReLU blocks and a
//!   projection, then conformer layers (half-step feed-forward, local
//!   attention over the previous `attention_context_left - 1` frames with a
//!   sinusoidal relative-position term and a `tanh` logit cap, a causal
//!   depthwise light conv, half-step feed-forward) and `output_proj`;
//! * `embed_audio`: a weight-free RMSNorm and a projection into the text
//!   tower's space.
//!
//! Every frame of the tower sees only itself and earlier frames, so padding a
//! clip at the end (to batch it with longer ones) leaves its real rows as
//! they are; the padding rows are dropped.

use crate::video::{ensure_binary, ffmpeg_binary};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, linear_no_bias, LayerNorm, Linear, VarBuilder};
use inl_core::{InlError, Result as InlResult};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::Deserialize;
use std::process::{Command, Stdio};

/// Sample rate the extractor and tower expect.
pub const SAMPLE_RATE: usize = 16_000;
/// Samples per analysis frame (20 ms).
const FRAME_LENGTH: usize = 320;
/// Samples between frames (10 ms).
const HOP_LENGTH: usize = 160;
/// FFT size: the frame zero-padded to the next power of two.
const FFT_LENGTH: usize = 512;
/// Mel bins per frame.
pub const MEL_BINS: usize = 128;
/// Added to the mel energies before the log.
const MEL_FLOOR: f64 = 1e-3;
/// The extractor truncates a clip at 30 s.
pub const MAX_SAMPLES: usize = 30 * SAMPLE_RATE;
/// The extractor pads every clip to a multiple of this many samples.
const PAD_MULTIPLE: usize = 128;
/// Additive mask value for keys outside a frame's window or past the clip;
/// the reference `attention_invalid_logits_value`.
const MASKED: f32 = -1e9;
/// Epsilon of the RMSNorms the reference builds with the class default
/// rather than `rms_norm_eps` (feed-forward and per-layer norms).
const NORM_EPS: f64 = 1e-6;

/// Decode `bytes` (WAV, MP3, FLAC, ...) to 16 kHz mono f32 samples with
/// `ffmpeg`, keeping at most [`MAX_SAMPLES`].
pub fn decode(bytes: &[u8]) -> InlResult<Vec<f32>> {
    let ffmpeg = ffmpeg_binary();
    ensure_binary(&ffmpeg)?;
    let dir = tempfile::tempdir().map_err(InlError::Io)?;
    let input = dir.path().join("input");
    std::fs::write(&input, bytes)?;
    let output = Command::new(&ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string()])
        .args(["-f", "f32le", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| InlError::Inference(format!("failed to run {ffmpeg}: {error}")))?;
    if !output.status.success() {
        return Err(InlError::InvalidInput(format!(
            "ffmpeg could not decode the audio: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .take(MAX_SAMPLES)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(samples)
}

/// Log-mel features of one clip, `[frames, MEL_BINS]` row-major. Frames past
/// `valid` overlap the end padding: they are zero, and masked in the tower.
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
}

impl AudioFeatures {
    /// Soft tokens the tower produces for this clip: real rows after the two
    /// stride-2 subsampling blocks.
    pub fn tokens(&self) -> usize {
        self.valid.div_ceil(2).div_ceil(2)
    }
}

/// The mel filter bank `[FFT_LENGTH / 2 + 1, MEL_BINS]` (HTK scale, 0 to
/// 8 kHz, unnormalised triangles), as `transformers` `mel_filter_bank` builds it.
fn mel_filters() -> Vec<f64> {
    let bins = FFT_LENGTH / 2 + 1;
    let to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let (mel_min, mel_max) = (to_mel(0.0), to_mel(SAMPLE_RATE as f64 / 2.0));
    let points: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| to_hz(mel_min + (mel_max - mel_min) * i as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let nyquist = (SAMPLE_RATE / 2) as f64;
    let mut filters = vec![0f64; bins * MEL_BINS];
    for bin in 0..bins {
        let hz = nyquist * bin as f64 / (bins - 1) as f64;
        for mel in 0..MEL_BINS {
            let down = (hz - points[mel]) / (points[mel + 1] - points[mel]);
            let up = (points[mel + 2] - hz) / (points[mel + 2] - points[mel + 1]);
            filters[bin * MEL_BINS + mel] = down.min(up).max(0.0);
        }
    }
    filters
}

/// `Gemma4AudioFeatureExtractor` on one 16 kHz clip (already truncated to
/// [`MAX_SAMPLES`]). Errors when the clip is too short for one frame.
pub fn log_mel(samples: &[f32]) -> InlResult<AudioFeatures> {
    let samples = &samples[..samples.len().min(MAX_SAMPLES)];
    let padded_len = samples.len().div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
    // Semicausal: half a frame of zeros in front, so frame 0 centres on t = 0.
    let pad_left = FRAME_LENGTH / 2;
    let total = pad_left + padded_len;
    // Each frame reads `FRAME_LENGTH + 1` samples (the extra one feeds the
    // unused pre-emphasis), so that is the window `unfold` slides.
    if total < FRAME_LENGTH + 1 {
        return Err(InlError::InvalidInput(format!(
            "the audio is {} samples, too short for one {} ms frame",
            samples.len(),
            FRAME_LENGTH * 1000 / SAMPLE_RATE
        )));
    }
    let frames = (total - (FRAME_LENGTH + 1)) / HOP_LENGTH + 1;
    // A frame is real when its last sample is.
    let valid = (0..frames)
        .take_while(|&i| i * HOP_LENGTH + FRAME_LENGTH < pad_left + samples.len())
        .count();

    let window: Vec<f32> = (0..FRAME_LENGTH)
        .map(|n| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME_LENGTH as f64).cos()) as f32
        })
        .collect();
    let filters = mel_filters();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT_LENGTH);
    let bins = FFT_LENGTH / 2 + 1;
    let sample = |index: usize| -> f32 {
        index
            .checked_sub(pad_left)
            .and_then(|i| samples.get(i).copied())
            .unwrap_or(0.0)
    };

    let mut data = vec![0f32; frames * MEL_BINS];
    let mut buffer = vec![Complex::new(0f64, 0f64); FFT_LENGTH];
    let mut magnitude = vec![0f64; bins];
    for frame in 0..valid {
        let start = frame * HOP_LENGTH;
        for (n, slot) in buffer.iter_mut().enumerate() {
            let value = if n < FRAME_LENGTH {
                (sample(start + n) * window[n]) as f64
            } else {
                0.0
            };
            *slot = Complex::new(value, 0.0);
        }
        fft.process(&mut buffer);
        for (bin, slot) in magnitude.iter_mut().enumerate() {
            *slot = buffer[bin].norm();
        }
        let row = &mut data[frame * MEL_BINS..(frame + 1) * MEL_BINS];
        for (mel, out) in row.iter_mut().enumerate() {
            let energy: f64 = (0..bins)
                .map(|bin| magnitude[bin] * filters[bin * MEL_BINS + mel])
                .sum();
            *out = (energy + MEL_FLOOR).ln() as f32;
        }
    }
    Ok(AudioFeatures {
        data,
        frames,
        valid,
    })
}

/// The checkpoint's `audio_config`.
#[derive(Deserialize, Debug, Clone)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    attention_chunk_size: usize,
    attention_context_left: usize,
    attention_context_right: usize,
    attention_logit_cap: f64,
    conv_kernel_size: usize,
    hidden_act: candle_nn::Activation,
    residual_weight: f64,
    rms_norm_eps: f64,
    subsampling_conv_channels: [usize; 2],
    pub output_proj_dims: usize,
    use_clipped_linears: bool,
}

#[derive(Deserialize)]
struct CheckpointConfig {
    audio_config: Option<AudioConfig>,
}

impl AudioConfig {
    /// Parse the `audio_config` block of a checkpoint `config.json`; `None`
    /// when the checkpoint has no audio tower.
    pub fn parse(bytes: &[u8]) -> InlResult<Option<Self>> {
        let checkpoint: CheckpointConfig = serde_json::from_slice(bytes)
            .map_err(|error| InlError::Inference(format!("config.json audio_config: {error}")))?;
        let Some(config) = checkpoint.audio_config else {
            return Ok(None);
        };
        config.validate()?;
        Ok(Some(config))
    }

    fn validate(&self) -> InlResult<()> {
        let invalid = |what: String| {
            Err(InlError::Inference(format!(
                "unsupported audio config: {what}"
            )))
        };
        // The blocked attention of the reference reduces to "the previous
        // `context_left - 1` frames, relative distance `d` embedded at `d`"
        // only for this geometry.
        if self.attention_context_right != 0
            || self.attention_context_left < 2
            || self.attention_chunk_size + 1 != self.attention_context_left
        {
            return invalid(format!(
                "attention chunk {} / left {} / right {}",
                self.attention_chunk_size,
                self.attention_context_left,
                self.attention_context_right
            ));
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return invalid("attention head counts".to_string());
        }
        if self.hidden_act != candle_nn::Activation::Silu {
            return invalid(format!("hidden_act {:?}", self.hidden_act));
        }
        // The reference sizes `input_proj_linear` as `channels[0] / 4 *
        // channels[1]`, while the tensor it sees is `MEL_BINS / 4 *
        // channels[1]` wide: the two agree only when `channels[0]` is the
        // mel bin count.
        if self.subsampling_conv_channels[0] != MEL_BINS {
            return invalid(format!(
                "subsampling_conv_channels {:?}",
                self.subsampling_conv_channels
            ));
        }
        Ok(())
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Keys each frame attends to: itself and the frames before it.
    fn window(&self) -> usize {
        self.attention_context_left - 1
    }
}

/// `x * (mean(x^2) + eps)^-1/2 (* weight)`, computed in f32.
struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps: eps as f32,
        })
    }

    fn weightless(dim: usize, eps: f64, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(dim, dtype, device)?,
            eps: eps as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&xs.contiguous()?, &self.weight, self.eps)
    }
}

/// `Gemma4ClippableLinear`: clamp the input, bias-free linear, clamp the output.
struct ClippedLinear {
    linear: Linear,
    input: Option<(f32, f32)>,
    output: Option<(f32, f32)>,
}

impl ClippedLinear {
    fn new(cfg: &AudioConfig, inputs: usize, outputs: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear_no_bias(inputs, outputs, vb.pp("linear"))?;
        if !cfg.use_clipped_linears {
            return Ok(Self {
                linear,
                input: None,
                output: None,
            });
        }
        let scalar =
            |name: &str| -> Result<f32> { vb.get((), name)?.to_dtype(DType::F32)?.to_scalar() };
        Ok(Self {
            linear,
            input: Some((scalar("input_min")?, scalar("input_max")?)),
            output: Some((scalar("output_min")?, scalar("output_max")?)),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = match self.input {
            Some((min, max)) => xs.clamp(min, max)?,
            None => xs.clone(),
        };
        let ys = xs.apply(&self.linear)?;
        match self.output {
            Some((min, max)) => ys.clamp(min, max),
            None => Ok(ys),
        }
    }
}

/// One stride-2 `3 x 3` conv, channel LayerNorm and ReLU.
struct SubsampleLayer {
    conv: Tensor,
    norm: LayerNorm,
}

impl SubsampleLayer {
    fn new(inputs: usize, outputs: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: vb.get((outputs, inputs, 3, 3), "conv.weight")?,
            norm: LayerNorm::new_no_bias(vb.get(outputs, "norm.weight")?, eps),
        })
    }

    /// `[B, C, T, F] -> [B, T', F', C']`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs.conv2d(&self.conv, 1, 2, 1, 1)?;
        ys.permute((0, 2, 3, 1))?
            .contiguous()?
            .apply(&self.norm)?
            .relu()
    }
}

struct FeedForward {
    pre_norm: RmsNorm,
    up: ClippedLinear,
    down: ClippedLinear,
    post_norm: RmsNorm,
    residual_weight: f64,
}

impl FeedForward {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        Ok(Self {
            pre_norm: RmsNorm::new(hidden, NORM_EPS, vb.pp("pre_layer_norm"))?,
            up: ClippedLinear::new(cfg, hidden, hidden * 4, vb.pp("ffw_layer_1"))?,
            down: ClippedLinear::new(cfg, hidden * 4, hidden, vb.pp("ffw_layer_2"))?,
            post_norm: RmsNorm::new(hidden, NORM_EPS, vb.pp("post_layer_norm"))?,
            residual_weight: cfg.residual_weight,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.up.forward(&self.pre_norm.forward(xs)?)?.silu()?;
        let ys = self.post_norm.forward(&self.down.forward(&ys)?)?;
        ys.affine(self.residual_weight, 0.0)? + xs
    }
}

/// Local self-attention: frame `t` attends to frames `t - d`, `d < window`,
/// with logits `q_t . k_{t-d} + q_t . r_d` (`r_d` the projected sinusoid of
/// `d`), capped by `cap * tanh(x / cap)`.
struct Attention {
    q: ClippedLinear,
    k: ClippedLinear,
    v: ClippedLinear,
    post: ClippedLinear,
    /// `[heads, head_dim, window]` relative keys, f32.
    relative_keys: Tensor,
    /// `[head_dim]` query scale: `head_dim^-1/2 / ln 2 * softplus(per_dim_scale)`, f32.
    query_scale: Tensor,
    key_scale: f64,
    cap: f64,
    heads: usize,
    head_dim: usize,
    window: usize,
}

impl Attention {
    fn new(cfg: &AudioConfig, positions: &Tensor, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let (heads, head_dim, window) = (cfg.num_attention_heads, cfg.head_dim(), cfg.window());
        let relative = linear_no_bias(hidden, hidden, vb.pp("relative_k_proj"))?;
        let relative_keys = positions
            .apply(&relative)?
            .to_dtype(DType::F32)?
            .reshape((window, heads, head_dim))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let per_dim = vb.get(head_dim, "per_dim_scale")?.to_dtype(DType::F32)?;
        // softplus(x) = ln(1 + e^x)
        let softplus = (per_dim.exp()? + 1.0)?.log()?;
        let ln2 = std::f64::consts::LN_2;
        let query_scale = softplus.affine((head_dim as f64).powf(-0.5) / ln2, 0.0)?;
        Ok(Self {
            q: ClippedLinear::new(cfg, hidden, hidden, vb.pp("q_proj"))?,
            k: ClippedLinear::new(cfg, hidden, hidden, vb.pp("k_proj"))?,
            v: ClippedLinear::new(cfg, hidden, hidden, vb.pp("v_proj"))?,
            post: ClippedLinear::new(cfg, hidden, hidden, vb.pp("post"))?,
            relative_keys,
            query_scale,
            key_scale: (1.0 + std::f64::consts::E).ln() / ln2,
            cap: cfg.attention_logit_cap,
            heads,
            head_dim,
            window,
        })
    }

    /// `[B, H, T, D]` f32 with each frame's `window` predecessors stacked:
    /// `[B, H, T, window, D]`, slot `d` holding frame `t - d` (zero before 0).
    fn windows(&self, xs: &Tensor, frames: usize) -> Result<Tensor> {
        let padded = xs.pad_with_zeros(2, self.window - 1, 0)?;
        let shifted = (0..self.window)
            .map(|d| padded.narrow(2, self.window - 1 - d, frames))
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&shifted, 3)
    }

    /// `mask`: `[B, 1, T, window]` additive f32.
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, frames, _) = xs.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            t.to_dtype(DType::F32)?
                .reshape((batch, frames, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q.forward(xs)?)?.broadcast_mul(&self.query_scale)?;
        let k = split(self.k.forward(xs)?)?.affine(self.key_scale, 0.0)?;
        let v = split(self.v.forward(xs)?)?;

        let keys = self.windows(&k, frames)?;
        let values = self.windows(&v, frames)?;
        // [BHT, 1, D] x [BHT, D, W] -> [B, H, T, W]: one batch dim, as
        // candle's matmul takes at most two.
        let rows = batch * self.heads * frames;
        let content = q
            .reshape((rows, 1, self.head_dim))?
            .matmul(&keys.transpose(3, 4)?.contiguous()?.reshape((
                rows,
                self.head_dim,
                self.window,
            ))?)?
            .reshape((batch, self.heads, frames, self.window))?;
        // [B, H, T, D] x [H, D, W] -> [B, H, T, W]
        let position = q.matmul(
            &self
                .relative_keys
                .unsqueeze(0)?
                .broadcast_as((batch, self.heads, self.head_dim, self.window))?
                .contiguous()?,
        )?;
        let logits = ((content + position)? / self.cap)?
            .tanh()?
            .affine(self.cap, 0.0)?;
        let weights = candle_nn::ops::softmax_last_dim(&logits.broadcast_add(mask)?)?;
        // [BHT, 1, W] x [BHT, W, D] -> [B, H, T, D]
        let out = weights
            .reshape((rows, 1, self.window))?
            .matmul(&values.reshape((rows, self.window, self.head_dim))?)?
            .reshape((batch, self.heads, frames, self.head_dim))?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, frames, self.heads * self.head_dim))?
            .to_dtype(xs.dtype())?;
        self.post.forward(&out)
    }
}

/// GLU, a causal depthwise conv along time, RMSNorm, SiLU, projection.
struct LightConv {
    pre_norm: RmsNorm,
    start: ClippedLinear,
    /// `[kernel, hidden]`: tap `j` multiplies frame `t - (kernel - 1) + j`.
    taps: Tensor,
    conv_norm: RmsNorm,
    end: ClippedLinear,
    kernel: usize,
}

impl LightConv {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let kernel = cfg.conv_kernel_size;
        // Applied in f32: five bf16 partial sums would drift from the
        // reference's single conv accumulation.
        let taps = vb
            .get((hidden, 1, kernel), "depthwise_conv1d.weight")?
            .squeeze(1)?
            .t()?
            .to_dtype(DType::F32)?
            .contiguous()?;
        Ok(Self {
            pre_norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("pre_layer_norm"))?,
            start: ClippedLinear::new(cfg, hidden, hidden * 2, vb.pp("linear_start"))?,
            taps,
            conv_norm: RmsNorm::new(hidden, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            end: ClippedLinear::new(cfg, hidden, hidden, vb.pp("linear_end"))?,
            kernel,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, frames, hidden) = xs.dims3()?;
        let ys = self.start.forward(&self.pre_norm.forward(xs)?)?;
        let ys =
            (ys.narrow(2, 0, hidden)? * candle_nn::ops::sigmoid(&ys.narrow(2, hidden, hidden)?)?)?;
        let padded = ys
            .to_dtype(DType::F32)?
            .pad_with_zeros(1, self.kernel - 1, 0)?;
        let mut conv = padded
            .narrow(1, 0, frames)?
            .broadcast_mul(&self.taps.get(0)?)?;
        for tap in 1..self.kernel {
            conv = (conv
                + padded
                    .narrow(1, tap, frames)?
                    .broadcast_mul(&self.taps.get(tap)?)?)?;
        }
        let ys = self
            .conv_norm
            .forward(&conv.to_dtype(xs.dtype())?)?
            .silu()?;
        self.end.forward(&ys)? + xs
    }
}

struct ConformerLayer {
    feed_forward1: FeedForward,
    norm_pre_attn: RmsNorm,
    attention: Attention,
    norm_post_attn: RmsNorm,
    light_conv: LightConv,
    feed_forward2: FeedForward,
    norm_out: RmsNorm,
}

impl ConformerLayer {
    fn new(cfg: &AudioConfig, positions: &Tensor, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let eps = NORM_EPS;
        Ok(Self {
            feed_forward1: FeedForward::new(cfg, vb.pp("feed_forward1"))?,
            norm_pre_attn: RmsNorm::new(hidden, eps, vb.pp("norm_pre_attn"))?,
            attention: Attention::new(cfg, positions, vb.pp("self_attn"))?,
            norm_post_attn: RmsNorm::new(hidden, eps, vb.pp("norm_post_attn"))?,
            light_conv: LightConv::new(cfg, vb.pp("lconv1d"))?,
            feed_forward2: FeedForward::new(cfg, vb.pp("feed_forward2"))?,
            norm_out: RmsNorm::new(hidden, eps, vb.pp("norm_out"))?,
        })
    }

    // The reference clamps to +-gradient_clipping (1e10) between steps; no
    // finite activation of this model gets there, so the clamps are omitted.
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = self.feed_forward1.forward(xs)?;
        let attended = self
            .attention
            .forward(&self.norm_pre_attn.forward(&xs)?, mask)?;
        let xs = (self.norm_post_attn.forward(&attended)? + xs)?;
        let xs = self.light_conv.forward(&xs)?;
        let xs = self.feed_forward2.forward(&xs)?;
        self.norm_out.forward(&xs)
    }
}

/// A batch of tower outputs: `hidden` is `[batch, frames, output_proj_dims]`;
/// row `b` has `valid[b]` real frames, then padding.
pub struct AudioOutput {
    pub hidden: Tensor,
    pub valid: Vec<usize>,
}

pub struct AudioTower {
    subsample0: SubsampleLayer,
    subsample1: SubsampleLayer,
    input_proj: Linear,
    layers: Vec<ConformerLayer>,
    output_proj: Linear,
    embed_norm: RmsNorm,
    embed_projection: Linear,
    cfg: AudioConfig,
    device: Device,
    dtype: DType,
}

impl AudioTower {
    /// Build from a `VarBuilder` rooted at the checkpoint root (it reads
    /// `audio_tower.*` and `embed_audio.*`). `text_hidden` is the width of a
    /// row of the text tower's `inputs_embeds`.
    pub fn new(cfg: AudioConfig, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let audio = vb.pp("audio_tower");
        let sub = audio.pp("subsample_conv_projection");
        let [c0, c1] = cfg.subsampling_conv_channels;
        let subsample0 = SubsampleLayer::new(1, c0, cfg.rms_norm_eps, sub.pp("layer0"))?;
        let subsample1 = SubsampleLayer::new(c0, c1, cfg.rms_norm_eps, sub.pp("layer1"))?;
        let input_proj =
            linear_no_bias((c0 / 4) * c1, cfg.hidden_size, sub.pp("input_proj_linear"))?;

        // Sinusoids of relative distances 0..window, `[sin..., cos...]`.
        let half = cfg.hidden_size / 2;
        let increment = (10_000f64).ln() / (half.max(2) - 1) as f64;
        let mut table = Vec::with_capacity(cfg.window() * cfg.hidden_size);
        for distance in 0..cfg.window() {
            let angles = (0..half).map(|i| distance as f64 * (-(i as f64) * increment).exp());
            table.extend(angles.clone().map(|a| a.sin() as f32));
            table.extend(angles.map(|a| a.cos() as f32));
        }
        let positions =
            Tensor::from_vec(table, (cfg.window(), cfg.hidden_size), &device)?.to_dtype(dtype)?;

        let layers = (0..cfg.num_hidden_layers)
            .map(|index| ConformerLayer::new(&cfg, &positions, audio.pp(format!("layers.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let output_proj = linear(
            cfg.hidden_size,
            cfg.output_proj_dims,
            audio.pp("output_proj"),
        )?;
        let embed_norm =
            RmsNorm::weightless(cfg.output_proj_dims, cfg.rms_norm_eps, &device, dtype)?;
        let embed_projection = linear_no_bias(
            cfg.output_proj_dims,
            text_hidden,
            vb.pp("embed_audio.embedding_projection"),
        )?;
        Ok(Self {
            subsample0,
            subsample1,
            input_proj,
            layers,
            output_proj,
            embed_norm,
            embed_projection,
            cfg,
            device,
            dtype,
        })
    }

    /// Run the tower over `clips`, padded to the longest. Returns the
    /// `audio_tower` output (before `embed_audio`).
    pub fn forward(&self, clips: &[&AudioFeatures]) -> Result<AudioOutput> {
        let batch = clips.len();
        let frames = clips.iter().map(|clip| clip.frames).max().unwrap_or(0);
        let mut data = Vec::with_capacity(batch * frames * MEL_BINS);
        for clip in clips {
            data.extend_from_slice(&clip.data);
            data.resize(data.len() + (frames - clip.frames) * MEL_BINS, 0.0);
        }
        let features = Tensor::from_vec(data, (batch, 1, frames, MEL_BINS), &self.device)?
            .to_dtype(self.dtype)?;

        // Conv 0 -> [B, T1, F1, C0]; padding frames are zero going in.
        let ys = self.subsample0.forward(&features)?;
        let frames1 = ys.dim(1)?;
        let valid1: Vec<usize> = clips.iter().map(|clip| clip.valid.div_ceil(2)).collect();
        let keep1: Vec<f32> = valid1
            .iter()
            .flat_map(|&valid| (0..frames1).map(move |t| if t < valid { 1.0 } else { 0.0 }))
            .collect();
        let keep1 =
            Tensor::from_vec(keep1, (batch, frames1, 1, 1), &self.device)?.to_dtype(self.dtype)?;
        let ys = ys
            .broadcast_mul(&keep1)?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        // Conv 1 -> [B, T2, F2, C1] -> [B, T2, F2 * C1]
        let ys = self.subsample1.forward(&ys)?;
        let (_, frames2, bins2, channels2) = ys.dims4()?;
        let mut xs = ys
            .reshape((batch, frames2, bins2 * channels2))?
            .apply(&self.input_proj)?;
        let valid2: Vec<usize> = valid1.iter().map(|valid| valid.div_ceil(2)).collect();

        let window = self.cfg.window();
        let mut mask = Vec::with_capacity(batch * frames2 * window);
        for &valid in &valid2 {
            for t in 0..frames2 {
                mask.extend(
                    (0..window).map(|d| if d <= t && t - d < valid { 0.0 } else { MASKED }),
                );
            }
        }
        let mask = Tensor::from_vec(mask, (batch, 1, frames2, window), &self.device)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask)?;
        }
        Ok(AudioOutput {
            hidden: xs.apply(&self.output_proj)?,
            valid: valid2,
        })
    }

    /// `embed_audio`: a weight-free RMSNorm and the projection into the text
    /// tower's space.
    pub fn embed_audio(&self, hidden: &Tensor) -> Result<Tensor> {
        self.embed_norm
            .forward(hidden)?
            .apply(&self.embed_projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_tokens_follow_the_extractor() {
        // 3.3 s: padded to 52864 samples -> 330 frames, the last one over the
        // padding; 329 real frames -> 165 -> 83 tokens.
        let features = log_mel(&vec![0.1f32; 52_800]).unwrap();
        assert_eq!(
            (features.frames, features.valid, features.tokens()),
            (330, 329, 83)
        );
        assert!(features.data[329 * MEL_BINS..].iter().all(|&v| v == 0.0));
        // 30 s is the cap: longer clips are cut, giving the reference's 750 tokens.
        let features = log_mel(&vec![0.0f32; 31 * SAMPLE_RATE]).unwrap();
        assert_eq!(features.tokens(), 750);
    }

    #[test]
    fn a_clip_shorter_than_one_frame_is_rejected() {
        let error = log_mel(&[0.0f32; 10]).unwrap_err();
        assert_eq!(error.code(), "invalid_input");
    }
}
