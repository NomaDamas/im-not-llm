//! The EmbeddingGemma 2 multimodal embedder: image, video and audio soft
//! tokens interleaved into the text tower, behind [`MultimodalEngine`].
//!
//! The text tower is `inl-text`'s EmbeddingGemma 2 (its `language_model.*`
//! weights, mean pooling, `512 -> 768` projection, L2 norm). This module adds
//! the vision tower (`vision_tower.*`, `embed_vision.*`), the audio tower
//! (`audio_tower.*`, `embed_audio.*`), the `<|image|>` / `<|video|>` /
//! `<|audio|>` placeholder expansion the reference processor performs, and the
//! `masked_scatter` merge that overwrites the placeholder rows with the soft
//! tokens.

use crate::audio::{self, AudioConfig, AudioFeatures, AudioTower};
use crate::preprocess::{self, PreprocessedImage};
use crate::video;
use crate::vision::{AttentionImpl, VisionBatch, VisionConfig, VisionTower};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use inl_core::attention::AttentionRequest;
use inl_core::deadline::BatchDeadline;
use inl_core::device::{
    device_label, dtype_name, model_cache_dir, resolve_device, DeviceSpec, DtypeSpec, LoadSlot,
};
use inl_core::engine::{EngineInfo, Modality, MultimodalEngine, MultimodalInput};
use inl_core::hub::ModelSource;
use inl_core::{InlError, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Model context length in tokens.
const DEFAULT_MAX_LENGTH: usize = 8192;
/// Default vision token budget per image.
const DEFAULT_MAX_SOFT_TOKENS: usize = 280;
/// Default vision token budget per video frame.
const DEFAULT_VIDEO_MAX_SOFT_TOKENS: usize = 140;
/// Default number of inputs in one batch.
const DEFAULT_BATCH_SIZE: usize = 8;
/// Default cap on padded tokens per forward pass.
const DEFAULT_MAX_BATCH_TOKENS: usize = 8192;
/// Default deadline for one forward pass, in seconds.
const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

/// Matryoshka output sizes EmbeddingGemma 2 was trained for.
const MATRYOSHKA_DIMS: [usize; 4] = [128, 256, 512, 768];

/// The full output dimension of the checkpoint.
const FULL_DIM: usize = 768;

/// Settings for a [`MultimodalEmbedder`], with builder-style setters.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalOptions {
    /// A Hugging Face repo id or a local model directory.
    pub id: String,
    /// Compute device; `Auto` is Metal on macOS, else CUDA if built, else CPU.
    pub device: DeviceSpec,
    /// Weight precision; only `f32` and `bf16` are accepted.
    pub dtype: Option<DtypeSpec>,
    pub attention: AttentionRequest,
    /// Query rows per tiled-attention block for the vision tower. `None` takes
    /// the `INL_VISION_ATTENTION_TILE` environment override or the default
    /// (`256`). Only the tiled path reads it; the Metal fused path ignores it.
    pub vision_attention_tile: Option<usize>,
    /// Model context length in tokens.
    pub max_length: usize,
    /// Soft-token budget per image (one of 70, 140, 280, 560, 1120).
    pub max_soft_tokens: usize,
    /// Soft-token budget per video frame (one of 70, 140, 280, 560, 1120).
    pub video_max_soft_tokens: usize,
    /// Frames per second captured from a video.
    pub video_fps: f64,
    /// Towers to load besides the text tower (always loaded): `None` loads
    /// every tower the checkpoint has. Leaving out both image and video skips
    /// the vision tower and its weights; leaving out audio skips the audio
    /// tower. A modality left out is rejected as input.
    pub modalities: Option<Vec<Modality>>,
    /// Matryoshka output dimension (one of 128, 256, 512, 768).
    pub output_dim: Option<usize>,
    /// Items per forward pass: inputs per text-tower pass, images/frames per
    /// vision-tower pass.
    pub batch_size: usize,
    /// Cap on padded tokens (`items x longest item`) per forward pass; the
    /// vision tower counts real patches (up to `9 x` the soft-token budget
    /// per image). An item over the cap runs alone.
    pub max_batch_tokens: usize,
    /// Deadline for one forward pass, in seconds. A pass that overruns keeps
    /// its thread, so every later call fails fast until the process restarts.
    pub timeout_seconds: u64,
    /// Directory for cached model files; `None` uses the shared cache.
    pub cache_dir: Option<PathBuf>,
}

impl Default for MultimodalOptions {
    fn default() -> Self {
        Self {
            id: "google/embeddinggemma-2".to_string(),
            device: DeviceSpec::Auto,
            dtype: None,
            attention: AttentionRequest::Auto,
            vision_attention_tile: None,
            max_length: DEFAULT_MAX_LENGTH,
            max_soft_tokens: DEFAULT_MAX_SOFT_TOKENS,
            video_max_soft_tokens: DEFAULT_VIDEO_MAX_SOFT_TOKENS,
            video_fps: video::DEFAULT_FPS,
            modalities: None,
            output_dim: None,
            batch_size: DEFAULT_BATCH_SIZE,
            max_batch_tokens: DEFAULT_MAX_BATCH_TOKENS,
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            cache_dir: None,
        }
    }
}

impl MultimodalOptions {
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_device(mut self, device: DeviceSpec) -> Self {
        self.device = device;
        self
    }

    pub fn with_dtype(mut self, dtype: Option<DtypeSpec>) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_attention(mut self, attention: AttentionRequest) -> Self {
        self.attention = attention;
        self
    }

    /// Set the vision tower's tiled-attention query tile (ignored by the Metal
    /// fused path). Must be positive.
    pub fn with_vision_attention_tile(mut self, tile: usize) -> Self {
        self.vision_attention_tile = Some(tile);
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_max_soft_tokens(mut self, tokens: usize) -> Self {
        self.max_soft_tokens = tokens;
        self
    }

    pub fn with_video_max_soft_tokens(mut self, tokens: usize) -> Self {
        self.video_max_soft_tokens = tokens;
        self
    }

    pub fn with_video_fps(mut self, fps: f64) -> Self {
        self.video_fps = fps;
        self
    }

    pub fn with_modalities(mut self, modalities: Vec<Modality>) -> Self {
        self.modalities = Some(modalities);
        self
    }

    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }

    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    fn validate(&self) -> Result<()> {
        if !preprocess::SUPPORTED_SOFT_TOKENS.contains(&self.max_soft_tokens) {
            return Err(InlError::Config(format!(
                "max_soft_tokens {} is not one of {:?}",
                self.max_soft_tokens,
                preprocess::SUPPORTED_SOFT_TOKENS
            )));
        }
        if !preprocess::SUPPORTED_SOFT_TOKENS.contains(&self.video_max_soft_tokens) {
            return Err(InlError::Config(format!(
                "video_max_soft_tokens {} is not one of {:?}",
                self.video_max_soft_tokens,
                preprocess::SUPPORTED_SOFT_TOKENS
            )));
        }
        if let Some(dim) = self.output_dim {
            if !MATRYOSHKA_DIMS.contains(&dim) {
                return Err(InlError::Config(format!(
                    "output_dim {dim} is not one of {MATRYOSHKA_DIMS:?}"
                )));
            }
        }
        if self.video_fps.is_nan() || self.video_fps <= 0.0 {
            return Err(InlError::Config("video_fps must be positive".to_string()));
        }
        Ok(())
    }
}

/// Which placeholder a media kind uses; indexes [`SpecialTokens::kinds`].
const IMAGE: usize = 0;
const VIDEO: usize = 1;
const AUDIO: usize = 2;

/// One media kind's placeholder token, the tokens bracketing an expanded
/// block of it, and the placeholder's id.
struct Placeholder {
    token: String,
    open: String,
    close: String,
    id: u32,
}

impl Placeholder {
    /// `open` + `count` placeholder tokens + `close`.
    fn block(&self, count: usize) -> String {
        format!("{}{}{}", self.open, self.token.repeat(count), self.close)
    }
}

/// The special tokens the placeholder expansion needs, by [`IMAGE`],
/// [`VIDEO`] and [`AUDIO`].
struct SpecialTokens {
    kinds: [Placeholder; 3],
}

impl SpecialTokens {
    fn load(config_bytes: &[u8], source: &ModelSource, tokenizer: &Tokenizer) -> Result<Self> {
        let tokenizer_config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.fetch("tokenizer_config.json")?)?)?;
        let string = |key: &str| -> Result<String> {
            tokenizer_config
                .get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| InlError::Inference(format!("tokenizer_config.json has no {key:?}")))
        };
        let ids: serde_json::Value = serde_json::from_slice(config_bytes)?;
        let id = |key: &str| -> Result<u32> {
            ids.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| InlError::Inference(format!("config.json has no {key:?}")))
        };
        let placeholder =
            |token: &str, open: &str, close: &str, id_key: &str| -> Result<Placeholder> {
                let placeholder = Placeholder {
                    token: string(token)?,
                    open: string(open)?,
                    close: string(close)?,
                    id: id(id_key)?,
                };
                match tokenizer.token_to_id(&placeholder.token) {
                    Some(found) if found == placeholder.id => Ok(placeholder),
                    other => Err(InlError::Inference(format!(
                        "tokenizer maps {:?} to {other:?}, expected id {}",
                        placeholder.token, placeholder.id
                    ))),
                }
            };
        Ok(Self {
            kinds: [
                placeholder("image_token", "boi_token", "eoi_token", "image_token_id")?,
                placeholder("video_token", "boi_token", "eoi_token", "video_token_id")?,
                placeholder("audio_token", "boa_token", "eoa_token", "audio_token_id")?,
            ],
        })
    }
}

/// Everything the forward pass needs, built once per load.
struct Loaded {
    text: inl_text::Gemma2Embedder,
    /// `None` when neither images nor video were asked for.
    vision: Option<VisionTower>,
    /// `None` when audio was not asked for.
    audio: Option<AudioTower>,
    tokenizer: Tokenizer,
    tokens: SpecialTokens,
    device: Device,
    dtype: DType,
    options: MultimodalOptions,
    info: EngineInfo,
    text_hidden: usize,
}

impl Loaded {
    fn load(options: &MultimodalOptions) -> Result<Self> {
        options.validate()?;
        let device = resolve_device(options.device)?;
        let dtype = match options.dtype {
            None | Some(DtypeSpec::F32) => DType::F32,
            Some(DtypeSpec::Bf16) => DType::BF16,
            Some(DtypeSpec::F16) => {
                return Err(InlError::Config(
                    "EmbeddingGemma 2 overflows float16 and returns NaN: use dtype f32 or bf16"
                        .to_string(),
                ))
            }
        };
        if dtype == DType::BF16 && device.is_cpu() {
            return Err(InlError::Config(
                "EmbeddingGemma 2 in bf16 needs a GPU: the CPU matmul has no bf16 kernel; \
                 use dtype f32 on the CPU"
                    .to_string(),
            ));
        }
        let cache = model_cache_dir(options.cache_dir.as_deref());
        let source = ModelSource::resolve(&options.id, cache.clone())?;

        // Text tower: reuse inl-text's loader (same weights, same lifecycle).
        let text = inl_text::Gemma2Embedder::from_hf(&inl_text::LoadOptions {
            repo_id: &options.id,
            default_cache_dir: cache,
            device: &device,
            dtype,
            max_length: options.max_length,
            attention: options.attention,
        })?;
        let text_hidden = text.hidden_size();

        let config_bytes = std::fs::read(source.fetch("config.json")?)?;
        let wants = |modality: Modality| {
            options
                .modalities
                .as_ref()
                .is_none_or(|list| list.contains(&modality))
        };
        let audio_config = match (wants(Modality::Audio), AudioConfig::parse(&config_bytes)?) {
            (true, Some(config)) => Some(config),
            (true, None) if options.modalities.is_some() => {
                return Err(InlError::Config(format!(
                    "{} has no audio tower",
                    options.id
                )))
            }
            _ => None,
        };
        let wants_vision = wants(Modality::Image) || wants(Modality::Video);
        let mut vision = None;
        let mut audio = None;
        if wants_vision || audio_config.is_some() {
            let weights = source.weight_files()?;
            // SAFETY: the safetensors files are not modified while the model is loaded.
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device) }
                .map_err(|e| InlError::Inference(format!("media weight load failed: {e}")))?;
            if wants_vision {
                let config = VisionConfig::parse(&config_bytes)?;
                let mut tower = VisionTower::new(config, text_hidden, vb.clone())
                    .map_err(|e| InlError::Inference(format!("vision build failed: {e}")))?
                    .with_attention(options.attention)?;
                if let Some(tile) = options.vision_attention_tile {
                    tower = tower.with_attention_tile(tile)?;
                }
                vision = Some(tower);
            }
            if let Some(config) = audio_config {
                audio = Some(
                    AudioTower::new(config, text_hidden, vb)
                        .map_err(|e| InlError::Inference(format!("audio build failed: {e}")))?,
                );
            }
        }

        let tokenizer = Tokenizer::from_file(source.fetch("tokenizer.json")?)
            .map_err(|e| InlError::Inference(format!("tokenizer parse failed: {e}")))?;
        let tokens = SpecialTokens::load(&config_bytes, &source, &tokenizer)?;

        let mut modalities = vec![Modality::Text];
        if vision.is_some() {
            modalities.extend(
                [Modality::Image, Modality::Video]
                    .into_iter()
                    .filter(|&m| wants(m)),
            );
        }
        if audio.is_some() {
            modalities.push(Modality::Audio);
        }
        let info = EngineInfo {
            model_id: options.id.clone(),
            dimension: options.output_dim.unwrap_or(FULL_DIM),
            dtype: dtype_name(dtype).to_string(),
            device: device_label(&device),
            // The vision tower's is the one choice made; the text tower's
            // EmbeddingGemma 2 attention is always eager.
            attention: match vision.as_ref().map(VisionTower::attention_impl) {
                Some(AttentionImpl::MetalSdpa) => "fused",
                Some(AttentionImpl::Tiled) => "tiled",
                None => "eager",
            }
            .to_string(),
            max_length: options.max_length,
            modalities,
        };
        Ok(Self {
            text,
            vision,
            audio,
            tokenizer,
            tokens,
            device,
            dtype,
            options: options.clone(),
            info,
            text_hidden,
        })
    }

    /// Decode every image, video frame and audio clip of `inputs` (CPU work:
    /// image decoding, resampling, `ffmpeg`, log-mel), recording where each
    /// input's media sit.
    fn decode_media(&self, inputs: &[MultimodalInput]) -> Result<DecodedMedia> {
        let mut decoded = DecodedMedia {
            items: Vec::new(),
            clips: Vec::new(),
            layout: Vec::with_capacity(inputs.len()),
        };
        for input in inputs {
            for (count, modality) in [
                (input.images.len(), Modality::Image),
                (input.videos.len(), Modality::Video),
                (input.audios.len(), Modality::Audio),
            ] {
                if count > 0 && !self.info.modalities.contains(&modality) {
                    return Err(InlError::InvalidInput(format!(
                        "this engine was loaded without {} support",
                        modality.as_str()
                    )));
                }
            }
            let mut media = InputMedia {
                images: Vec::with_capacity(input.images.len()),
                videos: Vec::with_capacity(input.videos.len()),
                audios: Vec::with_capacity(input.audios.len()),
            };
            let items = &mut decoded.items;
            let budget = self.options.max_soft_tokens;
            for image in &input.images {
                media.images.push(items.len());
                items.push(MediaItem {
                    pre: preprocess::preprocess(&image.bytes, budget)?,
                    budget,
                });
            }
            let budget = self.options.video_max_soft_tokens;
            // The reference cap of 32 frames, lowered further when that many
            // frame blocks would not fit the context.
            let max_frames =
                (self.options.max_length / (budget + 2)).clamp(1, video::DEFAULT_MAX_FRAMES);
            for video in &input.videos {
                let frames =
                    video::extract_frames(&video.bytes, self.options.video_fps, max_frames)?;
                if frames.is_empty() {
                    return Err(InlError::InvalidInput(
                        "the video decoded to zero frames".to_string(),
                    ));
                }
                media.videos.push((items.len(), frames.len()));
                for frame in &frames {
                    items.push(MediaItem {
                        pre: preprocess::preprocess_rgb(frame, budget)?,
                        budget,
                    });
                }
            }
            for clip in &input.audios {
                media.audios.push(decoded.clips.len());
                decoded
                    .clips
                    .push(audio::log_mel(&audio::decode(&clip.bytes)?)?);
            }
            decoded.layout.push(media);
        }
        Ok(decoded)
    }

    /// One vision pass over `frames` (all with the same `output_length`),
    /// padded only to the longest real patch count among them: padding is
    /// masked out of attention and skipped by the pooler, so trimming it
    /// changes nothing but the work done.
    fn run_vision(
        &self,
        frames: &[&PreprocessedImage],
        output_length: usize,
    ) -> Result<Vec<SoftTokens>> {
        let vision = self
            .vision
            .as_ref()
            .expect("images and video decode only when the vision tower is loaded");
        let batch = frames.len();
        let patches = frames
            .iter()
            .map(|frame| frame.real_patches)
            .max()
            .unwrap_or(0)
            .max(1);
        let row = 3 * preprocess::PATCH_SIZE * preprocess::PATCH_SIZE;
        let mut pixels = Vec::with_capacity(batch * patches * row);
        let mut positions = Vec::with_capacity(batch * patches * 2);
        let mut padding = Vec::with_capacity(batch * patches);
        for frame in frames {
            // Real patches come first; rows past them are padding.
            pixels.extend_from_slice(&frame.pixel_values[..patches * row]);
            positions.extend_from_slice(&frame.position_ids[..patches * 2]);
            padding.extend((0..patches).map(|p| if p < frame.real_patches { 0f32 } else { 1f32 }));
        }
        let pixel_values = infer(
            Tensor::from_vec(pixels, (batch, patches, row), &self.device)
                .and_then(|tensor| tensor.to_dtype(self.dtype)),
        )?;
        let position_ids = infer(Tensor::from_slice(
            &positions,
            (batch, patches, 2),
            &self.device,
        ))?;
        let padding_tensor = infer(Tensor::from_slice(&padding, (batch, patches), &self.device))?;
        let output = infer(vision.forward(VisionBatch {
            pixel_values: &pixel_values,
            position_ids: &position_ids,
            padding: &padding_tensor,
            pool_positions: &positions,
            pool_padding: &padding,
            output_length,
        }))?;
        let soft = infer(vision.embed_vision(&output.hidden))?;
        let flat: Vec<f32> = infer(
            soft.to_dtype(DType::F32)
                .and_then(|tensor| tensor.flatten_all())
                .and_then(|tensor| tensor.to_vec1()),
        )?;
        let mut offset = 0;
        Ok(output
            .tokens_per_row
            .iter()
            .map(|&tokens| {
                let len = tokens * self.text_hidden;
                let rows = flat[offset..offset + len].to_vec();
                offset += len;
                SoftTokens { rows, tokens }
            })
            .collect())
    }

    /// One audio pass over `clips`, padded to the longest: padding sits after
    /// every real frame and each tower frame sees only earlier frames, so the
    /// real rows are what each clip gives alone. Padding rows are dropped.
    fn run_audio(&self, clips: &[&AudioFeatures]) -> Result<Vec<SoftTokens>> {
        let tower = self
            .audio
            .as_ref()
            .expect("audio decodes only when the audio tower is loaded");
        let output = infer(tower.forward(clips))?;
        let soft = infer(
            tower
                .embed_audio(&output.hidden)
                .and_then(|tensor| tensor.to_dtype(DType::F32)),
        )?;
        let frames = infer(soft.dim(1))?;
        let flat: Vec<f32> = infer(soft.flatten_all().and_then(|tensor| tensor.to_vec1()))?;
        Ok(output
            .valid
            .iter()
            .enumerate()
            .map(|(row, &tokens)| {
                let start = row * frames * self.text_hidden;
                SoftTokens {
                    rows: flat[start..start + tokens * self.text_hidden].to_vec(),
                    tokens,
                }
            })
            .collect())
    }

    /// Expand the placeholders of one input for its soft tokens, tokenize it
    /// and gather the soft-token rows in placeholder order. `vision` holds the
    /// soft tokens of the decoded images and frames, `audio` of the clips.
    fn prepare_text(
        &self,
        input: &MultimodalInput,
        media: &InputMedia,
        vision: &[SoftTokens],
        audio: &[SoftTokens],
    ) -> Result<PreparedText> {
        let kinds = &self.tokens.kinds;
        // With no text the reference joins one placeholder per item with
        // spaces: images, then videos, then audio.
        let text = match &input.text {
            Some(text) => text.clone(),
            None => [
                (IMAGE, input.images.len()),
                (VIDEO, input.videos.len()),
                (AUDIO, input.audios.len()),
            ]
            .into_iter()
            .flat_map(|(kind, count)| std::iter::repeat_n(kinds[kind].token.as_str(), count))
            .collect::<Vec<_>>()
            .join(" "),
        };

        // The soft tokens of each kind in placeholder order, and one
        // replacement per item (a video's is a block per frame).
        let images: Vec<&SoftTokens> = media.images.iter().map(|&i| &vision[i]).collect();
        let videos: Vec<&[SoftTokens]> = media
            .videos
            .iter()
            .map(|&(first, frames)| &vision[first..first + frames])
            .collect();
        let audios: Vec<&SoftTokens> = media.audios.iter().map(|&i| &audio[i]).collect();
        let replacements: [Vec<String>; 3] = [
            images
                .iter()
                .map(|soft| kinds[IMAGE].block(soft.tokens))
                .collect(),
            videos
                .iter()
                .map(|frames| {
                    frames
                        .iter()
                        .map(|soft| kinds[VIDEO].block(soft.tokens))
                        .collect()
                })
                .collect(),
            audios
                .iter()
                .map(|soft| kinds[AUDIO].block(soft.tokens))
                .collect(),
        ];
        let expanded = expand(
            &text,
            &[
                (kinds[IMAGE].token.as_str(), replacements[IMAGE].as_slice()),
                (kinds[VIDEO].token.as_str(), replacements[VIDEO].as_slice()),
                (kinds[AUDIO].token.as_str(), replacements[AUDIO].as_slice()),
            ],
        )?;

        let encoding = self
            .tokenizer
            .encode(expanded, true)
            .map_err(|e| InlError::Inference(format!("tokenizer failed: {e}")))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq = ids.len();
        if seq == 0 {
            return Err(InlError::InvalidInput(
                "the input tokenized to nothing".to_string(),
            ));
        }
        if seq > self.options.max_length {
            return Err(InlError::InvalidInput(format!(
                "input is {seq} tokens, over the {}-token context",
                self.options.max_length
            )));
        }

        let ordered: [Vec<&SoftTokens>; 3] = [
            images,
            videos.iter().flat_map(|frames| frames.iter()).collect(),
            audios,
        ];
        let mut rows: [Vec<f32>; 3] = Default::default();
        for (kind, soft) in ordered.iter().enumerate() {
            let tokens: usize = soft.iter().map(|soft| soft.tokens).sum();
            let seen = ids.iter().filter(|&&id| id == kinds[kind].id).count();
            if seen != tokens {
                return Err(InlError::InvalidInput(format!(
                    "text has {seen} {} placeholders but the media produced {tokens} soft tokens",
                    kinds[kind].token
                )));
            }
            rows[kind] = soft
                .iter()
                .flat_map(|soft| soft.rows.iter().copied())
                .collect();
        }
        Ok(PreparedText { ids, rows })
    }

    /// One text-tower pass over prepared inputs, right-padded to the longest:
    /// token embeddings with the placeholder rows overwritten by soft tokens
    /// (the reference `masked_scatter`), then the pooled embedding per input.
    fn embed_texts(&self, batch: &[PreparedText]) -> Result<Vec<Vec<f32>>> {
        let rows = batch.len();
        let seq = batch.iter().map(|p| p.ids.len()).max().unwrap_or(0);
        let hidden = self.text_hidden;
        // Padding positions are masked and excluded from pooling, so any id
        // works there.
        let mut ids = Vec::with_capacity(rows * seq);
        for prepared in batch {
            ids.extend_from_slice(&prepared.ids);
            ids.extend(std::iter::repeat_n(0u32, seq - prepared.ids.len()));
        }
        let input_ids = infer(Tensor::from_vec(ids, (rows, seq), &self.device))?;
        let mut embeds: Vec<f32> = infer(
            self.text
                .token_embeddings(&input_ids)
                .and_then(|tensor| tensor.to_dtype(DType::F32))
                .and_then(|tensor| tensor.flatten_all())
                .and_then(|tensor| tensor.to_vec1()),
        )?;

        let placeholder_ids = self.tokens.kinds.each_ref().map(|kind| kind.id);
        for (row, prepared) in batch.iter().enumerate() {
            let mut cursors = [0usize; 3];
            for (pos, &id) in prepared.ids.iter().enumerate() {
                let Some(kind) = placeholder_ids.iter().position(|&p| p == id) else {
                    continue;
                };
                let start = cursors[kind];
                cursors[kind] += hidden;
                let dst = (row * seq + pos) * hidden;
                embeds[dst..dst + hidden]
                    .copy_from_slice(&prepared.rows[kind][start..start + hidden]);
            }
        }

        let embeds = infer(
            Tensor::from_vec(embeds, (rows, seq, hidden), &self.device)
                .and_then(|tensor| tensor.to_dtype(self.dtype)),
        )?;
        let lens: Vec<usize> = batch.iter().map(|p| p.ids.len()).collect();
        let mut embeddings = infer(self.text.embed_from_embeds(&embeds, &lens))?;
        if let Some(dim) = self.options.output_dim {
            for embedding in &mut embeddings {
                truncate_and_normalize(embedding, dim);
            }
        }
        Ok(embeddings)
    }
}

/// One image or video frame waiting for the vision tower.
struct MediaItem {
    pre: PreprocessedImage,
    /// Soft-token budget: the `output_length` of its vision pass.
    budget: usize,
}

/// The media of a chunk of inputs, decoded.
struct DecodedMedia {
    /// Images and video frames, for the vision tower.
    items: Vec<MediaItem>,
    /// Audio clips, for the audio tower.
    clips: Vec<AudioFeatures>,
    /// Per input, where its media sit in `items` and `clips`.
    layout: Vec<InputMedia>,
}

/// Where one input's media sit in the decoded lists.
struct InputMedia {
    images: Vec<usize>,
    /// `(first item, frames)` per video.
    videos: Vec<(usize, usize)>,
    /// Indices into the clips.
    audios: Vec<usize>,
}

/// One item's soft tokens in text space, `[tokens, text_hidden]` row-major.
struct SoftTokens {
    rows: Vec<f32>,
    tokens: usize,
}

/// A tokenized input with its soft-token rows in placeholder order, by
/// [`IMAGE`], [`VIDEO`] and [`AUDIO`].
struct PreparedText {
    ids: Vec<u32>,
    rows: [Vec<f32>; 3],
}

/// Split `lengths` into passes of at most `max_items` items and at most
/// `max_tokens` padded tokens (`items x longest`). Items are taken longest
/// first so each pass pads little; an item longer than `max_tokens` runs
/// alone. Returns indices into `lengths`.
fn plan_batches(lengths: &[usize], max_items: usize, max_tokens: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(lengths[i]));
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for index in order {
        match batches.last_mut() {
            Some(batch)
                if batch.len() < max_items
                    && (batch.len() + 1) * lengths[batch[0]] <= max_tokens =>
            {
                batch.push(index)
            }
            _ => batches.push(vec![index]),
        }
    }
    batches
}

/// Vision passes over `items`: grouped by soft-token budget (one
/// `output_length` per pass), then split by [`plan_batches`] on real patches.
fn plan_vision(
    items: &[MediaItem],
    max_items: usize,
    max_tokens: usize,
) -> Vec<(usize, Vec<usize>)> {
    let mut budgets: Vec<usize> = items.iter().map(|item| item.budget).collect();
    budgets.sort_unstable();
    budgets.dedup();
    let mut passes = Vec::new();
    for budget in budgets {
        let members: Vec<usize> = (0..items.len())
            .filter(|&i| items[i].budget == budget)
            .collect();
        let lengths: Vec<usize> = members.iter().map(|&i| items[i].pre.real_patches).collect();
        for batch in plan_batches(&lengths, max_items, max_tokens) {
            passes.push((budget, batch.into_iter().map(|j| members[j]).collect()));
        }
    }
    passes
}

/// Expand the placeholders the way the reference processor does: each
/// `(placeholder, replacements)` kind consumes its replacements in text order.
fn expand(text: &str, kinds: &[(&str, &[String])]) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    let mut used = vec![0usize; kinds.len()];
    loop {
        let next = kinds
            .iter()
            .enumerate()
            .filter_map(|(kind, (token, _))| remaining.find(token).map(|position| (position, kind)))
            .min();
        let Some((position, kind)) = next else {
            out.push_str(remaining);
            break;
        };
        let (token, replacements) = kinds[kind];
        out.push_str(&remaining[..position]);
        let replacement = replacements.get(used[kind]).ok_or_else(|| {
            InlError::InvalidInput(format!(
                "text has more {token} placeholders than media items ({})",
                replacements.len()
            ))
        })?;
        out.push_str(replacement);
        used[kind] += 1;
        remaining = &remaining[position + token.len()..];
    }
    for ((token, replacements), used) in kinds.iter().zip(used) {
        if used != replacements.len() {
            return Err(InlError::InvalidInput(format!(
                "text has {used} {token} placeholders for {} media items",
                replacements.len()
            )));
        }
    }
    Ok(out)
}

fn truncate_and_normalize(vector: &mut Vec<f32>, dim: usize) {
    vector.truncate(dim);
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// Candle tensor operations surface as `candle_core::Error`; the crate's
/// boundary is [`InlError`].
fn infer<T>(result: candle_core::Result<T>) -> Result<T> {
    result.map_err(|error| InlError::Inference(format!("candle: {error}")))
}

/// CPU-side work (decoding, tokenizing) off the async runtime, without the
/// forward-pass deadline.
async fn blocking<T, F>(task: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(task)
        .await
        .map_err(|error| InlError::Inference(format!("blocking task failed: {error}")))?
}

/// An EmbeddingGemma 2 multimodal embedder over text, images and video.
pub struct MultimodalEmbedder {
    options: MultimodalOptions,
    slot: LoadSlot<Arc<Loaded>>,
    deadline: BatchDeadline,
}

/// Names the work in deadline errors.
const WHAT: &str = "multimodal forward pass";

impl MultimodalEmbedder {
    pub fn new(options: MultimodalOptions) -> Self {
        Self {
            options,
            slot: LoadSlot::new(),
            deadline: BatchDeadline::new(),
        }
    }

    /// The options this embedder was built with.
    pub fn options(&self) -> &MultimodalOptions {
        &self.options
    }

    /// Load the model (once) and return its description.
    pub async fn engine_info(&self) -> Result<EngineInfo> {
        Ok(self.ensure_loaded().await?.info.clone())
    }

    /// Embed inputs, `batch_size` at a time. Their images and video frames
    /// run through the vision tower in shared passes, their audio clips
    /// through the audio tower, then the merged sequences through the text
    /// tower; each pass holds at most `batch_size` items and
    /// `max_batch_tokens` padded tokens (patches for the vision tower, soft
    /// tokens for the audio tower). Every pass runs under
    /// [`MultimodalOptions::timeout_seconds`]; a pass that overruns cannot be
    /// cancelled, so after one every later call fails fast with
    /// [`InlError::Timeout`] until the process restarts.
    pub async fn embed_inputs(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        self.deadline.ensure_healthy(WHAT)?;
        let loaded = self.ensure_loaded().await?;
        let budget = std::time::Duration::from_secs(self.options.timeout_seconds);
        let max_items = self.options.batch_size.max(1);
        let max_tokens = self.options.max_batch_tokens.max(1);
        let mut out = Vec::with_capacity(inputs.len());
        let mut inputs = inputs.into_iter().peekable();
        while inputs.peek().is_some() {
            let chunk: Arc<Vec<MultimodalInput>> =
                Arc::new(inputs.by_ref().take(max_items).collect());

            // Decoding is CPU work (and ffmpeg), not a forward pass.
            let DecodedMedia {
                items,
                clips,
                layout,
            } = {
                let (loaded, chunk) = (loaded.clone(), chunk.clone());
                blocking(move || loaded.decode_media(&chunk)).await?
            };
            let items = Arc::new(items);
            let mut soft: Vec<Option<SoftTokens>> = (0..items.len()).map(|_| None).collect();
            for (output_length, members) in plan_vision(&items, max_items, max_tokens) {
                let (loaded, items, pass) = (loaded.clone(), items.clone(), members.clone());
                let outputs = self
                    .deadline
                    .run(WHAT, budget, move || {
                        let frames: Vec<&PreprocessedImage> =
                            pass.iter().map(|&i| &items[i].pre).collect();
                        loaded.run_vision(&frames, output_length)
                    })
                    .await?;
                for (index, output) in members.into_iter().zip(outputs) {
                    soft[index] = Some(output);
                }
            }
            drop(items);
            let soft: Vec<SoftTokens> = soft
                .into_iter()
                .map(|output| output.expect("every media item is in one vision pass"))
                .collect();

            let clips = Arc::new(clips);
            let lengths: Vec<usize> = clips.iter().map(AudioFeatures::tokens).collect();
            let mut heard: Vec<Option<SoftTokens>> = (0..clips.len()).map(|_| None).collect();
            for members in plan_batches(&lengths, max_items, max_tokens) {
                let (loaded, clips, pass) = (loaded.clone(), clips.clone(), members.clone());
                let outputs = self
                    .deadline
                    .run(WHAT, budget, move || {
                        let pass: Vec<&AudioFeatures> = pass.iter().map(|&i| &clips[i]).collect();
                        loaded.run_audio(&pass)
                    })
                    .await?;
                for (index, output) in members.into_iter().zip(outputs) {
                    heard[index] = Some(output);
                }
            }
            drop(clips);
            let heard: Vec<SoftTokens> = heard
                .into_iter()
                .map(|output| output.expect("every clip is in one audio pass"))
                .collect();
            let prepared = {
                let (loaded, chunk) = (loaded.clone(), chunk.clone());
                blocking(move || {
                    chunk
                        .iter()
                        .zip(&layout)
                        .map(|(input, media)| loaded.prepare_text(input, media, &soft, &heard))
                        .collect::<Result<Vec<_>>>()
                })
                .await?
            };
            let lengths: Vec<usize> = prepared.iter().map(|p| p.ids.len()).collect();
            let mut prepared: Vec<Option<PreparedText>> = prepared.into_iter().map(Some).collect();
            let mut embeddings: Vec<Option<Vec<f32>>> = (0..lengths.len()).map(|_| None).collect();
            for members in plan_batches(&lengths, max_items, max_tokens) {
                let batch: Vec<PreparedText> = members
                    .iter()
                    .map(|&i| prepared[i].take().expect("each input is in one text pass"))
                    .collect();
                let loaded = loaded.clone();
                let outputs = self
                    .deadline
                    .run(WHAT, budget, move || loaded.embed_texts(&batch))
                    .await?;
                for (index, output) in members.into_iter().zip(outputs) {
                    embeddings[index] = Some(output);
                }
            }
            out.extend(
                embeddings
                    .into_iter()
                    .map(|output| output.expect("every input is embedded")),
            );
        }
        Ok(out)
    }

    async fn ensure_loaded(&self) -> Result<Arc<Loaded>> {
        self.slot
            .get_or_load(|| {
                let options = self.options.clone();
                async move {
                    let loaded = tokio::task::spawn_blocking(move || Loaded::load(&options))
                        .await
                        .map_err(|e| {
                            InlError::Inference(format!("model load task failed: {e}"))
                        })??;
                    Ok(Arc::new(loaded))
                }
            })
            .await
    }
}

#[async_trait]
impl MultimodalEngine for MultimodalEmbedder {
    async fn load(&self) -> Result<EngineInfo> {
        self.engine_info().await
    }

    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        self.embed_inputs(inputs).await
    }

    async fn unload(&self) -> bool {
        self.slot.unload().await
    }

    fn is_loaded(&self) -> bool {
        self.slot.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matryoshka_validation_rejects_unknown_dims() {
        let options = MultimodalOptions {
            output_dim: Some(100),
            ..Default::default()
        };
        let error = Loaded::load(&options).err().expect("must fail");
        assert_eq!(error.code(), "config");
    }

    #[test]
    fn f16_is_rejected() {
        let options = MultimodalOptions {
            dtype: Some(DtypeSpec::F16),
            ..Default::default()
        };
        let error = Loaded::load(&options).err().expect("must fail");
        assert!(matches!(error, InlError::Config(_)));
    }

    #[test]
    fn expand_consumes_media_in_order() {
        let images = vec!["I0".to_string(), "I1".to_string()];
        let videos = vec!["V0".to_string()];
        let audios = vec!["A0".to_string()];
        let text = "a <|image|> b <|audio|><|video|> c <|image|>";
        let kinds = [
            ("<|image|>", images.as_slice()),
            ("<|video|>", videos.as_slice()),
            ("<|audio|>", audios.as_slice()),
        ];
        assert_eq!(expand(text, &kinds).unwrap(), "a I0 b A0V0 c I1");
    }

    #[test]
    fn expand_rejects_a_placeholder_media_mismatch() {
        let none: &[String] = &[];
        let one = ["A0".to_string()];
        // A placeholder without its item, and an item without its placeholder.
        let error = expand("x <|audio|>", &[("<|image|>", none), ("<|audio|>", none)]).unwrap_err();
        assert_eq!(error.code(), "invalid_input");
        let error = expand("x", &[("<|image|>", none), ("<|audio|>", &one)]).unwrap_err();
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn truncate_renormalizes() {
        let mut vector = vec![3.0f32, 4.0, 100.0];
        truncate_and_normalize(&mut vector, 2);
        assert_eq!(vector.len(), 2);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn batches_respect_the_item_and_padded_token_caps() {
        // Longest first; 3 x 100 fits 300 tokens, a fourth would not.
        let batches = plan_batches(&[50, 100, 80, 100, 10], 8, 300);
        assert_eq!(batches, vec![vec![1, 3, 2], vec![0, 4]]);
        // The item cap binds before the token cap.
        let batches = plan_batches(&[1, 1, 1, 1, 1], 2, 1000);
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }

    #[test]
    fn an_item_over_the_token_cap_runs_alone() {
        let batches = plan_batches(&[5000, 10, 10], 8, 100);
        assert_eq!(batches, vec![vec![0], vec![1, 2]]);
        assert!(plan_batches(&[], 8, 100).is_empty());
    }
}
