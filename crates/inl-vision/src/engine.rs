//! The EmbeddingGemma 2 multimodal embedder: image (and video) soft tokens
//! interleaved into the text tower, behind [`MultimodalEngine`].
//!
//! The text tower is `inl-text`'s EmbeddingGemma 2 (its `language_model.*`
//! weights, mean pooling, `512 -> 768` projection, L2 norm). This module adds
//! the vision tower (`vision_tower.*`), the projection into text space
//! (`embed_vision.*`), the `<|image|>` / `<|video|>` placeholder expansion the
//! reference processor performs, and the `masked_scatter` merge that overwrites
//! the placeholder rows with the soft tokens.

use crate::preprocess::{self, PreprocessedImage};
use crate::video;
use crate::vision::{AttentionImpl, VisionBatch, VisionConfig, VisionTower};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use inl_core::attention::AttentionRequest;
use inl_core::device::{device_label, dtype_name, resolve_device, DeviceSpec, DtypeSpec, LoadSlot};
use inl_core::engine::{EngineInfo, MediaBytes, Modality, MultimodalEngine, MultimodalInput};
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
    /// Model context length in tokens.
    pub max_length: usize,
    /// Soft-token budget per image (one of 70, 140, 280, 560, 1120).
    pub max_soft_tokens: usize,
    /// Soft-token budget per video frame (one of 70, 140, 280, 560, 1120).
    pub video_max_soft_tokens: usize,
    /// Frames per second captured from a video.
    pub video_fps: f64,
    /// Matryoshka output dimension (one of 128, 256, 512, 768).
    pub output_dim: Option<usize>,
    /// Inputs per forward pass.
    pub batch_size: usize,
    /// Cap on padded tokens (`inputs x longest input`) per forward pass.
    pub max_batch_tokens: usize,
    /// Deadline for one forward pass, in seconds.
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
            max_length: DEFAULT_MAX_LENGTH,
            max_soft_tokens: DEFAULT_MAX_SOFT_TOKENS,
            video_max_soft_tokens: DEFAULT_VIDEO_MAX_SOFT_TOKENS,
            video_fps: video::DEFAULT_FPS,
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

/// The special tokens and ids the placeholder expansion needs.
struct SpecialTokens {
    image: String,
    boi: String,
    eoi: String,
    video: String,
    image_id: u32,
    video_id: u32,
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
        let tokens = Self {
            image: string("image_token")?,
            boi: string("boi_token")?,
            eoi: string("eoi_token")?,
            video: string("video_token")?,
            image_id: id("image_token_id")?,
            video_id: id("video_token_id")?,
        };
        for (text, expected) in [
            (&tokens.image, tokens.image_id),
            (&tokens.video, tokens.video_id),
        ] {
            match tokenizer.token_to_id(text) {
                Some(found) if found == expected => {}
                other => {
                    return Err(InlError::Inference(format!(
                        "tokenizer maps {text:?} to {other:?}, expected id {expected}"
                    )))
                }
            }
        }
        Ok(tokens)
    }

    /// `boi` + `count` placeholder tokens + `eoi`.
    fn block(&self, placeholder: &str, count: usize) -> String {
        format!("{}{}{}", self.boi, placeholder.repeat(count), self.eoi)
    }
}

/// Everything the forward pass needs, built once per load.
struct Loaded {
    text: inl_text::Gemma2Embedder,
    vision: VisionTower,
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
        let cache = options.cache_dir.clone().unwrap_or_else(std::env::temp_dir);
        let source = ModelSource::resolve(&options.id, cache)?;

        // Text tower: reuse inl-text's loader (same weights, same lifecycle).
        let text = inl_text::Gemma2Embedder::from_hf(&inl_text::LoadOptions {
            repo_id: &options.id,
            default_cache_dir: options.cache_dir.clone().unwrap_or_else(std::env::temp_dir),
            device: &device,
            dtype,
            max_length: options.max_length,
            attention: options.attention,
        })?;
        let text_hidden = text.hidden_size();

        let config_bytes = std::fs::read(source.fetch("config.json")?)?;
        let vision_config = VisionConfig::parse(&config_bytes)?;
        let weights = source.weight_files()?;
        // SAFETY: the safetensors files are not modified while the model is loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device) }
            .map_err(|e| InlError::Inference(format!("vision weight load failed: {e}")))?;
        let vision = VisionTower::new(vision_config, text_hidden, vb)
            .map_err(|e| InlError::Inference(format!("vision build failed: {e}")))?
            .with_attention(options.attention)?;

        let tokenizer = Tokenizer::from_file(source.fetch("tokenizer.json")?)
            .map_err(|e| InlError::Inference(format!("tokenizer parse failed: {e}")))?;
        let tokens = SpecialTokens::load(&config_bytes, &source, &tokenizer)?;

        let info = EngineInfo {
            model_id: options.id.clone(),
            dimension: options.output_dim.unwrap_or(FULL_DIM),
            dtype: dtype_name(dtype).to_string(),
            device: device_label(&device),
            attention: match vision.attention_impl() {
                AttentionImpl::MetalSdpa => "fused",
                AttentionImpl::Tiled => "tiled",
            }
            .to_string(),
            max_length: options.max_length,
            modalities: vec![Modality::Text, Modality::Image, Modality::Video],
        };
        Ok(Self {
            text,
            vision,
            tokenizer,
            tokens,
            device,
            dtype,
            options: options.clone(),
            info,
            text_hidden,
        })
    }

    /// Soft tokens for one image, projected into text space.
    fn image_soft_tokens(&self, image: &MediaBytes) -> Result<Vec<Vec<f32>>> {
        let pre = preprocess::preprocess(&image.bytes, self.options.max_soft_tokens)?;
        self.run_vision(&[pre], self.options.max_soft_tokens)
    }

    /// Soft tokens for one video: `(frames, rows)` where `rows` holds the
    /// per-frame soft tokens concatenated.
    fn video_soft_tokens(&self, video: &MediaBytes) -> Result<(usize, Vec<Vec<f32>>)> {
        let budget = self.options.video_max_soft_tokens;
        let max_frames = (self.options.max_length / (budget + 2)).max(1);
        let frames = video::extract_frames(&video.bytes, self.options.video_fps, max_frames)?;
        if frames.is_empty() {
            return Err(InlError::InvalidInput(
                "the video decoded to zero frames".to_string(),
            ));
        }
        let count = frames.len();
        let preprocessed = frames
            .iter()
            .map(|frame| preprocess::preprocess_rgb(frame, budget))
            .collect::<Result<Vec<_>>>()?;
        let rows = self.run_vision(&preprocessed, budget)?;
        Ok((count, rows))
    }

    /// Run the vision tower over a batch of equally laid-out frames.
    fn run_vision(
        &self,
        frames: &[PreprocessedImage],
        output_length: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = frames.len();
        let max_patches = frames[0].max_patches;
        let mut pixels = Vec::with_capacity(batch * max_patches * 768);
        let mut positions = Vec::with_capacity(batch * max_patches * 2);
        let mut padding = Vec::with_capacity(batch * max_patches);
        for frame in frames {
            pixels.extend_from_slice(&frame.pixel_values);
            positions.extend_from_slice(&frame.position_ids);
            for p in 0..frame.max_patches {
                padding.push(if p < frame.real_patches { 0f32 } else { 1f32 });
            }
        }
        let pixel_values = infer(
            Tensor::from_vec(pixels, (batch, max_patches, 768), &self.device)
                .and_then(|tensor| tensor.to_dtype(self.dtype)),
        )?;
        let position_ids = infer(Tensor::from_vec(
            positions,
            (batch, max_patches, 2),
            &self.device,
        ))?;
        let padding = infer(Tensor::from_vec(
            padding,
            (batch, max_patches),
            &self.device,
        ))?;
        let output = infer(self.vision.forward(VisionBatch {
            pixel_values: &pixel_values,
            position_ids: &position_ids,
            padding: &padding,
            output_length,
        }))?;
        let soft = infer(self.vision.embed_vision(&output.hidden))?;
        infer(
            soft.to_dtype(DType::F32)
                .and_then(|tensor| tensor.to_vec2()),
        )
    }

    /// Build the merged `inputs_embeds` and run the text tower.
    fn embed_one(&self, input: &MultimodalInput) -> Result<Vec<f32>> {
        if !input.audios.is_empty() {
            return Err(InlError::InvalidInput(
                "audio is not supported by this build".to_string(),
            ));
        }

        // Soft tokens, in order, for images and videos.
        let image_blocks = input
            .images
            .iter()
            .map(|image| self.image_soft_tokens(image))
            .collect::<Result<Vec<_>>>()?;
        let video_blocks = input
            .videos
            .iter()
            .map(|video| self.video_soft_tokens(video))
            .collect::<Result<Vec<_>>>()?;

        let mut image_features: Vec<Vec<f32>> = Vec::new();
        for block in &image_blocks {
            image_features.extend(block.iter().cloned());
        }
        let mut video_features: Vec<Vec<f32>> = Vec::new();
        for (_, block) in &video_blocks {
            video_features.extend(block.iter().cloned());
        }

        // The text (an image-only input is just the image placeholder).
        let text = match (&input.text, input.images.len(), input.videos.len()) {
            (Some(text), _, _) => text.clone(),
            (None, 1, 0) => self.tokens.image.clone(),
            (None, 0, 1) => self.tokens.video.clone(),
            (None, 0, 0) => String::new(),
            (None, _, _) => {
                return Err(InlError::InvalidInput(
                    "a multimodal input with no text must carry exactly one media item".to_string(),
                ))
            }
        };

        let image_replacements: Vec<String> = image_blocks
            .iter()
            .map(|block| self.tokens.block(&self.tokens.image, block.len()))
            .collect();
        let video_replacements: Vec<String> = video_blocks
            .iter()
            .map(|(frames, block)| {
                let per_frame = if *frames == 0 {
                    0
                } else {
                    block.len() / frames
                };
                self.tokens
                    .block(&self.tokens.video, per_frame)
                    .repeat(*frames)
            })
            .collect();
        let expanded = expand(
            &text,
            &self.tokens.image,
            &self.tokens.video,
            &image_replacements,
            &video_replacements,
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

        let (mut image_seen, mut video_seen) = (0usize, 0usize);
        for &id in &ids {
            if id == self.tokens.image_id {
                image_seen += 1;
            } else if id == self.tokens.video_id {
                video_seen += 1;
            }
        }
        if image_seen != image_features.len() {
            return Err(InlError::InvalidInput(format!(
                "text has {image_seen} image placeholders but the images produced {} soft tokens",
                image_features.len()
            )));
        }
        if video_seen != video_features.len() {
            return Err(InlError::InvalidInput(format!(
                "text has {video_seen} video placeholders but the videos produced {} soft tokens",
                video_features.len()
            )));
        }

        let input_ids = infer(Tensor::from_vec(ids.clone(), (1, seq), &self.device))?;
        let mut embeds: Vec<f32> = infer(
            self.text
                .token_embeddings(&input_ids)
                .and_then(|tensor| tensor.to_dtype(DType::F32))
                .and_then(|tensor| tensor.flatten_all())
                .and_then(|tensor| tensor.to_vec1()),
        )?;

        let (mut image_cursor, mut video_cursor) = (0usize, 0usize);
        for (pos, &id) in ids.iter().enumerate() {
            let row = if id == self.tokens.image_id {
                let row = &image_features[image_cursor];
                image_cursor += 1;
                Some(row)
            } else if id == self.tokens.video_id {
                let row = &video_features[video_cursor];
                video_cursor += 1;
                Some(row)
            } else {
                None
            };
            if let Some(row) = row {
                let dst = pos * self.text_hidden;
                embeds[dst..dst + self.text_hidden].copy_from_slice(row);
            }
        }

        let embeds = infer(
            Tensor::from_vec(embeds, (1, seq, self.text_hidden), &self.device)
                .and_then(|tensor| tensor.to_dtype(self.dtype)),
        )?;
        let mut embedding = infer(self.text.embed_from_embeds(&embeds, &[seq]))?
            .into_iter()
            .next()
            .ok_or_else(|| InlError::Inference("text tower returned no embedding".to_string()))?;
        if let Some(dim) = self.options.output_dim {
            truncate_and_normalize(&mut embedding, dim);
        }
        Ok(embedding)
    }
}

/// Expand the image/video placeholders the way the reference processor does,
/// consuming the prebuilt replacements in text order.
fn expand(
    text: &str,
    image_token: &str,
    video_token: &str,
    images: &[String],
    videos: &[String],
) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut remaining = text;
    let (mut image_index, mut video_index) = (0usize, 0usize);
    loop {
        let next_image = remaining.find(image_token);
        let next_video = remaining.find(video_token);
        let (position, is_image) = match (next_image, next_video) {
            (None, None) => {
                out.push_str(remaining);
                break;
            }
            (Some(p), None) => (p, true),
            (None, Some(p)) => (p, false),
            (Some(p), Some(q)) => (p.min(q), p <= q),
        };
        out.push_str(&remaining[..position]);
        if is_image {
            let replacement = images.get(image_index).ok_or_else(|| {
                InlError::InvalidInput("more image placeholders than images".to_string())
            })?;
            out.push_str(replacement);
            image_index += 1;
            remaining = &remaining[position + image_token.len()..];
        } else {
            let replacement = videos.get(video_index).ok_or_else(|| {
                InlError::InvalidInput("more video placeholders than videos".to_string())
            })?;
            out.push_str(replacement);
            video_index += 1;
            remaining = &remaining[position + video_token.len()..];
        }
    }
    if image_index != images.len() {
        return Err(InlError::InvalidInput(format!(
            "text has {image_index} image placeholders for {} images",
            images.len()
        )));
    }
    if video_index != videos.len() {
        return Err(InlError::InvalidInput(format!(
            "text has {video_index} video placeholders for {} videos",
            videos.len()
        )));
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

/// An EmbeddingGemma 2 multimodal embedder over text, images and video.
pub struct MultimodalEmbedder {
    options: MultimodalOptions,
    slot: LoadSlot<Arc<Loaded>>,
}

impl MultimodalEmbedder {
    pub fn new(options: MultimodalOptions) -> Self {
        Self {
            options,
            slot: LoadSlot::new(),
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

    /// Embed inputs, running each on its own blocking task under
    /// [`MultimodalOptions::timeout_seconds`]. A timed-out pass cannot be
    /// cancelled; the task keeps running to completion in the background.
    pub async fn embed_inputs(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        let loaded = self.ensure_loaded().await?;
        let timeout = std::time::Duration::from_secs(self.options.timeout_seconds);
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            let loaded = loaded.clone();
            let task = tokio::task::spawn_blocking(move || loaded.embed_one(&input));
            let embedding = match tokio::time::timeout(timeout, task).await {
                Ok(joined) => joined
                    .map_err(|e| InlError::Inference(format!("inference task failed: {e}")))??,
                Err(_) => {
                    return Err(InlError::Timeout(format!(
                        "one multimodal forward pass exceeded {}s",
                        self.options.timeout_seconds
                    )))
                }
            };
            out.push(embedding);
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
        let text = "a <|image|> b <|video|> c <|image|>";
        let expanded = expand(text, "<|image|>", "<|video|>", &images, &videos).unwrap();
        assert_eq!(expanded, "a I0 b V0 c I1");
    }

    #[test]
    fn expand_rejects_missing_media() {
        let error = expand("x <|image|>", "<|image|>", "<|video|>", &[], &[]).unwrap_err();
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
}
