//! What a server (or any host) needs from an inference engine. Engines live in
//! their own crates (`inl-text`, `inl-sparse`, `inl-vision`) and implement
//! these; `inl-server` only ever sees the traits.

use crate::error::{InlError, Result};
use async_trait::async_trait;

/// What a loaded engine reports about itself. Also the body of `/health`'s
/// `model` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineInfo {
    /// Hub repo id or local directory, without a provider prefix.
    pub model_id: String,
    /// Output vector length (dense) or vocabulary size (sparse).
    pub dimension: usize,
    /// `f32`, `f16` or `bf16`.
    pub dtype: String,
    /// `cpu`, `metal` or `cuda:N`.
    pub device: String,
    /// `eager`, `tiled` or `fused`.
    pub attention: String,
    /// Context length in tokens.
    pub max_length: usize,
    /// Which inputs the engine accepts.
    pub modalities: Vec<Modality>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    /// Parse modality names (`text`, `image`, `video`, `audio`), as a
    /// `--modalities` flag or a binding option gives them.
    pub fn parse_list<S: AsRef<str>>(names: &[S]) -> Result<Vec<Self>> {
        names
            .iter()
            .map(|name| {
                let name = name.as_ref().trim();
                [Self::Text, Self::Image, Self::Video, Self::Audio]
                    .into_iter()
                    .find(|modality| modality.as_str() == name)
                    .ok_or_else(|| {
                        InlError::Config(format!(
                            "unknown modality {name:?}: expected text, image, video or audio"
                        ))
                    })
            })
            .collect()
    }
}

/// A dense text embedder behind a server.
#[async_trait]
pub trait DenseEngine: Send + Sync {
    /// Load the model (once) and describe it.
    async fn load(&self) -> Result<EngineInfo>;

    /// Tokens per input, as the model would see them before truncation.
    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>>;

    /// Embed `texts` exactly as given (no prefixes added): unit-length
    /// vectors in input order.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Release the weights; the next call reloads. True when a model was
    /// resident.
    async fn unload(&self) -> bool;

    fn is_loaded(&self) -> bool;
}

/// One media item, as raw encoded bytes (PNG/JPEG/WebP/... for images,
/// MP4/WebM/... for video, WAV/MP3/... for audio). The engine decodes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaBytes {
    pub bytes: Vec<u8>,
    /// MIME type when the caller knows it (`image/png`); engines sniff the
    /// bytes when absent.
    pub mime: Option<String>,
}

/// One embedding input. A text-only input is a plain text embedding; media
/// are consumed in order by the `<|image|>` / `<|video|>` / `<|audio|>`
/// placeholders in `text` (EmbeddingGemma 2's interleaving). When `text` is
/// `None` and there is exactly one media item, the input is that item alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub images: Vec<MediaBytes>,
    pub videos: Vec<MediaBytes>,
    pub audios: Vec<MediaBytes>,
}

/// An embedder over text, images, video and audio into one shared space.
#[async_trait]
pub trait MultimodalEngine: Send + Sync {
    /// Load the model (once) and describe it. `info.modalities` lists what
    /// this build/checkpoint accepts; an input using another is
    /// [`crate::InlError::InvalidInput`].
    async fn load(&self) -> Result<EngineInfo>;

    /// Embed `inputs`: unit-length vectors in input order.
    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>>;

    /// Release the weights; the next call reloads. True when a model was
    /// resident.
    async fn unload(&self) -> bool;

    fn is_loaded(&self) -> bool;
}
