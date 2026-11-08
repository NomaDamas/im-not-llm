//! What a server (or any host) needs from an inference engine. Engines live in
//! their own crates (`inl-text`, `inl-sparse`, `inl-vision`) and implement
//! these; `inl-server` only ever sees the traits.

use crate::error::Result;
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
