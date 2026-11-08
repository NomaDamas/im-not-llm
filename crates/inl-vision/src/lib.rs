//! Multimodal (image / video) embedding for EmbeddingGemma 2.
//!
//! [`MultimodalEmbedder`] loads the checkpoint once and embeds text, images and
//! video into one shared space. It implements [`inl_core::MultimodalEngine`].

pub mod engine;
pub mod media;
pub mod preprocess;
pub mod video;
pub mod vision;

pub use engine::{MultimodalEmbedder, MultimodalOptions};
pub use media::RgbImage;
pub use preprocess::{decode, preprocess_rgb, PreprocessedImage};
pub use vision::{AttentionImpl, VisionBatch, VisionConfig, VisionOutput, VisionTower};

pub use inl_core::attention::AttentionRequest;
pub use inl_core::device::{DeviceSpec, DtypeSpec};
pub use inl_core::{Modality, MultimodalEngine, MultimodalInput};
