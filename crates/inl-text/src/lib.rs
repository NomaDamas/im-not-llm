//! Dense text embedders (Qwen3-Embedding, EmbeddingGemma 2) on candle.
//!
//! [`TextEmbedder`] runs one model in-process: it owns device/dtype selection,
//! batching, the per-batch deadline and the model lifecycle ([`TextEmbedder::unload`]).
//! Prefix handling is explicit: [`TextEmbedder::embed`] prepends `passage_prefix`
//! and [`TextEmbedder::embed_query`] prepends `query_prefix`, while the
//! [`DenseEngine`](inl_core::DenseEngine) path embeds texts exactly as given, so
//! a server can apply prefixes itself (or not at all).
//!
//! EmbeddingGemma 2 additionally supports Matryoshka truncation via
//! [`TextOptions::output_dim`] and task-prefix helpers ([`Gemma2Task`]).

pub mod gemma2;
mod native;
mod qwen3;
mod session;

pub use gemma2::{gemma2_document, Gemma2Embedder, Gemma2Task};
pub use native::{Architecture, TextEmbedder, TextIdentity, TextOptions};
pub use qwen3::LoadOptions;
pub use session::TextSession;

// Convenience re-exports so callers can configure a `TextOptions` without
// depending on `inl-core` directly.
pub use inl_core::attention::AttentionRequest;
pub use inl_core::device::{DeviceSpec, DtypeSpec};
pub use inl_core::DenseEngine;
