//! Wiring the `inl` binary to the concrete engine crates.
//!
//! The library (`inl-server`) only knows the [`inl_core::DenseEngine`],
//! [`inl_core::MultimodalEngine`] and [`inl_sparse::SparseEngine`] traits;
//! this module is the one place that names the engine crates and turns CLI
//! flags into an [`Engine`] plus the [`InstanceKey`] that names its state
//! directory.

use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_core::Result;
use inl_server::{Engine, EngineKind, InstanceKey};
use std::path::PathBuf;
use std::sync::Arc;

/// Model served when `--model` is absent, per engine kind.
pub const DEFAULT_DENSE_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";
pub const DEFAULT_SPARSE_MODEL: &str = "nlpai-lab/KURE-Sparse-nano";
pub const DEFAULT_MULTIMODAL_MODEL: &str = "google/embeddinggemma-2";

/// Engine flags shared by `serve`, `health`, `stop` and `unload`.
#[derive(Debug, Clone, Default)]
pub struct EngineArgs {
    /// Hub repo id or local directory; `None` is the kind's default.
    pub model: Option<String>,
    /// `auto`, `cpu`, `metal` or `cuda`; `None` is `auto`.
    pub device: Option<String>,
    /// `auto`, `f32`, `f16` or `bf16`; `None` is the engine default.
    pub dtype: Option<String>,
    /// Model cache directory; `None` is `INL_CACHE_DIR` / the per-user cache.
    pub cache_dir: Option<PathBuf>,
}

impl EngineArgs {
    /// The model id this invocation uses, defaulted per kind.
    pub fn model_or_default(&self, kind: EngineKind) -> String {
        match &self.model {
            Some(model) if !model.is_empty() => model.clone(),
            _ => default_model(kind).to_string(),
        }
    }

    /// The neutral identity of the model instance (for the state directory).
    pub fn instance_key(&self, kind: EngineKind) -> Result<String> {
        inl_server::instance_key(&InstanceKey {
            model: self.model_or_default(kind),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
            attention: None,
            max_length: None,
            kind,
        })
    }
}

pub fn default_model(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::Dense => DEFAULT_DENSE_MODEL,
        EngineKind::Sparse => DEFAULT_SPARSE_MODEL,
        EngineKind::Multimodal => DEFAULT_MULTIMODAL_MODEL,
    }
}

/// The query/passage prefixes a server applies when the caller sends
/// `input_type`: the checkpoint's own markers, absent for the dense text
/// encoders (their prompts are caller-supplied).
pub fn default_prefixes(kind: EngineKind) -> (Option<String>, Option<String>) {
    match kind {
        EngineKind::Sparse => (Some("[Q] ".to_string()), Some("[D] ".to_string())),
        EngineKind::Dense | EngineKind::Multimodal => (None, None),
    }
}

/// Build the real engine the CLI asked for. Construction is cheap; the model
/// loads lazily on the server's first request (or its `load`).
pub fn build(kind: EngineKind, args: &EngineArgs) -> Result<Engine> {
    let model = args.model_or_default(kind);
    match kind {
        EngineKind::Dense => {
            let options = inl_text::TextOptions {
                id: model,
                device: DeviceSpec::parse(args.device.as_deref())?,
                dtype: DtypeSpec::parse(args.dtype.as_deref())?,
                cache_dir: args.cache_dir.clone(),
                ..inl_text::TextOptions::default()
            };
            let embedder = inl_text::TextEmbedder::new(options)?;
            Ok(Engine::dense(Arc::new(embedder)))
        }
        EngineKind::Sparse => {
            let options = inl_sparse::SparseOptions {
                id: model,
                device: args.device.clone(),
                dtype: args.dtype.clone(),
                cache_dir: args.cache_dir.clone(),
                ..inl_sparse::SparseOptions::default()
            };
            let encoder = inl_sparse::SparseEncoder::new(options)?;
            Ok(Engine::sparse(Arc::new(encoder)))
        }
        EngineKind::Multimodal => {
            let options = inl_vision::MultimodalOptions {
                id: model,
                device: DeviceSpec::parse(args.device.as_deref())?,
                dtype: DtypeSpec::parse(args.dtype.as_deref())?,
                cache_dir: args.cache_dir.clone(),
                ..inl_vision::MultimodalOptions::default()
            };
            Ok(Engine::multimodal(Arc::new(
                inl_vision::MultimodalEmbedder::new(options),
            )))
        }
    }
}
