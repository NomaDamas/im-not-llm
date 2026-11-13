//! Wiring the `inl` binary to the concrete engine crates.
//!
//! The library (`inl-server`) only knows the [`inl_core::DenseEngine`],
//! [`inl_core::MultimodalEngine`], [`inl_sparse::SparseEngine`] and
//! [`inl_late::LateEngine`] traits; this module is the one place that names
//! the engine crates and turns CLI flags into an [`Engine`] plus the
//! [`InstanceKey`] that names its state directory.

use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_core::Result;
use inl_server::{Engine, EngineKind, InstanceKey};
use std::path::PathBuf;
use std::sync::Arc;

/// Model served when `--model` is absent, per engine kind.
pub const DEFAULT_DENSE_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";
pub const DEFAULT_SPARSE_MODEL: &str = "nlpai-lab/KURE-Sparse-nano";
pub const DEFAULT_MULTIMODAL_MODEL: &str = "google/embeddinggemma-2";
pub const DEFAULT_LATE_MODEL: &str = inl_late::DEFAULT_MODEL_ID;

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
    /// Multimodal towers to load (`text`, `image`, `video`, `audio`); `None`
    /// loads every tower the checkpoint has.
    pub modalities: Option<Vec<String>>,
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
            // The late encoder's context comes from the checkpoint.
            max_length: match kind {
                EngineKind::Dense => Some(inl_text::TextOptions::default().max_length),
                EngineKind::Sparse => Some(inl_sparse::SparseOptions::default().max_length),
                EngineKind::Multimodal => Some(inl_vision::MultimodalOptions::default().max_length),
                EngineKind::Late => inl_late::LateOptions::default().max_length,
            },
            kind,
        })
    }
}

pub fn default_model(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::Dense => DEFAULT_DENSE_MODEL,
        EngineKind::Sparse => DEFAULT_SPARSE_MODEL,
        EngineKind::Multimodal => DEFAULT_MULTIMODAL_MODEL,
        EngineKind::Late => DEFAULT_LATE_MODEL,
    }
}

/// The engine kind a `--model` serves when no kind flag is given, read from
/// its `config.json`: a ModernBERT checkpoint is a SPLADE encoder (the only
/// ModernBERT engine), a Qwen3.5 checkpoint a late-interaction encoder (the
/// only Qwen3.5 engine), anything else a dense text encoder. Multimodal stays
/// opt-in because the same EmbeddingGemma 2 checkpoint also serves dense.
/// No `--model` is the dense default and reads nothing.
pub fn detect_kind(args: &EngineArgs) -> Result<EngineKind> {
    let Some(model) = args.model.as_deref().filter(|model| !model.is_empty()) else {
        return Ok(EngineKind::Dense);
    };
    let cache = inl_core::device::model_cache_dir(args.cache_dir.as_deref());
    let path = inl_core::hub::ModelSource::resolve(model, cache)?.fetch("config.json")?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|error| inl_core::InlError::Config(format!("{model}: config.json: {error}")))?;
    let model_type = &config["model_type"];
    Ok(if model_type == inl_sparse::modernbert::MODEL_TYPE {
        EngineKind::Sparse
    } else if model_type == inl_late::qwen3_5::MODEL_TYPE {
        EngineKind::Late
    } else {
        EngineKind::Dense
    })
}

/// The query/passage prefixes a server applies when the caller sends
/// `input_type`: the checkpoint's own markers, absent for the dense text
/// encoders (their prompts are caller-supplied) and for the late encoder
/// (it adds its checkpoint's prompts per role).
pub fn default_prefixes(kind: EngineKind) -> (Option<String>, Option<String>) {
    match kind {
        EngineKind::Sparse => (Some("[Q] ".to_string()), Some("[D] ".to_string())),
        EngineKind::Dense | EngineKind::Multimodal | EngineKind::Late => (None, None),
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
                modalities: args
                    .modalities
                    .as_deref()
                    .map(inl_core::Modality::parse_list)
                    .transpose()?,
                ..inl_vision::MultimodalOptions::default()
            };
            Ok(Engine::multimodal(Arc::new(
                inl_vision::MultimodalEmbedder::new(options),
            )))
        }
        EngineKind::Late => {
            let options = inl_late::LateOptions {
                id: model,
                device: args.device.clone(),
                dtype: args.dtype.clone(),
                cache_dir: args.cache_dir.clone(),
                ..inl_late::LateOptions::default()
            };
            Ok(Engine::late(Arc::new(inl_late::LateEncoder::new(options)?)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind_of_config(config: &str) -> EngineKind {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        detect_kind(&EngineArgs {
            model: Some(dir.path().display().to_string()),
            ..EngineArgs::default()
        })
        .unwrap()
    }

    #[test]
    fn the_model_config_decides_between_sparse_late_and_dense() {
        assert_eq!(
            kind_of_config(r#"{"model_type":"modernbert"}"#),
            EngineKind::Sparse
        );
        assert_eq!(
            kind_of_config(r#"{"model_type":"qwen3"}"#),
            EngineKind::Dense
        );
        assert_eq!(
            kind_of_config(r#"{"model_type":"qwen3_5"}"#),
            EngineKind::Late
        );
        assert_eq!(
            kind_of_config(r#"{"model_type":"embedding_gemma2"}"#),
            EngineKind::Dense
        );
    }

    #[test]
    fn no_model_is_the_dense_default_without_reading_anything() {
        assert_eq!(
            detect_kind(&EngineArgs::default()).unwrap(),
            EngineKind::Dense
        );
    }

    #[test]
    fn a_model_without_a_config_is_an_error_not_a_guess() {
        let dir = tempfile::tempdir().unwrap();
        let error = detect_kind(&EngineArgs {
            model: Some(dir.path().display().to_string()),
            ..EngineArgs::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("config.json"), "{error}");
    }
}
