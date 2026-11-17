//! Wiring the `inl` binary to the concrete engine crates.
//!
//! The library (`inl-server`) only knows the [`inl_core::DenseEngine`],
//! [`inl_core::MultimodalEngine`], [`inl_sparse::SparseEngine`] and
//! [`inl_late::LateEngine`] traits; this module is the one place that names
//! the engine crates and turns CLI flags into an [`Engine`] plus the
//! [`InstanceKey`] that names its state directory.

use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_core::{InlError, Result};
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
/// No `--model` is the dense default and reads nothing. Under `--device ane`
/// the package's manifest decides (packages carry no `config.json`).
pub fn detect_kind(args: &EngineArgs) -> Result<EngineKind> {
    let Some(model) = args.model.as_deref().filter(|model| !model.is_empty()) else {
        return Ok(EngineKind::Dense);
    };
    if args.device.as_deref() == Some(ANE_DEVICE) {
        return Ok(match inl_ane::package_family(model)? {
            Some(family) if family.is_sparse() => EngineKind::Sparse,
            _ => EngineKind::Dense,
        });
    }
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
///
/// `--device ane` runs a dense or sparse model's Core ML package on the
/// Apple Neural Engine (`inl-ane`); its graphs are fp16, so no other dtype is
/// accepted. On Apple silicon it is also the default for models whose
/// package says so ([`effective_device`]).
pub fn build(kind: EngineKind, args: &EngineArgs) -> Result<Engine> {
    let model = args.model_or_default(kind);
    if effective_device(kind, args).as_deref() == Some(ANE_DEVICE) {
        return build_ane(kind, model, args);
    }
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

/// The `--device` that selects the Apple Neural Engine.
pub const ANE_DEVICE: &str = "ane";

/// The Neural Engine serves `model` as `kind` here without being asked: an
/// Apple silicon Mac, no device named (or `auto`), no dtype other than f16
/// asked for, and a pinned package marked default for this model
/// (`KnownPackage::default_on_apple`) of the same kind. `--device metal` /
/// `cpu` opts out.
fn ane_by_default(kind: EngineKind, model: &str, args: &EngineArgs) -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return false;
    }
    if !matches!(args.device.as_deref(), None | Some("auto"))
        || !matches!(args.dtype.as_deref(), None | Some("auto" | "f16"))
    {
        return false;
    }
    inl_ane::known_package(model).is_some_and(|package| {
        package.default_on_apple
            && match kind {
                EngineKind::Dense => !package.family.is_sparse(),
                EngineKind::Sparse => package.family.is_sparse(),
                EngineKind::Multimodal | EngineKind::Late => false,
            }
    })
}

/// The device `build` runs `kind` on: `ane` when the Neural Engine is the
/// default for the model (see [`ane_by_default`]), else what was asked.
pub fn effective_device(kind: EngineKind, args: &EngineArgs) -> Option<String> {
    if ane_by_default(kind, &args.model_or_default(kind), args) {
        Some(ANE_DEVICE.to_string())
    } else {
        args.device.clone()
    }
}

fn build_ane(kind: EngineKind, model: String, args: &EngineArgs) -> Result<Engine> {
    if !matches!(kind, EngineKind::Dense | EngineKind::Sparse) {
        return Err(InlError::Config(format!(
            "--device {ANE_DEVICE} serves dense and sparse text embeddings only, not {}",
            kind.as_str()
        )));
    }
    if !matches!(args.dtype.as_deref(), None | Some("auto" | "f16")) {
        return Err(InlError::Config(format!(
            "--device {ANE_DEVICE} runs fp16 graphs: --dtype must be auto or f16"
        )));
    }
    #[cfg(target_os = "macos")]
    {
        let options = inl_ane::AneOptions {
            id: model,
            cache_dir: args.cache_dir.clone(),
            ..inl_ane::AneOptions::default()
        };
        Ok(match kind {
            EngineKind::Sparse => {
                Engine::sparse(Arc::new(inl_ane::AneSparseEncoder::new(options)?))
            }
            _ => Engine::dense(Arc::new(inl_ane::AneEmbedder::new(options)?)),
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = model;
        Err(InlError::Config(format!(
            "--device {ANE_DEVICE} needs an Apple silicon Mac"
        )))
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

    #[test]
    fn a_neural_engine_package_directory_is_dense_without_a_config() {
        let dir = tempfile::tempdir().unwrap();
        let kind = detect_kind(&EngineArgs {
            model: Some(dir.path().display().to_string()),
            device: Some(ANE_DEVICE.into()),
            ..EngineArgs::default()
        });
        assert_eq!(kind.unwrap(), EngineKind::Dense);
    }

    #[test]
    fn a_sparse_neural_engine_package_serves_sparse() {
        let kind = detect_kind(&EngineArgs {
            model: Some(DEFAULT_SPARSE_MODEL.into()),
            device: Some(ANE_DEVICE.into()),
            ..EngineArgs::default()
        });
        assert_eq!(kind.unwrap(), EngineKind::Sparse);
    }

    fn ane(kind: EngineKind, dtype: Option<&str>) -> Result<Engine> {
        build(
            kind,
            &EngineArgs {
                device: Some("ane".into()),
                dtype: dtype.map(str::to_string),
                ..EngineArgs::default()
            },
        )
    }

    #[test]
    fn the_neural_engine_runs_its_fp16_text_graphs_and_nothing_else() {
        for kind in [EngineKind::Dense, EngineKind::Sparse] {
            let error = ane(kind, Some("f32")).err().unwrap();
            assert!(error.to_string().contains("f16"), "{kind:?}: {error}");
        }
        for kind in [EngineKind::Multimodal, EngineKind::Late] {
            let error = ane(kind, None).err().unwrap();
            assert!(
                error.to_string().contains("dense and sparse"),
                "{kind:?}: {error}"
            );
        }
    }

    fn device(
        kind: EngineKind,
        model: Option<&str>,
        device: Option<&str>,
        dtype: Option<&str>,
    ) -> Option<String> {
        effective_device(
            kind,
            &EngineArgs {
                model: model.map(str::to_string),
                device: device.map(str::to_string),
                dtype: dtype.map(str::to_string),
                ..EngineArgs::default()
            },
        )
    }

    #[test]
    fn apple_silicon_serves_the_default_models_on_the_neural_engine_unless_told_otherwise() {
        let on_apple = cfg!(all(target_os = "macos", target_arch = "aarch64"));
        let ane = on_apple.then(|| ANE_DEVICE.to_string());
        assert_eq!(device(EngineKind::Dense, None, None, None), ane);
        // Off Apple silicon the requested device passes through untouched.
        let passed_through =
            |device: Option<&str>| (!on_apple).then(|| device.map(str::to_string)).flatten();
        assert_eq!(
            device(EngineKind::Sparse, None, Some("auto"), None),
            ane.clone().or_else(|| passed_through(Some("auto")))
        );
        assert_eq!(device(EngineKind::Sparse, None, None, Some("f16")), ane);
        // An explicit device or a dtype the graphs do not run keeps candle.
        assert_eq!(
            device(EngineKind::Dense, None, Some("metal"), None).as_deref(),
            Some("metal")
        );
        assert_eq!(device(EngineKind::Dense, None, None, Some("f32")), None);
        // EmbeddingGemma 2 stays out of fp16 unless asked for.
        assert_eq!(
            device(
                EngineKind::Dense,
                Some("google/embeddinggemma-2"),
                None,
                None
            ),
            None
        );
        assert_eq!(device(EngineKind::Multimodal, None, None, None), None);
        // A sparse package never serves dense, nor the reverse.
        assert_eq!(
            device(EngineKind::Dense, Some(DEFAULT_SPARSE_MODEL), None, None),
            None
        );
        assert_eq!(
            device(EngineKind::Sparse, Some(DEFAULT_DENSE_MODEL), None, None),
            None
        );
    }
}
