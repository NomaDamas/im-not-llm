//! Node-API binding for the im-not-llm engines.
//!
//! It exposes the in-process engines (`inl_text::TextSession` and
//! `inl_sparse::SparseEncoder`) and the loopback daemon lifecycle
//! (`inl_server::Daemon` / `DaemonClient`) to JavaScript. Inference runs on the
//! tokio blocking pool, never on the JS thread; `AbortSignal` support and
//! `error.code` live in the hand-written `index.js` wrapper.
//!
//! This crate only converts types and maps errors; it never reimplements the
//! engine. The engine contracts come from `inl-core::engine`.

use inl_core::attention::AttentionRequest;
use inl_core::device::{DeviceSpec, DtypeSpec};
use inl_core::{InlError, Result as InlResult};
use inl_server::{
    default_state_dir, instance_key, parse_loopback_host, Engine, EngineKind, Health, InstanceKey,
    ServeOptions,
};
use inl_server::{Daemon as EngineDaemon, DaemonClient as EngineDaemonClient};
use inl_sparse::SparseOptions;
use inl_text::{TextOptions, TextSession};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::watch;

/// Model served when `engineOptions.model` is absent, per kind (matches the
/// `inl serve` CLI).
const DEFAULT_DENSE_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";
const DEFAULT_SPARSE_MODEL: &str = "nlpai-lab/KURE-Sparse-nano";

/// Failure classes a host can tell apart. napi-rs only lets a native error
/// carry a `Status`, so the class travels as a `<CODE>: ` message prefix that
/// the JS wrapper turns into `error.code`.
#[derive(Clone, Copy)]
enum ErrorCode {
    Config,
    Inference,
    Timeout,
    InvalidInput,
    Aborted,
    Other,
}

impl ErrorCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Config => "ERR_INL_CONFIG",
            Self::Inference => "ERR_INL_INFERENCE",
            Self::Timeout => "ERR_INL_TIMEOUT",
            Self::InvalidInput => "ERR_INL_INVALID_INPUT",
            Self::Aborted => "ERR_INL_ABORTED",
            Self::Other => "ERR_INL",
        }
    }

    fn status(self) -> Status {
        match self {
            Self::Config | Self::InvalidInput => Status::InvalidArg,
            Self::Aborted => Status::Cancelled,
            _ => Status::GenericFailure,
        }
    }

    fn error(self, message: impl std::fmt::Display) -> Error {
        Error::new(self.status(), format!("{}: {message}", self.as_str()))
    }
}

/// Map an engine error onto its JS class via [`InlError::code`].
fn to_napi(error: InlError) -> Error {
    let code = match error.code() {
        "config" => ErrorCode::Config,
        "timeout" => ErrorCode::Timeout,
        "invalid_input" => ErrorCode::InvalidInput,
        "inference" => ErrorCode::Inference,
        _ => ErrorCode::Other,
    };
    code.error(error)
}

fn parse_kind(value: Option<&str>) -> InlResult<EngineKind> {
    match value {
        None | Some("dense") => Ok(EngineKind::Dense),
        Some("sparse") => Ok(EngineKind::Sparse),
        Some("multimodal") => Ok(EngineKind::Multimodal),
        Some(other) => Err(InlError::Config(format!(
            "invalid engine kind '{other}': expected dense, sparse, or multimodal"
        ))),
    }
}

fn default_model(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::Sparse => DEFAULT_SPARSE_MODEL,
        EngineKind::Dense | EngineKind::Multimodal => DEFAULT_DENSE_MODEL,
    }
}

/// The checkpoint's own markers the server applies when the caller sends
/// `input_type` (matches the `inl serve` CLI: none for the dense encoders).
fn default_prefixes(kind: EngineKind) -> (Option<String>, Option<String>) {
    match kind {
        EngineKind::Sparse => (Some("[Q] ".to_string()), Some("[D] ".to_string())),
        EngineKind::Dense | EngineKind::Multimodal => (None, None),
    }
}

fn seconds(value: u32) -> u64 {
    u64::from(value)
}

/// Cancels the calls it was passed to. Used by the JS wrapper to implement
/// `AbortSignal`; not part of the public API.
#[napi]
pub struct AbortHandle {
    tx: Arc<watch::Sender<bool>>,
}

#[napi]
impl AbortHandle {
    #[napi(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            tx: Arc::new(watch::channel(false).0),
        }
    }

    /// Stop waiting for the call. Remaining forward passes are not started; a
    /// pass already running finishes in the background (it cannot be
    /// interrupted).
    #[napi]
    pub fn abort(&self) {
        self.tx.send_replace(true);
    }
}

async fn abortable<T>(
    abort: Option<&AbortHandle>,
    work: impl Future<Output = InlResult<T>>,
) -> Result<T> {
    let Some(abort) = abort else {
        return work.await.map_err(to_napi);
    };
    let mut aborted = abort.tx.subscribe();
    let aborted_error = || ErrorCode::Aborted.error("inference call was aborted");
    if *aborted.borrow() {
        return Err(aborted_error());
    }
    tokio::select! {
        biased;
        _ = aborted.wait_for(|aborted| *aborted) => Err(aborted_error()),
        result = work => result.map_err(to_napi),
    }
}

// ---- dense text ------------------------------------------------------------

/// Settings for a [`TextEmbedder`]. Omitted fields take the engine's defaults.
#[napi(object)]
#[derive(Default)]
pub struct TextEmbedderOptions {
    /// A Hugging Face repo id or a local model directory (required).
    pub id: Option<String>,
    /// `auto` (default), `cpu`, `metal` or `cuda`.
    pub device: Option<String>,
    /// `auto` (default), `f32`, `f16` or `bf16`.
    pub dtype: Option<String>,
    /// `auto` (default), `eager`, `tiled` or `fused`.
    pub attention: Option<String>,
    /// Tokens kept per input (default 2048).
    pub max_length: Option<u32>,
    /// Most inputs per forward pass (default 64).
    pub batch_size: Option<u32>,
    /// Cap on padded tokens per forward pass (default 4096).
    pub max_batch_tokens: Option<u32>,
    /// Default deadline for one forward pass, in seconds (default 60).
    pub timeout_seconds: Option<u32>,
    /// Deadline for one forward pass; overrides `timeoutSeconds`.
    pub per_batch_timeout_seconds: Option<u32>,
    /// Model cache directory (else the shared `HF_HOME` cache).
    pub cache_dir: Option<String>,
    /// Prepended by `embedQuery`.
    pub query_prefix: Option<String>,
    /// Prepended by `embed`.
    pub passage_prefix: Option<String>,
    /// Matryoshka output dimension (EmbeddingGemma 2 only).
    pub output_dim: Option<u32>,
    /// Release the model after this many idle seconds; it reloads on the next
    /// call. Default: stay resident until `unload()` / `dispose()`.
    pub idle_unload_seconds: Option<u32>,
}

/// The dense embedding space the vectors belong to.
#[napi(object)]
pub struct TextIdentity {
    /// The model id as configured: a hub repo id or local directory.
    pub id: String,
    pub dimension: u32,
    /// Resolved precision: `f32`, `f16` or `bf16`.
    pub dtype: String,
    /// Resolved device: `cpu`, `metal` or `cuda`.
    pub device: String,
    pub max_length: u32,
}

fn text_session(options: TextEmbedderOptions) -> InlResult<TextSession> {
    let mut text = TextOptions {
        id: options.id.unwrap_or_default(),
        device: DeviceSpec::parse(options.device.as_deref())?,
        dtype: DtypeSpec::parse(options.dtype.as_deref())?,
        attention: AttentionRequest::parse(options.attention.as_deref())?,
        ..TextOptions::default()
    };
    if let Some(value) = options.max_length {
        text.max_length = value as usize;
    }
    if let Some(value) = options.batch_size {
        text.batch_size = value as usize;
    }
    if let Some(value) = options.max_batch_tokens {
        text.max_batch_tokens = value as usize;
    }
    if let Some(value) = options.timeout_seconds {
        text.timeout_seconds = seconds(value);
    }
    text.per_batch_timeout_seconds = options.per_batch_timeout_seconds.map(seconds);
    text.cache_dir = options.cache_dir.map(PathBuf::from);
    text.query_prefix = options.query_prefix;
    text.passage_prefix = options.passage_prefix;
    text.output_dim = options.output_dim.map(|value| value as usize);
    let idle = options
        .idle_unload_seconds
        .map(|value| Duration::from_secs(seconds(value)));
    TextSession::new(text, idle)
}

/// In-process dense text embedder. The model loads on first use and stays
/// resident until `unload()`, `dispose()` or `idleUnloadSeconds`.
#[napi]
pub struct TextEmbedder {
    session: Arc<TextSession>,
}

#[napi]
impl TextEmbedder {
    #[napi(constructor)]
    pub fn new(options: Option<TextEmbedderOptions>) -> Result<Self> {
        let session = text_session(options.unwrap_or_default()).map_err(to_napi)?;
        Ok(Self {
            session: Arc::new(session),
        })
    }

    /// Model id, dimension, dtype and device. Loads the model if it is not
    /// resident, since those are only known once it is.
    #[napi]
    pub async fn identity(&self) -> Result<TextIdentity> {
        let identity = self.session.identity().await.map_err(to_napi)?;
        Ok(TextIdentity {
            id: identity.id,
            dimension: identity.dimension as u32,
            dtype: identity.dtype.to_string(),
            device: identity.device.to_string(),
            max_length: identity.max_length as u32,
        })
    }

    /// Embed passages (the passage prefix is applied): unit-length vectors in
    /// input order.
    #[napi]
    pub async fn embed(
        &self,
        texts: Vec<String>,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<Float32Array>> {
        let vectors = abortable(abort, self.session.embed(&texts)).await?;
        Ok(vectors.into_iter().map(Float32Array::new).collect())
    }

    /// Embed a search query (the query prefix is applied).
    #[napi]
    pub async fn embed_query(
        &self,
        text: String,
        abort: Option<&AbortHandle>,
    ) -> Result<Float32Array> {
        let vector = abortable(abort, self.session.embed_query(&text)).await?;
        Ok(Float32Array::new(vector))
    }

    /// True while the model weights are resident.
    #[napi]
    pub fn is_loaded(&self) -> bool {
        self.session.is_loaded()
    }

    /// Release the model weights; the next call reloads them. Resolves to
    /// whether a model was resident.
    #[napi]
    pub async fn unload(&self) -> bool {
        self.session.unload().await
    }

    /// Release the model and reject every later call.
    #[napi]
    pub async fn dispose(&self) {
        self.session.dispose().await;
    }
}

// ---- learned sparse --------------------------------------------------------

/// Settings for a [`SparseEncoder`]. Omitted fields take the encoder defaults.
#[napi(object)]
#[derive(Default)]
pub struct SparseEncoderOptions {
    /// A Hugging Face repo id or a local model directory (required).
    pub id: Option<String>,
    /// `auto` (default), `cpu`, `metal` or `cuda`.
    pub device: Option<String>,
    /// `auto` (default), `f32`, `f16` or `bf16`.
    pub dtype: Option<String>,
    /// Context length in tokens (default 512).
    pub max_length: Option<u32>,
    /// Inputs per forward pass (default 64).
    pub batch_size: Option<u32>,
    /// Cap on padded tokens per forward pass (default 1024).
    pub max_batch_tokens: Option<u32>,
    /// Default deadline for one forward pass, in seconds (default 60).
    pub timeout_seconds: Option<u32>,
    /// Deadline for one forward pass; overrides `timeoutSeconds`.
    pub per_batch_timeout_seconds: Option<u32>,
    /// Model cache directory (else the shared `HF_HOME` cache).
    pub cache_dir: Option<String>,
    /// Query marker prepended by `encodeQuery` (default `[Q] `).
    pub query_prefix: Option<String>,
    /// Passage marker prepended by `encode` (default `[D] `).
    pub passage_prefix: Option<String>,
}

/// The sparse space the encoder produces.
#[napi(object)]
pub struct SparseIdentity {
    /// The model id as configured: a hub repo id or local directory.
    pub id: String,
    /// Vocabulary size of the sparse space.
    pub dimension: u32,
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub max_length: u32,
}

/// One sparse term-weight vector: `indices` strictly ascending, `values`
/// finite and positive, parallel arrays.
#[napi(object)]
pub struct SparseVector {
    pub indices: Uint32Array,
    pub values: Float32Array,
}

impl From<inl_sparse::SparseVector> for SparseVector {
    fn from(vector: inl_sparse::SparseVector) -> Self {
        Self {
            indices: Uint32Array::new(vector.indices().to_vec()),
            values: Float32Array::new(vector.values().to_vec()),
        }
    }
}

fn sparse_encoder(options: SparseEncoderOptions) -> InlResult<inl_sparse::SparseEncoder> {
    let mut sparse = SparseOptions {
        id: options.id.unwrap_or_default(),
        device: options.device,
        dtype: options.dtype,
        ..SparseOptions::default()
    };
    if let Some(value) = options.max_length {
        sparse.max_length = value as usize;
    }
    if let Some(value) = options.batch_size {
        sparse.batch_size = value as usize;
    }
    if let Some(value) = options.max_batch_tokens {
        sparse.max_batch_tokens = value as usize;
    }
    if let Some(value) = options.timeout_seconds {
        sparse.timeout_seconds = seconds(value);
    }
    sparse.per_batch_timeout_seconds = options.per_batch_timeout_seconds.map(seconds);
    sparse.cache_dir = options.cache_dir.map(PathBuf::from);
    // `None` keeps the checkpoint's own `[Q] `/`[D] ` markers.
    if options.query_prefix.is_some() {
        sparse.query_prefix = options.query_prefix;
    }
    if options.passage_prefix.is_some() {
        sparse.passage_prefix = options.passage_prefix;
    }
    inl_sparse::SparseEncoder::new(sparse)
}

/// In-process learned-sparse (SPLADE) encoder. The model loads on first use and
/// stays resident until `unload()`.
#[napi]
pub struct SparseEncoder {
    inner: Arc<inl_sparse::SparseEncoder>,
}

#[napi]
impl SparseEncoder {
    #[napi(constructor)]
    pub fn new(options: Option<SparseEncoderOptions>) -> Result<Self> {
        let encoder = sparse_encoder(options.unwrap_or_default()).map_err(to_napi)?;
        Ok(Self {
            inner: Arc::new(encoder),
        })
    }

    /// The sparse space this encoder produces. Loads the model if needed.
    #[napi]
    pub async fn identity(&self) -> Result<SparseIdentity> {
        let identity = self.inner.identity().await.map_err(to_napi)?;
        Ok(SparseIdentity {
            id: identity.model_id,
            dimension: identity.dimension as u32,
            dtype: identity.dtype,
            device: identity.device,
            attention: identity.attention,
            max_length: identity.max_length as u32,
        })
    }

    /// Encode documents (the passage marker is prepended to every input).
    #[napi]
    pub async fn encode(
        &self,
        texts: Vec<String>,
        abort: Option<&AbortHandle>,
    ) -> Result<Vec<SparseVector>> {
        let vectors = abortable(abort, self.inner.encode(&texts)).await?;
        Ok(vectors.into_iter().map(SparseVector::from).collect())
    }

    /// Encode one query (the query marker is prepended).
    #[napi]
    pub async fn encode_query(
        &self,
        text: String,
        abort: Option<&AbortHandle>,
    ) -> Result<SparseVector> {
        let vector = abortable(abort, self.inner.encode_query(&text)).await?;
        Ok(SparseVector::from(vector))
    }

    /// True while the model weights are resident.
    #[napi]
    pub fn is_loaded(&self) -> bool {
        self.inner.is_loaded()
    }

    /// Release the model weights; the next call reloads them. Resolves to
    /// whether a model was resident.
    #[napi]
    pub async fn unload(&self) -> bool {
        self.inner.unload().await
    }
}

// ---- daemon lifecycle ------------------------------------------------------

/// Which engine a daemon serves and how it is built.
#[napi(object)]
#[derive(Default)]
pub struct EngineOptions {
    /// `dense` (default), `sparse` or `multimodal`.
    pub kind: Option<String>,
    /// Hub repo id or local directory; omitted uses the kind's default.
    pub model: Option<String>,
    /// `auto` (default), `cpu`, `metal` or `cuda`.
    pub device: Option<String>,
    /// `auto` (default), `f32`, `f16` or `bf16`.
    pub dtype: Option<String>,
    /// `auto` (default), `eager`, `tiled` or `fused` (dense only).
    pub attention: Option<String>,
    /// Context length in tokens; omitted uses the engine default.
    pub max_length: Option<u32>,
    /// Model cache directory (else the shared `HF_HOME` cache).
    pub cache_dir: Option<String>,
    /// Multimodal only: towers to load, from `text`, `image`, `video` and
    /// `audio`; omitted loads every tower the checkpoint has.
    pub modalities: Option<Vec<String>>,
}

/// Loopback server settings for [`Daemon::spawn`].
#[napi(object)]
#[derive(Default)]
pub struct ServerOptions {
    /// Loopback host; defaults to `127.0.0.1`.
    pub host: Option<String>,
    /// TCP port; `0` (default) picks a free one.
    pub port: Option<u32>,
    /// Stop after this many idle seconds; omitted serves until stopped.
    pub idle_timeout_seconds: Option<u32>,
    /// Most inputs per request (default 32).
    pub max_inputs: Option<u32>,
    /// Largest request body in bytes (default 8 MiB).
    pub max_request_bytes: Option<u32>,
    /// Directory for the `lock`/`pid`/`host`/`port` files; omitted uses the
    /// per-user default for the instance.
    pub state_dir: Option<String>,
    /// Prefix applied to `input_type: "query"` requests; defaults per kind.
    pub query_prefix: Option<String>,
    /// Prefix applied to `input_type: "passage"` requests; defaults per kind.
    pub passage_prefix: Option<String>,
}

/// Where a [`DaemonClient`] looks for a served instance.
#[napi(object)]
#[derive(Default)]
pub struct DiscoverOptions {
    /// Hub repo id or local directory; required (identifies the instance).
    pub model: Option<String>,
    /// `dense` (default), `sparse` or `multimodal`.
    pub kind: Option<String>,
    /// `auto` (default), `cpu`, `metal` or `cuda`.
    pub device: Option<String>,
    /// `auto` (default), `f32`, `f16` or `bf16`.
    pub dtype: Option<String>,
    /// Directory holding the discovery files; omitted uses the per-user
    /// default for the instance.
    pub state_dir: Option<String>,
}

/// A `GET /health` body.
#[napi(object)]
pub struct DaemonHealth {
    /// `loading`, `ok` or `stopping`.
    pub status: String,
    /// `dense`, `sparse` or `multimodal`.
    pub kind: String,
    pub model_id: String,
    pub dimension: u32,
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub max_length: u32,
    pub modalities: Vec<String>,
    pub max_inputs: u32,
    pub instance: String,
    pub pid: u32,
    pub host: String,
    pub port: u32,
    pub url: String,
    pub version: String,
    pub uptime_seconds: i64,
    pub requests_total: i64,
    pub loaded: bool,
}

impl From<Health> for DaemonHealth {
    fn from(health: Health) -> Self {
        Self {
            status: health.status,
            kind: health.kind,
            model_id: health.model_id,
            dimension: health.dimension as u32,
            dtype: health.dtype,
            device: health.device,
            attention: health.attention,
            max_length: health.max_length as u32,
            modalities: health.modalities,
            max_inputs: health.max_inputs as u32,
            instance: health.instance,
            pid: health.pid,
            host: health.host,
            port: health.port as u32,
            url: health.url,
            version: health.version,
            uptime_seconds: health.uptime_seconds as i64,
            requests_total: health.requests_total as i64,
            loaded: health.loaded,
        }
    }
}

fn engine_option_model(options: &EngineOptions, kind: EngineKind) -> String {
    options
        .model
        .clone()
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| default_model(kind).to_string())
}

/// Build the neutral instance key (state-directory name) for a configuration.
fn engine_instance(options: &EngineOptions, kind: EngineKind) -> InlResult<String> {
    instance_key(&InstanceKey {
        model: engine_option_model(options, kind),
        device: options.device.clone(),
        dtype: options.dtype.clone(),
        // Matching the `inl serve` CLI: prefixes and batching knobs are not
        // part of the instance identity.
        attention: None,
        max_length: None,
        kind,
    })
}

fn build_engine(options: &EngineOptions, kind: EngineKind) -> InlResult<Engine> {
    let model = engine_option_model(options, kind);
    match kind {
        EngineKind::Dense => {
            let mut text = TextOptions {
                id: model,
                device: DeviceSpec::parse(options.device.as_deref())?,
                dtype: DtypeSpec::parse(options.dtype.as_deref())?,
                attention: AttentionRequest::parse(options.attention.as_deref())?,
                ..TextOptions::default()
            };
            if let Some(value) = options.max_length {
                text.max_length = value as usize;
            }
            text.cache_dir = options.cache_dir.clone().map(PathBuf::from);
            Ok(Engine::dense(Arc::new(inl_text::TextEmbedder::new(text)?)))
        }
        EngineKind::Sparse => {
            let mut sparse = SparseOptions {
                id: model,
                device: options.device.clone(),
                dtype: options.dtype.clone(),
                ..SparseOptions::default()
            };
            if let Some(value) = options.max_length {
                sparse.max_length = value as usize;
            }
            sparse.cache_dir = options.cache_dir.clone().map(PathBuf::from);
            Ok(Engine::sparse(Arc::new(inl_sparse::SparseEncoder::new(
                sparse,
            )?)))
        }
        EngineKind::Multimodal => {
            let mut multimodal = inl_vision::MultimodalOptions {
                id: model,
                device: DeviceSpec::parse(options.device.as_deref())?,
                dtype: DtypeSpec::parse(options.dtype.as_deref())?,
                attention: AttentionRequest::parse(options.attention.as_deref())?,
                ..inl_vision::MultimodalOptions::default()
            };
            if let Some(value) = options.max_length {
                multimodal.max_length = value as usize;
            }
            multimodal.cache_dir = options.cache_dir.clone().map(PathBuf::from);
            multimodal.modalities = options
                .modalities
                .as_deref()
                .map(inl_core::Modality::parse_list)
                .transpose()?;
            Ok(Engine::multimodal(Arc::new(
                inl_vision::MultimodalEmbedder::new(multimodal),
            )))
        }
    }
}

fn serve_options(
    engine: &EngineOptions,
    server: ServerOptions,
    kind: EngineKind,
) -> InlResult<ServeOptions> {
    let instance = engine_instance(engine, kind)?;
    let state_dir = server
        .state_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| default_state_dir(&instance));
    let host = match server.host.as_deref() {
        Some(host) => parse_loopback_host(host)?,
        None => "127.0.0.1".parse().expect("valid loopback literal"),
    };
    let (query_prefix, passage_prefix) = default_prefixes(kind);
    Ok(ServeOptions {
        host,
        port: server.port.unwrap_or(0) as u16,
        idle_timeout: server
            .idle_timeout_seconds
            .map(|value| Duration::from_secs(seconds(value))),
        max_inputs: server.max_inputs.map(|value| value as usize).unwrap_or(32),
        max_request_bytes: server
            .max_request_bytes
            .map(|value| value as usize)
            .unwrap_or(8 * 1024 * 1024),
        state_dir,
        instance,
        query_prefix: server.query_prefix.or(query_prefix),
        passage_prefix: server.passage_prefix.or(passage_prefix),
    })
}

/// An in-process server started by [`Daemon::spawn`]: it owns the listener and
/// the state files and can be stopped or awaited.
#[napi]
pub struct Daemon {
    inner: Mutex<Option<EngineDaemon>>,
}

impl Daemon {
    fn take(&self) -> Result<EngineDaemon> {
        self.inner
            .lock()
            .map_err(|_| ErrorCode::Other.error("daemon lock poisoned"))?
            .take()
            .ok_or_else(|| ErrorCode::Other.error("the daemon has already been stopped"))
    }

    fn with<R>(&self, f: impl FnOnce(&EngineDaemon) -> R) -> Result<R> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| ErrorCode::Other.error("daemon lock poisoned"))?;
        guard
            .as_ref()
            .map(f)
            .ok_or_else(|| ErrorCode::Other.error("the daemon has already been stopped"))
    }
}

#[napi]
impl Daemon {
    /// Serve `serverOptions` (with the engine from `engineOptions`) in this
    /// process, or adopt the server already serving that instance. The
    /// returned daemon's `url`/`health` are usable immediately.
    #[napi]
    pub async fn spawn(engine: EngineOptions, server: ServerOptions) -> Result<Daemon> {
        let kind = parse_kind(engine.kind.as_deref()).map_err(to_napi)?;
        let options = serve_options(&engine, server, kind).map_err(to_napi)?;
        let built = build_engine(&engine, kind).map_err(to_napi)?;
        let daemon = EngineDaemon::spawn(built, options).await.map_err(to_napi)?;
        Ok(Daemon {
            inner: Mutex::new(Some(daemon)),
        })
    }

    /// The server's base URL.
    #[napi]
    pub fn url(&self) -> Result<String> {
        self.with(EngineDaemon::url)
    }

    /// The server's current health.
    #[napi]
    pub fn health(&self) -> Result<DaemonHealth> {
        self.with(|daemon| DaemonHealth::from(daemon.health()))
    }

    /// Stop the server and wait for it to be gone. Resolves to the stop reason
    /// (`signal`, `idle` or `wedged`).
    #[napi]
    pub async fn stop(&self) -> Result<String> {
        let daemon = self.take()?;
        daemon
            .stop()
            .await
            .map(|reason| reason.as_str().to_string())
            .map_err(to_napi)
    }

    /// Wait for the server to stop on its own. Resolves to the stop reason.
    #[napi]
    pub async fn wait(&self) -> Result<String> {
        let daemon = self.take()?;
        daemon
            .wait()
            .await
            .map(|reason| reason.as_str().to_string())
            .map_err(to_napi)
    }
}

/// Out-of-process control of a served instance: discovery files plus HTTP.
#[napi]
pub struct DaemonClient {
    inner: EngineDaemonClient,
}

#[napi]
impl DaemonClient {
    /// A client for the server recorded in the instance's state directory, or
    /// `null` when no server has written its discovery files there.
    #[napi]
    pub fn discover(options: DiscoverOptions) -> Result<Option<DaemonClient>> {
        let kind = parse_kind(options.kind.as_deref()).map_err(to_napi)?;
        let model = options
            .model
            .clone()
            .filter(|model| !model.is_empty())
            .unwrap_or_else(|| default_model(kind).to_string());
        let instance = instance_key(&InstanceKey {
            model,
            device: options.device.clone(),
            dtype: options.dtype.clone(),
            attention: None,
            max_length: None,
            kind,
        })
        .map_err(to_napi)?;
        let state_dir = options
            .state_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| default_state_dir(&instance));
        Ok(EngineDaemonClient::discover(&state_dir, &instance).map(|inner| DaemonClient { inner }))
    }

    /// The server's base URL.
    #[napi]
    pub fn url(&self) -> String {
        self.inner.url().to_string()
    }

    /// The server's health when it answers as this instance, whatever its
    /// status; `null` when it does not answer.
    #[napi]
    pub async fn health(&self) -> Option<DaemonHealth> {
        self.inner.health().await.map(DaemonHealth::from)
    }

    /// Poll `/health` until the server is ready or `timeoutMs` elapses.
    #[napi]
    pub async fn wait_ready(&self, timeout_ms: u32) -> Result<DaemonHealth> {
        self.inner
            .wait_ready(Duration::from_millis(u64::from(timeout_ms)))
            .await
            .map(DaemonHealth::from)
            .map_err(to_napi)
    }

    /// `POST /admin/shutdown`: ask the server to drain and exit.
    #[napi]
    pub async fn stop(&self) -> Result<()> {
        self.inner.stop().await.map_err(to_napi)
    }

    /// `POST /admin/unload`: release the model weights; true when a model was
    /// resident.
    #[napi]
    pub async fn unload(&self) -> Result<bool> {
        self.inner.unload().await.map_err(to_napi)
    }
}
