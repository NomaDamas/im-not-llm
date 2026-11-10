//! `inl serve`: one resident native engine shared over loopback HTTP.
//!
//! Protocols:
//! * TEI: `POST /embed` `{ "inputs", "truncate"? }` -> `[[f32]]`
//! * OpenAI: `POST /v1/embeddings` `{ "model", "input" }` -> `{ "data", ... }`
//! * TEI sparse: `POST /embed_sparse` `{ "inputs" }` -> `[[{index,value}]]`
//! * multimodal: `POST /embed_multimodal` `{ "inputs": [...] }` ->
//!   `{ "embeddings": [[f32]], "model", "dimensions" }`
//! * `GET /health`: model, device, build and liveness.
//!
//! A server runs one engine, reported by `/health` as its `kind`:
//! * `dense` serves `/embed` and `/v1/embeddings`;
//! * `sparse` (a learned-sparse SPLADE encoder) serves `/embed_sparse`;
//! * `multimodal` serves `/embed_multimodal`, `/v1/embeddings` (items may be
//!   media) and `/embed` (text only).
//!
//! A server that does not serve a route answers 404 there.
//!
//! Both text protocols accept an `input_type: "query" | "passage"` extension
//! that applies the configured prefix; without it the text is embedded as
//! sent.
//!
//! The server is unauthenticated, so it only binds loopback addresses. A
//! state directory per model instance holds a `lock` (held for the server's
//! lifetime) and `pid` / `host` / `port` files; a second `start` that finds
//! the lock taken adopts the running server instead of loading another copy
//! of the model. The `lock`-guarded `POST /admin/unload` and
//! `POST /admin/shutdown` routes back the out-of-process lifecycle API in
//! [`DaemonClient`].

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use inl_core::device::cache_root;
use inl_core::{
    DenseEngine, EngineInfo, InlError, MediaBytes, Modality, MultimodalEngine, MultimodalInput,
    Result,
};
use inl_sparse::{SparseEngine, SparseVector};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::future::{Future, IntoFuture};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::Instant;

#[cfg(test)]
mod tests;

/// Default for `--max-inputs`: inputs accepted in one request. The batch
/// planner splits a request into forward passes, so this only bounds how
/// long one request can occupy the model.
pub const DEFAULT_MAX_INPUTS: usize = 512;

/// Default for `--max-request-bytes` (32 MiB).
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Default for `--idle-timeout` in seconds.
pub const DEFAULT_IDLE_TIMEOUT_SECONDS: u64 = 300;

/// How long a second `start` waits for a server that holds the lock (and
/// may still be loading or downloading the model) to answer `/health`.
const ADOPT_WAIT: Duration = Duration::from_secs(600);

/// How long a stopping server waits for in-flight requests.
const DRAIN_WAIT: Duration = Duration::from_secs(30);

const STATUS_LOADING: &str = "loading";
const STATUS_OK: &str = "ok";
const STATUS_STOPPING: &str = "stopping";

/// Which engine a server (or a discovery entry) runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum EngineKind {
    #[default]
    Dense,
    Sparse,
    Multimodal,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Multimodal => "multimodal",
        }
    }
}

/// The engine behind one server. Dense, learned-sparse and multimodal engines
/// are driven through this one enum so a server's shape is decided at start.
#[derive(Clone)]
pub enum Engine {
    Dense(Arc<dyn DenseEngine>),
    Sparse(Arc<dyn SparseEngine>),
    Multimodal(Arc<dyn MultimodalEngine>),
}

impl Engine {
    pub fn dense(engine: Arc<dyn DenseEngine>) -> Self {
        Self::Dense(engine)
    }

    pub fn sparse(engine: Arc<dyn SparseEngine>) -> Self {
        Self::Sparse(engine)
    }

    pub fn multimodal(engine: Arc<dyn MultimodalEngine>) -> Self {
        Self::Multimodal(engine)
    }

    pub fn kind(&self) -> EngineKind {
        match self {
            Self::Dense(_) => EngineKind::Dense,
            Self::Sparse(_) => EngineKind::Sparse,
            Self::Multimodal(_) => EngineKind::Multimodal,
        }
    }

    async fn load(&self) -> Result<EngineInfo> {
        match self {
            Self::Dense(engine) => engine.load().await,
            Self::Sparse(engine) => engine.load().await,
            Self::Multimodal(engine) => engine.load().await,
        }
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        match self {
            Self::Dense(engine) => engine.token_counts(texts).await,
            Self::Sparse(engine) => engine.token_counts(texts).await,
            Self::Multimodal(_) => Err(InlError::Config(
                "token counts are not available for a multimodal engine".to_string(),
            )),
        }
    }

    async fn embed_dense(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Dense(engine) => engine.embed(texts).await,
            _ => Err(InlError::Config(
                "this server has no dense text engine".to_string(),
            )),
        }
    }

    async fn encode_sparse(&self, texts: Vec<String>) -> Result<Vec<SparseVector>> {
        match self {
            Self::Sparse(engine) => engine.encode(texts).await,
            _ => Err(InlError::Config(
                "this server has no sparse engine".to_string(),
            )),
        }
    }

    async fn embed_multimodal(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Multimodal(engine) => engine.embed(inputs).await,
            _ => Err(InlError::Config(
                "this server has no multimodal engine".to_string(),
            )),
        }
    }

    /// Release the weights; the next request reloads.
    pub async fn unload(&self) -> bool {
        match self {
            Self::Dense(engine) => engine.unload().await,
            Self::Sparse(engine) => engine.unload().await,
            Self::Multimodal(engine) => engine.unload().await,
        }
    }

    pub fn is_loaded(&self) -> bool {
        match self {
            Self::Dense(engine) => engine.is_loaded(),
            Self::Sparse(engine) => engine.is_loaded(),
            Self::Multimodal(engine) => engine.is_loaded(),
        }
    }
}

/// Everything that changes which model instance a state directory belongs to.
/// Prefixes and batching knobs are excluded: callers send prefixed text, and
/// batching does not change results beyond left-padding noise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceKey {
    /// Hub repo id or local directory, without a provider prefix.
    pub model: String,
    /// `auto`, `cpu`, `metal` or `cuda`; `None` is `auto`.
    pub device: Option<String>,
    /// `auto`, `f32`, `f16` or `bf16`; `None` is `auto`.
    pub dtype: Option<String>,
    /// `auto`, `eager`, `tiled` or `fused`; `None` is `auto`.
    pub attention: Option<String>,
    /// Context length in tokens; `None` is the engine's default.
    pub max_length: Option<usize>,
    pub kind: EngineKind,
}

impl Default for InstanceKey {
    fn default() -> Self {
        Self {
            model: String::new(),
            device: None,
            dtype: None,
            attention: None,
            max_length: None,
            kind: EngineKind::Dense,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub host: IpAddr,
    pub port: u16,
    /// `None` serves until stopped.
    pub idle_timeout: Option<Duration>,
    pub max_inputs: usize,
    pub max_request_bytes: usize,
    /// Directory for `lock`, `pid`, `host` and `port`.
    pub state_dir: PathBuf,
    /// Identity of the model configuration; see [`instance_key`].
    pub instance: String,
    pub query_prefix: Option<String>,
    pub passage_prefix: Option<String>,
}

/// Build capabilities of this binary, surfaced by `/health`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildInfo {
    pub version: String,
    /// Enabled accel features: `cuda`, `metal`, `accelerate`, `mkl`.
    pub features: Vec<String>,
}

/// What this build can do, computed from compile-time cfg.
pub fn build_info() -> BuildInfo {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda".to_string());
    }
    if cfg!(target_os = "macos") {
        features.push("metal".to_string());
    }
    if cfg!(feature = "accelerate") {
        features.push("accelerate".to_string());
    }
    if cfg!(feature = "mkl") {
        features.push("mkl".to_string());
    }
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        features,
    }
}

fn dense_kind() -> String {
    EngineKind::Dense.as_str().to_string()
}

/// `GET /health` body; also what an adopting process reports.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Health {
    /// `loading`, `ok` or `stopping`.
    pub status: String,
    /// `dense`, `sparse` or `multimodal`.
    #[serde(default = "dense_kind")]
    pub kind: String,
    pub model_id: String,
    pub dimension: usize,
    pub dtype: String,
    pub device: String,
    pub attention: String,
    pub max_length: usize,
    /// Which inputs the engine accepts, as `text`/`image`/`video`/`audio`.
    #[serde(default)]
    pub modalities: Vec<String>,
    pub max_inputs: usize,
    pub instance: String,
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub url: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub requests_total: u64,
    pub loaded: bool,
    pub build: BuildInfo,
}

impl Health {
    /// True when the server is ready to answer embedding requests.
    pub fn is_ready(&self) -> bool {
        self.status == STATUS_OK
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// SIGTERM / Ctrl-C, the caller's stop future, or `/admin/shutdown`.
    Signal,
    /// No embedding request for the idle timeout.
    Idle,
    /// A forward pass exceeded its deadline; its thread cannot be
    /// cancelled, so the process must exit to free the model.
    Wedged,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Signal => "signal",
            Self::Idle => "idle",
            Self::Wedged => "wedged",
        }
    }

    /// Process exit code for this stop: a wedged model exits like an embedder
    /// timeout ([`InlError::Timeout`]).
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Signal | Self::Idle => 0,
            Self::Wedged => 7,
        }
    }
}

/// Both variants are built once per process, so the size gap costs nothing.
#[allow(clippy::large_enum_variant)]
pub enum Started {
    /// This process loaded the model and owns the listener.
    Serving(Server),
    /// Another process already serves this instance.
    Adopted(Health),
}

pub struct Server {
    listener: tokio::net::TcpListener,
    state: Arc<AppState>,
    files: StateFiles,
    idle_timeout: Option<Duration>,
    max_request_bytes: usize,
}

struct AppState {
    engine: Engine,
    info: EngineInfo,
    kind: EngineKind,
    instance: String,
    host: String,
    port: u16,
    url: String,
    max_inputs: usize,
    max_request_bytes: usize,
    query_prefix: Option<String>,
    passage_prefix: Option<String>,
    started: std::time::Instant,
    status: Mutex<&'static str>,
    requests_total: AtomicU64,
    in_flight: AtomicUsize,
    last_activity: Mutex<Instant>,
    wedged: tokio::sync::Notify,
    shutdown: tokio::sync::Notify,
}

/// Lock plus the discovery files; removed when the server stops.
struct StateFiles {
    dir: PathBuf,
    _lock: LockFile,
}

struct LockFile(std::fs::File);

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

impl Drop for StateFiles {
    fn drop(&mut self) {
        self.remove();
    }
}

impl StateFiles {
    fn publish(&self, health: &Health) -> Result<()> {
        write_atomic(&self.dir.join("pid"), &health.pid.to_string())?;
        write_atomic(&self.dir.join("host"), &health.host)?;
        // Last: a reader that sees `port` sees the other files too.
        write_atomic(&self.dir.join("port"), &health.port.to_string())
    }

    fn remove(&self) {
        for name in ["port", "host", "pid"] {
            let _ = std::fs::remove_file(self.dir.join(name));
        }
    }
}

/// Write `content` to `path` atomically (temp file in the same directory,
/// then rename), so a reader never sees a half-written discovery file.
fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "state".to_string());
    let tmp = parent.join(format!(".{name}.{}", std::process::id()));
    std::fs::write(&tmp, format!("{content}\n"))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Accept only loopback addresses: the server has no authentication.
pub fn parse_loopback_host(value: &str) -> Result<IpAddr> {
    let ip = if value.eq_ignore_ascii_case("localhost") {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        value.parse::<IpAddr>().map_err(|_| {
            InlError::Config(format!(
                "serve host '{value}' is not an IP address; use 127.0.0.1 or ::1"
            ))
        })?
    };
    if !ip.is_loopback() {
        return Err(InlError::Config(format!(
            "serve host {ip} is not a loopback address; the server has no authentication, \
             so it only binds 127.0.0.1 or ::1"
        )));
    }
    Ok(ip)
}

/// Identity of a served model: everything that changes the vectors the engine
/// produces. A stable name prefix plus a short digest of the material.
pub fn instance_key(key: &InstanceKey) -> Result<String> {
    let model = key.model.trim();
    if model.is_empty() {
        return Err(InlError::Config("model id is empty".to_string()));
    }
    let auto = |value: &Option<String>| match value.as_deref() {
        None | Some("auto") => "auto".to_string(),
        Some(other) => other.to_string(),
    };
    let material = json!({
        "kind": key.kind.as_str(),
        "model": model,
        "device": auto(&key.device),
        "dtype": auto(&key.dtype),
        "attention": auto(&key.attention),
        "max_length": key.max_length,
    });
    let digest = Sha256::digest(material.to_string());
    let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let name: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let infix = match key.kind {
        EngineKind::Dense => "",
        EngineKind::Sparse => "-sparse",
        EngineKind::Multimodal => "-multimodal",
    };
    Ok(format!("{name}{infix}-{}", &digest[..12]))
}

/// Default state directory for an instance:
/// `~/.cache/im-not-llm/serve/<instance>` (`%LOCALAPPDATA%\im-not-llm\serve`
/// on Windows).
pub fn default_state_dir(instance: &str) -> PathBuf {
    cache_root().join("serve").join(instance)
}

/// A live server found under a serve root.
#[derive(Debug, Clone)]
pub struct FoundInstance {
    pub instance: String,
    pub state_dir: PathBuf,
    pub health: Health,
}

/// Servers under `serve_root` (`<cache_root>/serve`) that serve `model` as
/// `kind` and answer `/health`, whatever device / dtype / attention they were
/// started with. Lifecycle commands use this so they need not repeat every
/// serve flag. A directory is a candidate only by name; what decides is the
/// answering server's own `kind` and `model_id`, so a stale directory left by a
/// crashed server (nothing listens) is skipped.
pub async fn find_instances(
    serve_root: &Path,
    model: &str,
    kind: EngineKind,
) -> Vec<FoundInstance> {
    let Ok(entries) = std::fs::read_dir(serve_root) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let state_dir = entry.path();
        let Some(instance) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let Some(client) = DaemonClient::discover(&state_dir, &instance) else {
            continue;
        };
        let Some(health) = client.health().await else {
            continue;
        };
        if health.kind == kind.as_str() && health.model_id == model {
            found.push(FoundInstance {
                instance,
                state_dir,
                health,
            });
        }
    }
    found.sort_by(|a, b| a.instance.cmp(&b.instance));
    found
}

/// HTTP client for talking to an `inl serve` on this machine.
///
/// The server is loopback plain HTTP, so TLS never runs; building the client
/// without the platform root-certificate store skips what would otherwise be
/// the dominant cost of every CLI start (reading the system keychain took
/// ~90 ms on macOS). Proxies are bypassed because the peer is local.
pub(crate) fn loopback_client(timeout: Duration) -> Option<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(timeout)
        .no_proxy()
        .tls_built_in_root_certs(false)
        .build()
        .ok()
}

/// Read `port`/`host` in `state_dir` and return the server's health when it
/// answers as `instance`.
pub async fn probe(state_dir: &Path, instance: &str) -> Option<Health> {
    DaemonClient::discover(state_dir, instance)?.probe().await
}

fn base_url(addr: SocketAddr) -> String {
    format!("http://{addr}")
}

/// Become the server for `options.instance`, or adopt the one that already
/// is. Loads the model only in the first case.
pub async fn start(engine: Arc<dyn DenseEngine>, options: ServeOptions) -> Result<Started> {
    start_engine(Engine::dense(engine), options).await
}

/// Like [`start`], for a sparse (SPLADE) engine: serves `POST /embed_sparse`.
pub async fn start_sparse(engine: Arc<dyn SparseEngine>, options: ServeOptions) -> Result<Started> {
    start_engine(Engine::sparse(engine), options).await
}

/// Like [`start`], for a multimodal engine: serves `POST /embed_multimodal`.
pub async fn start_multimodal(
    engine: Arc<dyn MultimodalEngine>,
    options: ServeOptions,
) -> Result<Started> {
    start_engine(Engine::multimodal(engine), options).await
}

async fn start_engine(engine: Engine, options: ServeOptions) -> Result<Started> {
    std::fs::create_dir_all(&options.state_dir)?;
    let lock = acquire_or_adopt(&options).await?;
    let lock = match lock {
        Ok(lock) => lock,
        Err(health) => return Ok(Started::Adopted(health)),
    };
    let files = StateFiles {
        dir: options.state_dir.clone(),
        _lock: lock,
    };
    // Whatever a dead owner left behind is stale now.
    files.remove();

    let kind = engine.kind();
    let info = engine.load().await?;
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(options.host, options.port))
        .await
        .map_err(|error| {
            InlError::Other(format!(
                "cannot bind {}:{}: {error}",
                options.host, options.port
            ))
        })?;
    let addr = listener.local_addr()?;
    let state = Arc::new(AppState {
        engine,
        kind,
        info,
        instance: options.instance.clone(),
        host: addr.ip().to_string(),
        port: addr.port(),
        url: base_url(addr),
        max_inputs: options.max_inputs.max(1),
        max_request_bytes: options.max_request_bytes.max(1),
        query_prefix: options.query_prefix,
        passage_prefix: options.passage_prefix,
        started: std::time::Instant::now(),
        status: Mutex::new(STATUS_LOADING),
        requests_total: AtomicU64::new(0),
        in_flight: AtomicUsize::new(0),
        last_activity: Mutex::new(Instant::now()),
        wedged: tokio::sync::Notify::new(),
        shutdown: tokio::sync::Notify::new(),
    });
    files.publish(&state.snapshot())?;
    state.set_status(STATUS_OK);
    Ok(Started::Serving(Server {
        listener,
        state,
        files,
        idle_timeout: options.idle_timeout,
        max_request_bytes: options.max_request_bytes.max(1),
    }))
}

/// `Ok(lock)` when this process should serve, `Err(health)` to adopt.
async fn acquire_or_adopt(options: &ServeOptions) -> Result<std::result::Result<LockFile, Health>> {
    use fs2::FileExt;
    let path = options.state_dir.join("lock");
    let give_up = Instant::now() + ADOPT_WAIT;
    loop {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        if file.try_lock_exclusive().is_ok() {
            return Ok(Ok(LockFile(file)));
        }
        if let Some(health) = probe(&options.state_dir, &options.instance).await {
            return Ok(Err(health));
        }
        if Instant::now() >= give_up {
            return Err(InlError::Other(format!(
                "another inl serve holds {} but did not answer /health within {}s",
                path.display(),
                ADOPT_WAIT.as_secs()
            )));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

impl AppState {
    /// `GET /health` (and the discovery files) as of now.
    fn snapshot(&self) -> Health {
        let status = *self.status.lock().expect("status lock");
        Health {
            status: status.to_string(),
            kind: self.kind.as_str().to_string(),
            model_id: self.info.model_id.clone(),
            dimension: self.info.dimension,
            dtype: self.info.dtype.clone(),
            device: self.info.device.clone(),
            attention: self.info.attention.clone(),
            max_length: self.info.max_length,
            modalities: self
                .info
                .modalities
                .iter()
                .map(|m| m.as_str().to_string())
                .collect(),
            max_inputs: self.max_inputs,
            instance: self.instance.clone(),
            pid: std::process::id(),
            host: self.host.clone(),
            port: self.port,
            url: self.url.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.started.elapsed().as_secs(),
            requests_total: self.requests_total.load(Ordering::SeqCst),
            loaded: self.engine.is_loaded(),
            build: build_info(),
        }
    }

    fn set_status(&self, status: &'static str) {
        *self.status.lock().expect("status lock") = status;
    }

    fn busy(&self) -> Busy<'_> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        self.requests_total.fetch_add(1, Ordering::SeqCst);
        self.touch();
        Busy(self)
    }

    fn touch(&self) {
        *self.last_activity.lock().expect("activity lock") = Instant::now();
    }

    fn prefix(&self, input_type: Option<InputType>) -> Option<&str> {
        match input_type? {
            InputType::Query => self.query_prefix.as_deref(),
            InputType::Passage => self.passage_prefix.as_deref(),
        }
    }

    fn serves_model(&self, requested: &str) -> bool {
        requested.eq_ignore_ascii_case(&self.info.model_id)
    }

    fn has_modality(&self, modality: Modality) -> bool {
        self.info.modalities.contains(&modality)
    }
}

impl Server {
    pub fn health(&self) -> Health {
        self.state.snapshot()
    }

    pub fn url(&self) -> String {
        self.state.url.clone()
    }

    pub fn state_dir(&self) -> &Path {
        &self.files.dir
    }

    /// Serve until `stop` resolves, the idle timeout passes with no
    /// embedding request, `/admin/shutdown` is called, or a forward pass
    /// wedges. Removes the discovery files and releases the lock on return.
    pub async fn run_until<F>(self, stop: F) -> Result<StopReason>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let Server {
            listener,
            state,
            files,
            idle_timeout,
            max_request_bytes,
        } = self;
        let router = router(Arc::clone(&state), max_request_bytes);
        let (reason_tx, reason_rx) = tokio::sync::oneshot::channel();
        let watched = Arc::clone(&state);
        let signal = async move {
            let idle = async {
                match idle_timeout {
                    Some(timeout) => wait_idle(&watched, timeout).await,
                    None => std::future::pending().await,
                }
            };
            let reason = tokio::select! {
                () = stop => StopReason::Signal,
                () = idle => StopReason::Idle,
                () = watched.wedged.notified() => StopReason::Wedged,
                () = watched.shutdown.notified() => StopReason::Signal,
            };
            watched.set_status(STATUS_STOPPING);
            let _ = reason_tx.send(reason);
        };
        let mut serve = Box::pin(
            axum::serve(listener, router)
                .with_graceful_shutdown(signal)
                .into_future(),
        );
        let reason = tokio::select! {
            biased;
            reason = reason_rx => {
                let reason = reason.unwrap_or(StopReason::Signal);
                // Let in-flight requests finish, but never hang on one.
                let _ = tokio::time::timeout(DRAIN_WAIT, serve).await;
                reason
            }
            served = &mut serve => {
                served?;
                StopReason::Signal
            }
        };
        drop(files);
        Ok(reason)
    }
}

/// Resolve once no embedding request has run for `timeout`.
async fn wait_idle(state: &AppState, timeout: Duration) {
    loop {
        let last = *state.last_activity.lock().expect("activity lock");
        tokio::time::sleep_until(last + timeout).await;
        if state.in_flight.load(Ordering::SeqCst) > 0 {
            // A request outlasting the timeout re-arms it when it ends.
            tokio::time::sleep(timeout).await;
            continue;
        }
        let last = *state.last_activity.lock().expect("activity lock");
        if last.elapsed() >= timeout {
            return;
        }
    }
}

/// Marks an embedding request in flight; the idle clock restarts when it ends.
struct Busy<'a>(&'a AppState);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.touch();
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

fn router(state: Arc<AppState>, max_request_bytes: usize) -> Router {
    let routes = Router::new()
        .route("/health", get(health))
        .route("/admin/unload", post(admin_unload))
        .route("/admin/shutdown", post(admin_shutdown));
    let routes = match state.engine.kind() {
        EngineKind::Dense => routes
            .route("/embed", post(tei_embed))
            .route("/v1/embeddings", post(openai_embeddings)),
        EngineKind::Sparse => routes.route("/embed_sparse", post(tei_embed_sparse)),
        EngineKind::Multimodal => routes
            .route("/embed", post(tei_embed))
            .route("/embed_multimodal", post(embed_multimodal))
            .route("/v1/embeddings", post(openai_embeddings)),
    };
    routes
        .layer(DefaultBodyLimit::max(max_request_bytes))
        .with_state(BodyLimited {
            state,
            max_request_bytes,
        })
}

#[derive(Clone)]
struct BodyLimited {
    state: Arc<AppState>,
    max_request_bytes: usize,
}

async fn health(State(app): State<BodyLimited>) -> Json<Health> {
    Json(app.state.snapshot())
}

/// `POST /admin/unload`: release the weights without stopping the server
/// (loopback-only like every route, so only this machine can call it).
async fn admin_unload(State(app): State<BodyLimited>) -> Json<serde_json::Value> {
    let unloaded = app.state.engine.unload().await;
    Json(json!({ "unloaded": unloaded, "loaded": app.state.engine.is_loaded() }))
}

/// `POST /admin/shutdown`: ask the server to drain and exit.
async fn admin_shutdown(State(app): State<BodyLimited>) -> Json<serde_json::Value> {
    app.state.set_status(STATUS_STOPPING);
    app.state.shutdown.notify_one();
    Json(json!({ "status": STATUS_STOPPING }))
}

/// Query/passage extension to the text protocols.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum InputType {
    Query,
    Passage,
}

#[derive(Deserialize)]
struct TeiRequest {
    inputs: serde_json::Value,
    #[serde(default)]
    truncate: Option<bool>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    truncation_direction: Option<String>,
    #[serde(default)]
    prompt_name: Option<String>,
    #[serde(default)]
    input_type: Option<InputType>,
}

#[derive(Deserialize)]
struct OpenAiRequest {
    #[serde(default)]
    model: Option<String>,
    input: serde_json::Value,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    input_type: Option<InputType>,
}

/// An item of the multimodal `POST /embed_multimodal` body.
#[derive(Debug, Deserialize)]
struct MultimodalItem {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    videos: Vec<String>,
    #[serde(default)]
    audios: Vec<String>,
}

#[derive(Deserialize)]
struct MultimodalRequest {
    inputs: Vec<MultimodalItem>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default)]
    input_type: Option<InputType>,
}

/// One `input` item of the multimodal OpenAI protocol.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiItem {
    Text(String),
    Spec(OpenAiSpec),
}

#[derive(Debug, Deserialize)]
struct OpenAiSpec {
    #[serde(rename = "type")]
    kind: String,
    data: String,
}

#[derive(Clone, Copy)]
enum Protocol {
    Tei,
    OpenAi,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    code: Option<&'static str>,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: None,
        }
    }

    fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    fn render(self, protocol: Protocol) -> Response {
        match protocol {
            Protocol::Tei => {
                let error_type = match self.status {
                    StatusCode::SERVICE_UNAVAILABLE => "Unhealthy",
                    status if status.is_server_error() => "Backend",
                    _ => "Validation",
                };
                (
                    self.status,
                    Json(json!({ "error": self.message, "error_type": error_type })),
                )
                    .into_response()
            }
            Protocol::OpenAi => {
                // OpenAI reports every malformed request as 400.
                let status = if self.status == StatusCode::UNPROCESSABLE_ENTITY {
                    StatusCode::BAD_REQUEST
                } else {
                    self.status
                };
                let kind = if status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                (
                    status,
                    Json(json!({
                        "error": {
                            "message": self.message,
                            "type": kind,
                            "param": null,
                            "code": self.code,
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}

fn parse_body<T: serde::de::DeserializeOwned>(
    body: std::result::Result<Bytes, BytesRejection>,
    max_request_bytes: usize,
) -> std::result::Result<T, ApiError> {
    let bytes = body.map_err(|rejection| {
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "request body exceeds the {max_request_bytes}-byte limit (--max-request-bytes)"
                ),
            )
        } else {
            ApiError::new(rejection.status(), rejection.body_text())
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        let status = match error.classify() {
            serde_json::error::Category::Data => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::BAD_REQUEST,
        };
        ApiError::new(status, format!("invalid request body: {error}"))
    })
}

/// A string or an array of strings. Token-id arrays are not supported.
fn input_texts(
    value: serde_json::Value,
    field: &str,
) -> std::result::Result<Vec<String>, ApiError> {
    let invalid = || {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "`{field}` must be a string or an array of strings (token arrays are not supported)"
            ),
        )
    };
    match value {
        serde_json::Value::String(text) => Ok(vec![text]),
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                serde_json::Value::String(text) => Ok(text),
                _ => Err(invalid()),
            })
            .collect(),
        _ => Err(invalid()),
    }
}

/// Qwen3 uses byte-level BPE: an input never has more tokens than UTF-8
/// bytes plus template tokens, so only inputs this close to the context
/// need tokenizing to enforce it.
const TOKEN_SLACK: usize = 8;

struct Embedded {
    vectors: Vec<Vec<f32>>,
    /// Exact prompt tokens, when counted.
    tokens: usize,
}

/// Inputs that passed validation, with the exact prompt tokens when counted.
struct Checked {
    texts: Vec<String>,
    tokens: usize,
}

/// Validate, prefix and check the context. `count_all` tokenizes every input
/// (OpenAI usage); otherwise only inputs that may exceed the context.
async fn checked_inputs_counting(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    count_all: bool,
    protocol: Protocol,
) -> std::result::Result<Checked, ApiError> {
    if texts.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "`inputs` cannot be empty",
        ));
    }
    if texts.len() > state.max_inputs {
        return Err(batch_too_large(texts.len(), state.max_inputs));
    }
    let texts: Vec<String> = match state.prefix(input_type) {
        Some(prefix) => texts.into_iter().map(|t| format!("{prefix}{t}")).collect(),
        None => texts,
    };
    let max_length = state.info.max_length;
    let (indices, candidates): (Vec<usize>, Vec<String>) = texts
        .iter()
        .enumerate()
        .filter(|(_, text)| count_all || text.len() + TOKEN_SLACK > max_length)
        .map(|(index, text)| (index, text.clone()))
        .unzip();
    let counts = if candidates.is_empty() {
        Vec::new()
    } else {
        state
            .engine
            .token_counts(&candidates)
            .await
            .map_err(|error| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
    };
    if !truncate {
        if let Some((index, tokens)) = indices
            .iter()
            .zip(&counts)
            .find(|(_, &tokens)| tokens > max_length)
        {
            let message = match protocol {
                Protocol::Tei => format!(
                    "input {index}: `inputs` must have less than {max_length} tokens. Given: {tokens}"
                ),
                Protocol::OpenAi => format!(
                    "input {index}: this model's maximum context length is {max_length} tokens, \
                     however you requested {tokens} tokens"
                ),
            };
            return Err(ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, message)
                .with_code("context_length_exceeded"));
        }
    }
    let tokens = counts.iter().map(|&t| t.min(max_length)).sum();
    Ok(Checked { texts, tokens })
}

fn batch_too_large(given: usize, max: usize) -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        format!("batch size {given} > maximum allowed batch size {max}"),
    )
}

async fn checked_inputs(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    protocol: Protocol,
) -> std::result::Result<Vec<String>, ApiError> {
    Ok(
        checked_inputs_counting(state, texts, input_type, truncate, false, protocol)
            .await?
            .texts,
    )
}

/// Validate, prefix, check the context, embed (dense servers only).
async fn embed_inputs(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
    truncate: bool,
    count_all: bool,
    protocol: Protocol,
) -> std::result::Result<Embedded, ApiError> {
    let Checked { texts, tokens } =
        checked_inputs_counting(state, texts, input_type, truncate, count_all, protocol).await?;
    let vectors = state
        .engine
        .embed_dense(texts)
        .await
        .map_err(|error| engine_error(state, error))?;
    Ok(Embedded { vectors, tokens })
}

async fn tei_embed(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: TeiRequest = parse_body(body, app.max_request_bytes)?;
        let unsupported =
            |message: &str| Err(ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, message));
        if request.normalize == Some(false) {
            return unsupported(
                "`normalize: false` is not supported: native embeddings are always unit length",
            );
        }
        if request.prompt_name.is_some() {
            return unsupported(
                "`prompt_name` is not supported: send prefixed text or `input_type`",
            );
        }
        if request
            .truncation_direction
            .as_deref()
            .is_some_and(|d| !d.eq_ignore_ascii_case("right"))
        {
            return unsupported("only `truncation_direction: \"right\"` is supported");
        }
        let texts = input_texts(request.inputs, "inputs")?;
        let truncate = request.truncate.unwrap_or(false);
        match state.engine {
            Engine::Dense(_) => {
                let embedded = embed_inputs(
                    state,
                    texts,
                    request.input_type,
                    truncate,
                    false,
                    Protocol::Tei,
                )
                .await?;
                Ok(embedded.vectors)
            }
            Engine::Multimodal(_) => {
                if texts.is_empty() {
                    return Err(ApiError::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "`inputs` cannot be empty",
                    ));
                }
                if texts.len() > state.max_inputs {
                    return Err(batch_too_large(texts.len(), state.max_inputs));
                }
                let inputs = multimodal_inputs_from_texts(state, texts, request.input_type);
                let vectors = state
                    .engine
                    .embed_multimodal(inputs)
                    .await
                    .map_err(|error| engine_error(state, error))?;
                Ok(vectors)
            }
            Engine::Sparse(_) => Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "this server serves sparse vectors; use /embed_sparse",
            )),
        }
    }
    .await;
    match result {
        Ok(vectors) => Json(vectors).into_response(),
        Err(error) => error.render(Protocol::Tei),
    }
}

/// Text-only inputs with the caller's prefix applied, for a multimodal
/// engine's TEI `/embed`.
fn multimodal_inputs_from_texts(
    state: &AppState,
    texts: Vec<String>,
    input_type: Option<InputType>,
) -> Vec<MultimodalInput> {
    texts
        .into_iter()
        .map(|text| {
            let text = match state.prefix(input_type) {
                Some(prefix) => format!("{prefix}{text}"),
                None => text,
            };
            MultimodalInput {
                text: Some(text),
                ..MultimodalInput::default()
            }
        })
        .collect()
}

async fn openai_embeddings(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: OpenAiRequest = parse_body(body, app.max_request_bytes)?;
        if let Some(model) = &request.model {
            if !state.serves_model(model) {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    format!(
                        "the model `{model}` is not served here; this server serves `{}`",
                        state.info.model_id
                    ),
                )
                .with_code("model_not_found"));
            }
        }
        if let Some(dimensions) = request.dimensions {
            if dimensions != state.info.dimension {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "`dimensions` {dimensions} is not supported; this model returns {}",
                        state.info.dimension
                    ),
                ));
            }
        }
        let base64 = match request.encoding_format.as_deref() {
            None | Some("float") => false,
            Some("base64") => true,
            Some(other) => {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("`encoding_format` must be \"float\" or \"base64\", got \"{other}\""),
                ))
            }
        };
        let embedded = match state.engine {
            Engine::Dense(_) => {
                let texts = input_texts(request.input, "input")?;
                embed_inputs(
                    state,
                    texts,
                    request.input_type,
                    false,
                    true,
                    Protocol::OpenAi,
                )
                .await?
            }
            Engine::Multimodal(_) => {
                let inputs = openai_multimodal_inputs(state, request.input, request.input_type)?;
                let vectors = state
                    .engine
                    .embed_multimodal(inputs)
                    .await
                    .map_err(|error| engine_error(state, error))?;
                Embedded { vectors, tokens: 0 }
            }
            Engine::Sparse(_) => {
                return Err(ApiError::new(
                    StatusCode::NOT_FOUND,
                    "this server serves sparse vectors; use /embed_sparse",
                ))
            }
        };
        Ok((embedded, base64))
    }
    .await;
    match result {
        Ok((embedded, base64)) => {
            let data: Vec<serde_json::Value> = embedded
                .vectors
                .into_iter()
                .enumerate()
                .map(|(index, vector)| {
                    let embedding = if base64 {
                        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
                        json!(base64::engine::general_purpose::STANDARD.encode(bytes))
                    } else {
                        json!(vector)
                    };
                    json!({ "object": "embedding", "index": index, "embedding": embedding })
                })
                .collect();
            Json(json!({
                "object": "list",
                "data": data,
                "model": state.info.model_id,
                "usage": { "prompt_tokens": embedded.tokens, "total_tokens": embedded.tokens },
            }))
            .into_response()
        }
        Err(error) => error.render(Protocol::OpenAi),
    }
}

/// The multimodal OpenAI `input`: a string, or an array of strings and
/// `{ "type": "image" | "video" | "audio" | "text", "data": "..." }` items.
fn openai_multimodal_inputs(
    state: &AppState,
    value: serde_json::Value,
    input_type: Option<InputType>,
) -> std::result::Result<Vec<MultimodalInput>, ApiError> {
    let items: Vec<OpenAiItem> = match value {
        serde_json::Value::String(text) => vec![OpenAiItem::Text(text)],
        other => serde_json::from_value(other).map_err(|error| {
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("`input` must be a string or an array of text/media items: {error}"),
            )
        })?,
    };
    if items.is_empty() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "`input` cannot be empty",
        ));
    }
    if items.len() > state.max_inputs {
        return Err(batch_too_large(items.len(), state.max_inputs));
    }
    let mut inputs = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let mut input = MultimodalInput::default();
        match item {
            OpenAiItem::Text(text) => {
                input.text = Some(match state.prefix(input_type) {
                    Some(prefix) => format!("{prefix}{text}"),
                    None => text,
                });
            }
            OpenAiItem::Spec(spec) => {
                let data = decode_media(&spec.data).map_err(|message| {
                    ApiError::new(StatusCode::BAD_REQUEST, format!("input {index}: {message}"))
                })?;
                match spec.kind.as_str() {
                    "image" => push_media(state, &mut input.images, Modality::Image, data, index)?,
                    "video" => push_media(state, &mut input.videos, Modality::Video, data, index)?,
                    "audio" => push_media(state, &mut input.audios, Modality::Audio, data, index)?,
                    "text" => {
                        let text = String::from_utf8(data.bytes).map_err(|_| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                format!("input {index}: text data is not valid UTF-8"),
                            )
                        })?;
                        input.text = Some(match state.prefix(input_type) {
                            Some(prefix) => format!("{prefix}{text}"),
                            None => text,
                        });
                    }
                    other => {
                        return Err(ApiError::new(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            format!(
                                "input {index}: item type `{other}` is not one of text/image/video/audio"
                            ),
                        ))
                    }
                }
            }
        }
        inputs.push(input);
    }
    Ok(inputs)
}

fn push_media(
    state: &AppState,
    slot: &mut Vec<MediaBytes>,
    modality: Modality,
    media: MediaBytes,
    index: usize,
) -> std::result::Result<(), ApiError> {
    if !state.has_modality(modality) {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!(
                "input {index}: this engine does not accept {} inputs",
                modality.as_str()
            ),
        ));
    }
    slot.push(media);
    Ok(())
}

/// Decode one base64 media payload; 400 for anything that is not base64.
fn decode_media(data: &str) -> std::result::Result<MediaBytes, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(data.trim()))
        .map_err(|error| format!("media data is not valid base64: {error}"))?;
    if bytes.is_empty() {
        return Err("media data is empty".to_string());
    }
    Ok(MediaBytes { bytes, mime: None })
}

#[derive(Serialize)]
struct SparseValue {
    index: u32,
    value: f32,
}

/// `POST /embed_sparse`: TEI's learned-sparse protocol.
async fn tei_embed_sparse(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: TeiRequest = parse_body(body, app.max_request_bytes)?;
        if request.prompt_name.is_some() {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "`prompt_name` is not supported: send prefixed text or `input_type`",
            ));
        }
        if request
            .truncation_direction
            .as_deref()
            .is_some_and(|d| !d.eq_ignore_ascii_case("right"))
        {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "only `truncation_direction: \"right\"` is supported",
            ));
        }
        let texts = input_texts(request.inputs, "inputs")?;
        let truncate = request.truncate.unwrap_or(false);
        if !matches!(state.engine, Engine::Sparse(_)) {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "this server serves dense vectors; use /embed",
            ));
        }
        let texts =
            checked_inputs(state, texts, request.input_type, truncate, Protocol::Tei).await?;
        state
            .engine
            .encode_sparse(texts)
            .await
            .map_err(|error| engine_error(state, error))
    }
    .await;
    match result {
        Ok(vectors) => Json(
            vectors
                .iter()
                .map(|vector| {
                    vector
                        .indices()
                        .iter()
                        .zip(vector.values())
                        .map(|(&index, &value)| SparseValue { index, value })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(error) => error.render(Protocol::Tei),
    }
}

/// `POST /embed_multimodal`: `{ "inputs": [ {text, images, videos, audios} ] }`.
async fn embed_multimodal(
    State(app): State<BodyLimited>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let state = &app.state;
    let _busy = state.busy();
    let result = async {
        let request: MultimodalRequest = parse_body(body, app.max_request_bytes)?;
        if request.normalize == Some(false) {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "`normalize: false` is not supported: embeddings are always unit length",
            ));
        }
        if request.inputs.is_empty() {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "`inputs` cannot be empty",
            ));
        }
        if request.inputs.len() > state.max_inputs {
            return Err(batch_too_large(request.inputs.len(), state.max_inputs));
        }
        let prefix = state.prefix(request.input_type).map(str::to_string);
        let mut total_bytes = 0usize;
        let mut inputs = Vec::with_capacity(request.inputs.len());
        for (index, item) in request.inputs.into_iter().enumerate() {
            let mut input = MultimodalInput::default();
            if let Some(text) = item.text {
                input.text = Some(match &prefix {
                    Some(prefix) => format!("{prefix}{text}"),
                    None => text,
                });
            }
            for (kind, slot, list) in [
                ("image", &mut input.images, item.images),
                ("video", &mut input.videos, item.videos),
                ("audio", &mut input.audios, item.audios),
            ] {
                if list.is_empty() {
                    continue;
                }
                // Decode first: an undecodable payload is a request-format
                // error (400) whatever the engine accepts.
                let mut decoded = Vec::with_capacity(list.len());
                for data in list {
                    let media = decode_media(&data).map_err(|message| {
                        ApiError::new(
                            StatusCode::BAD_REQUEST,
                            format!("input {index}: {message}"),
                        )
                    })?;
                    total_bytes += media.bytes.len();
                    decoded.push(media);
                }
                let modality = match kind {
                    "image" => Modality::Image,
                    "video" => Modality::Video,
                    _ => Modality::Audio,
                };
                if !state.has_modality(modality) {
                    return Err(ApiError::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!(
                            "input {index}: this engine does not accept {kind} inputs"
                        ),
                    ));
                }
                slot.extend(decoded);
            }
            if input.text.is_none()
                && input.images.is_empty()
                && input.videos.is_empty()
                && input.audios.is_empty()
            {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("input {index} has no text and no media"),
                ));
            }
            inputs.push(input);
        }
        if total_bytes > state.max_request_bytes {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "decoded media is {total_bytes} bytes, over the {}-byte limit (--max-request-bytes)",
                    state.max_request_bytes
                ),
            ));
        }
        let vectors = state
            .engine
            .embed_multimodal(inputs)
            .await
            .map_err(|error| engine_error(state, error))?;
        Ok(json!({
            "embeddings": vectors,
            "model": state.info.model_id,
            "dimensions": state.info.dimension,
        }))
    }
    .await;
    match result {
        Ok(body) => Json(body).into_response(),
        Err(error) => error.render(Protocol::Tei),
    }
}

/// An engine failure as an HTTP error. Input the engine refused (media it
/// cannot decode) is the caller's error; a deadline means the model's thread
/// is stuck for good, so the server also schedules its own shutdown.
fn engine_error(state: &AppState, error: InlError) -> ApiError {
    match error {
        InlError::InvalidInput(_) => {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, error.to_string())
        }
        InlError::Timeout(_) => {
            state.set_status(STATUS_STOPPING);
            state.wedged.notify_one();
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{error}; the server is shutting down"),
            )
        }
        other => ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    }
}

// ---- programmatic lifecycle ------------------------------------------------

/// An in-process server started by [`Daemon::spawn`]: it owns the listener
/// and the state files and can be stopped or awaited.
pub struct ServerHandle {
    url: String,
    state: Arc<AppState>,
    state_dir: PathBuf,
    stop: Arc<tokio::sync::Notify>,
    join: tokio::task::JoinHandle<Result<StopReason>>,
}

impl ServerHandle {
    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn health(&self) -> Health {
        self.state.snapshot()
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Signal the server and wait for it to drain and release its lock.
    pub async fn stop(self) -> Result<StopReason> {
        self.stop.notify_one();
        self.join.await.map_err(join_error)?
    }

    /// Serve until `stop` resolves, then drain and release the lock.
    pub async fn run_until<F>(self, stop: F) -> Result<StopReason>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let ServerHandle {
            stop: notify, join, ..
        } = self;
        let mut join = join;
        tokio::select! {
            result = &mut join => result.map_err(join_error)?,
            () = stop => {
                notify.notify_one();
                join.await.map_err(join_error)?
            }
        }
    }

    /// Wait for the server to stop on its own (signal, idle, wedge).
    pub async fn wait(self) -> Result<StopReason> {
        self.join.await.map_err(join_error)?
    }
}

fn join_error(error: tokio::task::JoinError) -> InlError {
    InlError::Other(format!("serve task failed: {error}"))
}

fn http_error(error: reqwest::Error) -> InlError {
    InlError::Other(format!("http request failed: {error}"))
}

/// A running server, in this process or another. `Daemon::spawn` returns one
/// either way: a server already serving the instance is adopted instead of
/// loading a second copy of the model.
/// Both variants are built once per process, so the size gap costs nothing.
#[allow(clippy::large_enum_variant)]
pub enum Daemon {
    /// This process loaded the model and owns the listener.
    Serving(ServerHandle),
    /// Another process already serves this instance.
    Adopted { health: Health, state_dir: PathBuf },
}

impl Daemon {
    /// Serve `options.instance` in this process, or adopt the one that already
    /// does. The returned daemon's `url`/`health` are usable immediately.
    pub async fn spawn(engine: Engine, options: ServeOptions) -> Result<Daemon> {
        let state_dir = options.state_dir.clone();
        match start_engine(engine, options).await? {
            Started::Adopted(health) => Ok(Daemon::Adopted { health, state_dir }),
            Started::Serving(server) => {
                let url = server.url();
                let state = Arc::clone(&server.state);
                let stop = Arc::new(tokio::sync::Notify::new());
                let trigger = Arc::clone(&stop);
                let join = tokio::spawn(async move {
                    server
                        .run_until(async move { trigger.notified().await })
                        .await
                });
                Ok(Daemon::Serving(ServerHandle {
                    url,
                    state,
                    state_dir,
                    stop,
                    join,
                }))
            }
        }
    }

    pub fn url(&self) -> String {
        match self {
            Self::Serving(handle) => handle.url().to_string(),
            Self::Adopted { health, .. } => health.url.clone(),
        }
    }

    pub fn health(&self) -> Health {
        match self {
            Self::Serving(handle) => handle.health(),
            Self::Adopted { health, .. } => health.clone(),
        }
    }

    pub fn state_dir(&self) -> &Path {
        match self {
            Self::Serving(handle) => handle.state_dir(),
            Self::Adopted { state_dir, .. } => state_dir,
        }
    }

    /// Stop the server (in-process: signal and drain; adopted: ask the owner
    /// over HTTP) and wait for it to be gone.
    pub async fn stop(self) -> Result<StopReason> {
        match self {
            Self::Serving(handle) => handle.stop().await,
            Self::Adopted { health, state_dir } => {
                if let Some(client) = DaemonClient::discover(&state_dir, &health.instance) {
                    client.stop().await?;
                }
                Ok(StopReason::Signal)
            }
        }
    }

    /// Wait for the server to stop on its own.
    pub async fn wait(self) -> Result<StopReason> {
        match self {
            Self::Serving(handle) => handle.wait().await,
            Self::Adopted { health, state_dir } => loop {
                let client =
                    DaemonClient::discover(&state_dir, &health.instance).ok_or_else(|| {
                        InlError::Other("the served instance disappeared".to_string())
                    })?;
                if client.probe().await.is_none() {
                    return Ok(StopReason::Signal);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
        }
    }
}

/// Out-of-process control of an `inl serve`: discovery files plus HTTP.
pub struct DaemonClient {
    url: String,
    /// When set, only a server answering as this instance counts. Unset means
    /// the state directory itself names the server (see
    /// [`DaemonClient::from_state_dir`]).
    instance: Option<String>,
    client: reqwest::Client,
}

impl DaemonClient {
    /// A client for the server recorded in `state_dir`'s discovery files, or
    /// `None` when no `port` file exists.
    pub fn discover(state_dir: &Path, instance: &str) -> Option<Self> {
        let mut client = Self::from_state_dir(state_dir)?;
        client.instance = Some(instance.to_string());
        Some(client)
    }

    /// A client for whichever server `state_dir`'s discovery files record,
    /// without insisting on an instance key. A state directory already names
    /// one server, and a caller that only knows the directory (a lifecycle
    /// command) cannot rebuild the key from the device, dtype and attention
    /// the server was started with. Use [`DaemonClient::discover`] when the
    /// instance is known and a different server must not be taken for it.
    pub fn from_state_dir(state_dir: &Path) -> Option<Self> {
        let port: u16 = std::fs::read_to_string(state_dir.join("port"))
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let host = std::fs::read_to_string(state_dir.join("host"))
            .map(|h| h.trim().to_string())
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        let ip = parse_loopback_host(&host).ok()?;
        let url = base_url(SocketAddr::new(ip, port));
        Some(Self {
            url,
            instance: None,
            client: loopback_client(Duration::from_secs(2))?,
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// The server's health when it answers as this instance, whatever its
    /// `status` (including `loading` / `stopping`).
    pub async fn health(&self) -> Option<Health> {
        let health: Health = self
            .client
            .get(format!("{}/health", self.url))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;
        match &self.instance {
            Some(wanted) => (&health.instance == wanted).then_some(health),
            None => Some(health),
        }
    }

    /// The server's health when it answers as this instance and is ready.
    pub async fn probe(&self) -> Option<Health> {
        self.health().await.filter(Health::is_ready)
    }

    /// Poll `/health` until the server is ready or `timeout` elapses.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<Health> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(health) = self.probe().await {
                return Ok(health);
            }
            if Instant::now() >= deadline {
                return Err(InlError::Timeout(format!(
                    "{} did not become ready within {}s",
                    self.url,
                    timeout.as_secs()
                )));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// `POST /admin/shutdown`: ask the server to drain and exit.
    pub async fn stop(&self) -> Result<()> {
        let response = self
            .client
            .post(format!("{}/admin/shutdown", self.url))
            .json(&json!({}))
            .send()
            .await
            .map_err(http_error)?;
        if !response.status().is_success() {
            return Err(InlError::Other(format!(
                "{} refused shutdown: {}",
                self.url,
                response.status()
            )));
        }
        Ok(())
    }

    /// `POST /admin/unload`: release the model weights; true when a model was
    /// resident.
    pub async fn unload(&self) -> Result<bool> {
        let response = self
            .client
            .post(format!("{}/admin/unload", self.url))
            .json(&json!({}))
            .send()
            .await
            .map_err(http_error)?;
        if !response.status().is_success() {
            return Err(InlError::Other(format!(
                "{} refused unload: {}",
                self.url,
                response.status()
            )));
        }
        let body: serde_json::Value = response.json().await.map_err(http_error)?;
        Ok(body["unloaded"].as_bool().unwrap_or(false))
    }
}
