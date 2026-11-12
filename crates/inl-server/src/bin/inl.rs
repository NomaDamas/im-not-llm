//! `inl`: the im-not-llm daemon and lifecycle CLI.
//!
//! * `inl serve`  — run an engine behind loopback HTTP (TEI / OpenAI / sparse
//!   / multimodal / late-interaction routes), adopting a server already
//!   running for the same instance.
//! * `inl health` — print the served instance's `/health` JSON; exit 0 only
//!   when its status is `ok`.
//! * `inl stop`   — ask a running server to drain and exit.
//! * `inl unload` — release a running server's weights without stopping it.
//! * `inl devices`— list cpu/metal/cuda availability for this build.

#[path = "../engines.rs"]
mod engines;

use clap::{Args, Parser, Subcommand};
use engines::EngineArgs;
use inl_core::{device::DeviceSpec, InlError, Result};
use inl_server::{
    parse_loopback_host, Daemon, DaemonClient, EngineKind, ServeOptions, StopReason,
    DEFAULT_IDLE_TIMEOUT_SECONDS, DEFAULT_MAX_INPUTS, DEFAULT_MAX_REQUEST_BYTES,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

#[derive(Parser)]
#[command(
    name = "inl",
    version,
    about = "On-device inference engines for non-LLM models"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve an engine over loopback HTTP
    Serve(ServeArgs),
    /// Print a running server's health JSON (exit 0 only when status is ok)
    Health(DiscoverArgs),
    /// Ask a running server to drain and exit
    Stop(DiscoverArgs),
    /// Release a running server's weights without stopping it
    Unload(DiscoverArgs),
    /// List cpu/metal/cuda availability for this build
    Devices,
}

#[derive(Args)]
struct ServeArgs {
    /// Loopback address to bind (127.0.0.1, ::1 or localhost).
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// TCP port; 0 picks a free one (written to the `port` file).
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Seconds without an embedding request before the server exits; 0 serves
    /// until stopped.
    #[arg(long, default_value_t = DEFAULT_IDLE_TIMEOUT_SECONDS)]
    idle_timeout: u64,
    /// Most inputs accepted in one request (larger requests get 413).
    #[arg(long, default_value_t = DEFAULT_MAX_INPUTS)]
    max_inputs: usize,
    /// Largest request body in bytes (larger requests get 413).
    #[arg(long, default_value_t = DEFAULT_MAX_REQUEST_BYTES)]
    max_request_bytes: usize,
    /// Directory for the `pid`, `host`, `port` and `lock` files. Defaults to
    /// ~/.cache/im-not-llm/serve/<model instance>.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
    /// Hub repo id or local directory to serve.
    #[arg(long)]
    model: Option<String>,
    /// `auto`, `cpu`, `metal` or `cuda`.
    #[arg(long)]
    device: Option<String>,
    /// `auto`, `f32`, `f16` or `bf16`.
    #[arg(long)]
    dtype: Option<String>,
    /// Serve the learned-sparse (SPLADE) encoder. Optional: without a kind
    /// flag the kind follows `--model` (a ModernBERT checkpoint is sparse, a
    /// Qwen3.5 checkpoint late interaction, anything else dense).
    #[arg(long)]
    sparse: bool,
    /// Serve a multimodal embedder: `POST /embed_multimodal` and media items
    /// on `POST /v1/embeddings`. Never inferred, because the same checkpoint
    /// can also serve dense text.
    #[arg(long)]
    multimodal: bool,
    /// Serve a late-interaction (multi-vector) encoder: `POST /embed` returns
    /// one vector per token. Optional, like `--sparse`.
    #[arg(long)]
    late: bool,
    /// With `--multimodal`: the towers to load, comma-separated from `text`,
    /// `image`, `video` and `audio` (default: every tower the checkpoint has).
    /// A modality left out is not loaded and is refused with 422.
    #[arg(long, value_delimiter = ',', requires = "multimodal")]
    modalities: Option<Vec<String>>,
}

impl ServeArgs {
    fn kind(&self) -> Result<EngineKind> {
        match kind_flag(self.sparse, self.multimodal, self.late)? {
            Some(kind) => Ok(kind),
            None => engines::detect_kind(&self.engine_args()),
        }
    }

    fn engine_args(&self) -> EngineArgs {
        EngineArgs {
            model: self.model.clone(),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
            cache_dir: None,
            modalities: self.modalities.clone(),
        }
    }
}

#[derive(Args)]
struct DiscoverArgs {
    /// Directory holding the `port`/`host` files. Defaults to the instance's
    /// state directory.
    #[arg(long, value_name = "DIR")]
    state_dir: Option<PathBuf>,
    /// Model id the server serves (defaults to the kind's model). Device,
    /// dtype and kind need not be repeated: the server is found by model.
    #[arg(long)]
    model: Option<String>,
    /// `auto`, `cpu`, `metal` or `cuda`.
    #[arg(long)]
    device: Option<String>,
    /// `auto`, `f32`, `f16` or `bf16`.
    #[arg(long)]
    dtype: Option<String>,
    /// Only look for a sparse encoder.
    #[arg(long)]
    sparse: bool,
    /// Only look for a multimodal embedder.
    #[arg(long)]
    multimodal: bool,
    /// Only look for a late-interaction encoder.
    #[arg(long)]
    late: bool,
}

impl DiscoverArgs {
    /// The running server this command talks to. An explicit `--state-dir`
    /// names it directly; otherwise it is found by model (and kind, when a
    /// kind flag narrows it) among the live servers, whatever device / dtype
    /// they were started with. More than one match is an error that lists
    /// them, because stopping the wrong server is worse than asking.
    async fn client(&self) -> Result<Option<DaemonClient>> {
        let kinds = match kind_flag(self.sparse, self.multimodal, self.late)? {
            Some(kind) => vec![kind],
            // Like `serve`: no model and no kind flag is the dense default.
            None if self.model.as_deref().is_none_or(str::is_empty) => vec![EngineKind::Dense],
            None => vec![
                EngineKind::Dense,
                EngineKind::Sparse,
                EngineKind::Multimodal,
                EngineKind::Late,
            ],
        };
        let engine_args = EngineArgs {
            model: self.model.clone(),
            device: self.device.clone(),
            dtype: self.dtype.clone(),
            cache_dir: None,
            modalities: None,
        };
        if let Some(state_dir) = &self.state_dir {
            return Ok(DaemonClient::from_state_dir(state_dir));
        }
        let serve_root = inl_core::device::cache_root().join("serve");
        let mut found = Vec::new();
        for kind in kinds {
            let model = engine_args.model_or_default(kind);
            found.extend(inl_server::find_instances(&serve_root, &model, kind).await);
        }
        match found.len() {
            0 => Ok(None),
            1 => {
                let one = found.remove(0);
                Ok(DaemonClient::discover(&one.state_dir, &one.instance))
            }
            _ => Err(InlError::Config(format!(
                "{} servers match: {}; pass --state-dir to pick one",
                found.len(),
                found
                    .iter()
                    .map(|f| format!(
                        "{} {} ({}, {})",
                        f.health.model_id, f.health.url, f.health.device, f.health.dtype
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

const NO_SERVER: &str = "inl: no running server found (start one with `inl serve`)";

/// The kind a flag names; `None` when no kind flag is given.
fn kind_flag(sparse: bool, multimodal: bool, late: bool) -> Result<Option<EngineKind>> {
    match (sparse, multimodal, late) {
        (false, false, false) => Ok(None),
        (true, false, false) => Ok(Some(EngineKind::Sparse)),
        (false, true, false) => Ok(Some(EngineKind::Multimodal)),
        (false, false, true) => Ok(Some(EngineKind::Late)),
        _ => Err(InlError::Config(
            "`--sparse`, `--multimodal` and `--late` are mutually exclusive".to_string(),
        )),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("inl: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Commands::Serve(args) => serve(args).await,
        Commands::Health(args) => health(args).await,
        Commands::Stop(args) => stop(args).await,
        Commands::Unload(args) => unload(args).await,
        Commands::Devices => {
            devices();
            Ok(ExitCode::SUCCESS)
        }
    }
}

async fn serve(args: ServeArgs) -> Result<ExitCode> {
    let host = parse_loopback_host(&args.host)?;
    let kind = args.kind()?;
    let engine_args = args.engine_args();
    let instance = engine_args.instance_key(kind)?;
    let (query_prefix, passage_prefix) = engines::default_prefixes(kind);
    let options = ServeOptions {
        host,
        port: args.port,
        idle_timeout: (args.idle_timeout > 0).then(|| Duration::from_secs(args.idle_timeout)),
        max_inputs: args.max_inputs.max(1),
        max_request_bytes: args.max_request_bytes.max(1),
        state_dir: args
            .state_dir
            .clone()
            .unwrap_or_else(|| inl_server::default_state_dir(&instance)),
        instance,
        query_prefix,
        passage_prefix,
    };
    let engine = engines::build(kind, &engine_args)?;
    match Daemon::spawn(engine, options).await? {
        Daemon::Adopted { health, .. } => {
            println!("adopted: {} on {}", health.model_id, health.url);
            Ok(ExitCode::SUCCESS)
        }
        Daemon::Serving(handle) => {
            let health = handle.health();
            println!(
                "serving: {} on {} ({} dims, {}, {}, pid {})",
                health.model_id,
                health.url,
                health.dimension,
                health.device,
                health.dtype,
                health.pid,
            );
            let reason = handle.run_until(shutdown_signal()).await?;
            tracing_stop(reason);
            Ok(ExitCode::from(reason.exit_code() as u8))
        }
    }
}

async fn health(args: DiscoverArgs) -> Result<ExitCode> {
    let Some(client) = args.client().await? else {
        eprintln!("{NO_SERVER}");
        return Ok(ExitCode::FAILURE);
    };
    match client.health().await {
        Some(health) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&health).map_err(InlError::from)?
            );
            if health.is_ready() {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::FAILURE)
            }
        }
        None => {
            eprintln!("inl: {} did not answer /health", client.url());
            Ok(ExitCode::FAILURE)
        }
    }
}

async fn stop(args: DiscoverArgs) -> Result<ExitCode> {
    let Some(client) = args.client().await? else {
        eprintln!("{NO_SERVER}");
        return Ok(ExitCode::FAILURE);
    };
    client.stop().await?;
    println!("stopped: {}", client.url());
    Ok(ExitCode::SUCCESS)
}

async fn unload(args: DiscoverArgs) -> Result<ExitCode> {
    let Some(client) = args.client().await? else {
        eprintln!("{NO_SERVER}");
        return Ok(ExitCode::FAILURE);
    };
    let unloaded = client.unload().await?;
    println!(
        "unload: {} model ({})",
        if unloaded {
            "released a"
        } else {
            "nothing to release, no"
        },
        client.url()
    );
    Ok(ExitCode::SUCCESS)
}

fn devices() {
    println!("cpu: available");
    println!(
        "metal: {}",
        availability(cfg!(target_os = "macos"), DeviceSpec::Metal)
    );
    println!(
        "cuda: {}",
        availability(cfg!(feature = "cuda"), DeviceSpec::Cuda)
    );
}

fn availability(compiled: bool, spec: DeviceSpec) -> &'static str {
    if compiled && inl_core::device::resolve_device(spec).is_ok() {
        "available"
    } else {
        "unavailable"
    }
}

fn tracing_stop(reason: StopReason) {
    tracing::info!(reason = reason.as_str(), "inl serve stopped");
}

/// SIGTERM or Ctrl-C.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = term.recv() => {}
                    _ = tokio::signal::ctrl_c() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
