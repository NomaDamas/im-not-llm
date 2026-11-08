//! A `#[cfg(test)]`-free fake engine driven by the real server library, for
//! manual end-to-end checks:
//!
//! ```text
//! cargo run --example fake_server            # serve (wait for Ctrl-C)
//! curl http://127.0.0.1:$PORT/health
//! curl -s http://127.0.0.1:$PORT/embed -H 'content-type: application/json' \
//!      -d '{"inputs":["hello"]}'
//! curl -s http://127.0.0.1:$PORT/embed_multimodal -H 'content-type: application/json' \
//!      -d '{"inputs":[{"text":"a photo","images":["AQIDBA=="]}]}'
//! cargo run --example fake_server -- stop   # shut it down via DaemonClient
//! ```
//!
//! The state directory is `$INL_FAKE_STATE_DIR` (default
//! `target/fake-server`), so the `stop`/`health`/`unload` invocations find the
//! running instance's discovery files.

use async_trait::async_trait;
use inl_core::{EngineInfo, Modality, MultimodalEngine, MultimodalInput, Result};
use inl_server::{
    instance_key, Daemon, DaemonClient, Engine, EngineKind, InstanceKey, ServeOptions,
};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

const MODEL: &str = "fake/multimodal-1";

/// Embeds each input as `[text length, media count, 1.0]`.
struct FakeMultimodal;

#[async_trait]
impl MultimodalEngine for FakeMultimodal {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: MODEL.to_string(),
            dimension: 3,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "eager".to_string(),
            max_length: 512,
            modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Video,
                Modality::Audio,
            ],
        })
    }

    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|input| {
                let media = input.images.len() + input.videos.len() + input.audios.len();
                vec![
                    input.text.as_deref().map(str::len).unwrap_or(0) as f32,
                    media as f32,
                    1.0,
                ]
            })
            .collect())
    }

    async fn unload(&self) -> bool {
        false
    }

    fn is_loaded(&self) -> bool {
        true
    }
}

fn state_dir() -> PathBuf {
    std::env::var("INL_FAKE_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/fake-server"))
}

fn instance() -> String {
    instance_key(&InstanceKey {
        model: MODEL.to_string(),
        kind: EngineKind::Multimodal,
        ..InstanceKey::default()
    })
    .expect("fake instance key")
}

fn options() -> ServeOptions {
    ServeOptions {
        host: "127.0.0.1".parse().unwrap(),
        port: 0,
        idle_timeout: None,
        max_inputs: 8,
        max_request_bytes: 8 * 1024 * 1024,
        state_dir: state_dir(),
        instance: instance(),
        query_prefix: None,
        passage_prefix: None,
    }
}

fn client() -> Option<DaemonClient> {
    DaemonClient::discover(&state_dir(), &instance())
}

#[tokio::main]
async fn main() -> ExitCode {
    let command = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "serve".to_string());
    let result = match command.as_str() {
        "serve" => serve().await,
        "stop" => match client() {
            Some(client) => client.stop().await.map(|()| {
                println!("stopped {}", client.url());
            }),
            None => {
                eprintln!("no server recorded in {}", state_dir().display());
                return ExitCode::FAILURE;
            }
        },
        "unload" => match client() {
            Some(client) => client.unload().await.map(|unloaded| {
                println!("unloaded: {unloaded}");
            }),
            None => {
                eprintln!("no server recorded in {}", state_dir().display());
                return ExitCode::FAILURE;
            }
        },
        "health" => match client() {
            Some(client) => match client.health().await {
                Some(health) => {
                    println!("{}", serde_json::to_string_pretty(&health).unwrap());
                    Ok(())
                }
                None => {
                    eprintln!("{} did not answer /health", client.url());
                    return ExitCode::FAILURE;
                }
            },
            None => {
                eprintln!("no server recorded in {}", state_dir().display());
                return ExitCode::FAILURE;
            }
        },
        other => {
            eprintln!("unknown command `{other}`: use serve, stop, unload or health");
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fake_server: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn serve() -> Result<()> {
    let engine = Engine::multimodal(Arc::new(FakeMultimodal));
    match Daemon::spawn(engine, options()).await? {
        Daemon::Adopted { health, .. } => {
            println!("adopted {} on {}", health.model_id, health.url);
        }
        Daemon::Serving(handle) => {
            let health = handle.health();
            println!("serving {} on {}", health.model_id, health.url);
            println!("state dir {}", state_dir().display());
            let reason = handle.run_until(shutdown_signal()).await?;
            println!("stopped: {}", reason.as_str());
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        } else {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
