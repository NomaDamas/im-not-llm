//! Behavior of `inl serve` against fake engines on a real loopback socket:
//! protocol shapes, prefixes, limits, idle exit, adoption, files, the
//! programmatic lifecycle and the multimodal routes.

use super::*;
use inl_core::{DenseEngine, EngineInfo, InlError, Modality, MultimodalEngine, MultimodalInput};
use inl_sparse::{SparseEngine, SparseVector};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

/// Embeds a text as `[len, first byte, 1]`; one token per whitespace word.
#[derive(Default)]
struct FakeEngine {
    seen: StdMutex<Vec<Vec<String>>>,
    delay: Option<Duration>,
    timeout: AtomicBool,
    loaded: AtomicBool,
}

impl FakeEngine {
    fn loaded() -> Self {
        Self {
            loaded: AtomicBool::new(true),
            ..Self::default()
        }
    }
}

#[async_trait::async_trait]
impl DenseEngine for FakeEngine {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: "Fake/Model-1".to_string(),
            dimension: 3,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "eager".to_string(),
            max_length: 4,
            modalities: vec![Modality::Text],
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        Ok(texts.iter().map(|t| t.split_whitespace().count()).collect())
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if self.timeout.load(Ordering::SeqCst) {
            return Err(InlError::Timeout("native embedding batch stuck".into()));
        }
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        self.seen.lock().unwrap().push(texts.clone());
        Ok(texts
            .iter()
            .map(|t| {
                vec![
                    t.len() as f32,
                    f32::from(t.bytes().next().unwrap_or(0)),
                    0.5,
                ]
            })
            .collect())
    }

    async fn unload(&self) -> bool {
        self.loaded.swap(false, Ordering::SeqCst)
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }
}

fn options(dir: &std::path::Path) -> ServeOptions {
    ServeOptions {
        host: "127.0.0.1".parse().unwrap(),
        port: 0,
        idle_timeout: None,
        max_inputs: 8,
        max_request_bytes: 64 * 1024,
        state_dir: dir.to_path_buf(),
        instance: "test-instance".to_string(),
        query_prefix: Some("Q: ".to_string()),
        passage_prefix: Some("P: ".to_string()),
    }
}

struct Harness {
    url: String,
    engine: Arc<FakeEngine>,
    stopped: tokio::task::JoinHandle<Result<StopReason>>,
    stop: Arc<tokio::sync::Notify>,
    _dir: tempfile::TempDir,
}

async fn spawn_with(engine: FakeEngine, tweak: impl FnOnce(&mut ServeOptions)) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options(dir.path());
    tweak(&mut opts);
    let engine = Arc::new(engine);
    let started = start(engine.clone(), opts).await.unwrap();
    let Started::Serving(server) = started else {
        panic!("expected a fresh server");
    };
    let url = server.health().url.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let signal = stop.clone();
    let stopped = tokio::spawn(async move {
        server
            .run_until(async move { signal.notified().await })
            .await
    });
    Harness {
        url,
        engine,
        stopped,
        stop,
        _dir: dir,
    }
}

async fn spawn() -> Harness {
    spawn_with(FakeEngine::default(), |_| {}).await
}

async fn post(url: &str, body: Value) -> (u16, Value) {
    let response = reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status().as_u16();
    (status, response.json().await.unwrap())
}

#[tokio::test]
async fn tei_embed_returns_one_vector_per_input_in_order() {
    let h = spawn().await;
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a", "bb"]})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!([[1.0, 97.0, 0.5], [2.0, 98.0, 0.5]]));
    // A bare string is one input.
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": "c"})).await;
    assert_eq!((status, body), (200, json!([[1.0, 99.0, 0.5]])));
}

#[tokio::test]
async fn prefixes_are_caller_supplied_unless_input_type_is_given() {
    let h = spawn().await;
    post(&format!("{}/embed", h.url), json!({"inputs": ["x"]})).await;
    post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["x"], "input_type": "query"}),
    )
    .await;
    post(
        &format!("{}/v1/embeddings", h.url),
        json!({"input": ["x"], "model": "Fake/Model-1", "input_type": "passage"}),
    )
    .await;
    let seen = h.engine.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            vec!["x".to_string()],
            vec!["Q: x".into()],
            vec!["P: x".into()]
        ]
    );
    let (status, body) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["x"], "input_type": "document"}),
    )
    .await;
    assert_eq!(status, 422, "{body}");
}

#[tokio::test]
async fn openai_embeddings_match_the_openai_shape() {
    let h = spawn().await;
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": ["a b", "c"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "Fake/Model-1");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(body["data"][1]["index"], 1);
    assert_eq!(body["data"][1]["object"], "embedding");
    assert_eq!(body["data"][1]["embedding"], json!([1.0, 99.0, 0.5]));
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 3);
}

#[tokio::test]
async fn openai_base64_encoding_decodes_to_the_float_vector() {
    let h = spawn().await;
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": "ab", "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(body["data"][0]["embedding"].as_str().unwrap())
        .unwrap();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(floats, vec![2.0, 97.0, 0.5]);
}

#[tokio::test]
async fn openai_model_must_name_the_served_model() {
    let h = spawn().await;
    for model in ["Fake/Model-1", "fake/model-1"] {
        let (status, body) = post(
            &format!("{}/v1/embeddings", h.url),
            json!({"model": model, "input": "a"}),
        )
        .await;
        assert_eq!(status, 200, "{model}: {body}");
    }
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "text-embedding-3-small", "input": "a"}),
    )
    .await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": "a", "dimensions": 2}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn inputs_over_the_context_are_rejected_unless_truncation_is_requested() {
    let h = spawn().await;
    let long = "w w w w w"; // 5 tokens > max_length 4
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a", long]})).await;
    assert_eq!(status, 413, "{body}");
    let message = body["error"].as_str().unwrap();
    assert!(message.contains("must have less than"), "{message}");
    assert_eq!(body["error_type"], "Validation");
    assert!(h.engine.seen.lock().unwrap().is_empty());

    let (status, _) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": [long], "truncate": true}),
    )
    .await;
    assert_eq!(status, 200);

    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": [long]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("maximum context length"));
    // The prefix counts against the context: "Q: w w" is 3 words <= 4, "Q: w w w w" is 5.
    let (status, _) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["w w w w"], "input_type": "query"}),
    )
    .await;
    assert_eq!(status, 413);
}

#[tokio::test]
async fn request_size_limits_return_structured_errors() {
    let h = spawn().await;
    let many: Vec<String> = (0..9).map(|i| i.to_string()).collect();
    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": many })).await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("maximum allowed batch size 8"));

    let huge = "x".repeat(70 * 1024);
    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": [huge] })).await;
    assert_eq!(status, 413, "{body}");
    assert_eq!(body["error_type"], "Validation");

    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": ["x".repeat(70 * 1024)]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    let (status, body) = post(&format!("{}/embed", h.url), json!({ "inputs": [] })).await;
    assert_eq!(status, 422, "{body}");
}

#[tokio::test]
async fn malformed_requests_get_errors_and_the_server_keeps_serving() {
    let h = spawn().await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/embed", h.url))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(body["error"].is_string(), "{body}");

    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": [1, 2]})).await;
    assert_eq!(status, 422, "{body}");
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Model-1", "input": [[1, 2, 3]]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"].is_string());
    let (status, body) = post(
        &format!("{}/embed", h.url),
        json!({"inputs": ["a"], "normalize": false}),
    )
    .await;
    assert_eq!(status, 422, "{body}");

    let (status, _) = post(&format!("{}/embed", h.url), json!({"inputs": ["ok"]})).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn health_reports_the_loaded_model_and_build() {
    let h = spawn_with(FakeEngine::loaded(), |_| {}).await;
    let body: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["kind"], "dense");
    assert_eq!(body["model_id"], "Fake/Model-1");
    assert_eq!(body["dimension"], 3);
    assert_eq!(body["dtype"], "f32");
    assert_eq!(body["device"], "cpu");
    assert_eq!(body["instance"], "test-instance");
    assert_eq!(body["pid"], std::process::id());
    assert_eq!(body["modalities"], json!(["text"]));
    assert_eq!(body["loaded"], true);
    assert_eq!(body["requests_total"], 0);
    assert!(body["uptime_seconds"].as_u64().is_some());
    assert_eq!(body["build"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(body["build"]["features"].is_array());

    // An embedding request bumps the counter in `/health`.
    post(&format!("{}/embed", h.url), json!({"inputs": ["a"]})).await;
    let body: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["requests_total"], 1);
}

#[tokio::test]
async fn admin_unload_releases_the_model_and_admin_shutdown_stops_the_server() {
    let engine = FakeEngine::loaded();
    let h = spawn_with(engine, |_| {}).await;
    let url = format!("{}/admin/unload", h.url);
    let (status, body) = post(&url, json!({})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["unloaded"], true);
    assert_eq!(body["loaded"], false);

    let (status, body) = post(&format!("{}/admin/shutdown", h.url), json!({})).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], "stopping");
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("admin shutdown stops the server")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Signal);
}

#[tokio::test]
async fn a_stuck_forward_pass_answers_503_and_stops_the_server() {
    let engine = FakeEngine::default();
    engine.timeout.store(true, Ordering::SeqCst);
    let h = spawn_with(engine, |_| {}).await;
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["a"]})).await;
    assert_eq!(status, 503, "{body}");
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("server stops after a wedged pass")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Wedged);
    assert_eq!(reason.exit_code(), 7);
}

#[tokio::test]
async fn idle_timeout_stops_the_server_but_not_during_a_request() {
    let engine = FakeEngine {
        delay: Some(Duration::from_millis(700)),
        ..FakeEngine::default()
    };
    // Built first: constructing a client can take longer than the timeout.
    let client = reqwest::Client::new();
    let h = spawn_with(engine, |o| {
        o.idle_timeout = Some(Duration::from_millis(300))
    })
    .await;
    let started = Instant::now();
    // The request outlasts the idle timeout; it must still complete.
    let response = client
        .post(format!("{}/embed", h.url))
        .json(&json!({"inputs": ["a"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(started.elapsed() >= Duration::from_millis(700));
    // Health probes do not keep the server alive.
    let probe = h.url.clone();
    tokio::spawn(async move {
        loop {
            let _ = reqwest::get(format!("{probe}/health")).await;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("idle server stops")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Idle);
    assert!(started.elapsed() >= Duration::from_millis(1000));
}

#[tokio::test]
async fn pid_and_port_files_exist_while_serving_and_are_removed_on_stop() {
    let dir = tempfile::tempdir().unwrap();
    let Started::Serving(server) = start(Arc::new(FakeEngine::default()), options(dir.path()))
        .await
        .unwrap()
    else {
        panic!("expected a fresh server");
    };
    let port = server.health().port;
    let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap();
    assert_eq!(read("port").trim(), port.to_string());
    assert_eq!(read("pid").trim(), std::process::id().to_string());
    assert_eq!(read("host").trim(), "127.0.0.1");
    let reason = server.run_until(async {}).await.unwrap();
    assert_eq!(reason, StopReason::Signal);
    assert!(!dir.path().join("port").exists());
    assert!(!dir.path().join("pid").exists());
}

#[tokio::test]
async fn a_second_start_adopts_the_running_server_without_loading() {
    struct NeverLoad;
    #[async_trait::async_trait]
    impl DenseEngine for NeverLoad {
        async fn load(&self) -> Result<EngineInfo> {
            panic!("an adopting process must not load a model");
        }
        async fn token_counts(&self, _: &[String]) -> Result<Vec<usize>> {
            unreachable!()
        }
        async fn embed(&self, _: Vec<String>) -> Result<Vec<Vec<f32>>> {
            unreachable!()
        }
        async fn unload(&self) -> bool {
            unreachable!()
        }
        fn is_loaded(&self) -> bool {
            false
        }
    }
    let h = spawn().await;
    let dir = h._dir.path().to_path_buf();
    let adopted = start(Arc::new(NeverLoad), options(&dir)).await.unwrap();
    let Started::Adopted(health) = adopted else {
        panic!("expected adoption");
    };
    assert_eq!(health.url, h.url);
    assert_eq!(health.model_id, "Fake/Model-1");
    h.stop.notify_one();
    assert_eq!(h.stopped.await.unwrap().unwrap(), StopReason::Signal);
}

#[tokio::test]
async fn stale_files_from_a_dead_server_are_replaced() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("port"), "1\n").unwrap();
    std::fs::write(dir.path().join("pid"), "999999\n").unwrap();
    let started = start(Arc::new(FakeEngine::default()), options(dir.path()))
        .await
        .unwrap();
    let Started::Serving(server) = started else {
        panic!("a dead owner leaves the lock free, so this process serves");
    };
    let port = std::fs::read_to_string(dir.path().join("port")).unwrap();
    assert_eq!(port.trim(), server.health().port.to_string());
}

#[test]
fn only_loopback_hosts_are_accepted() {
    for ok in ["127.0.0.1", "::1", "localhost", "127.0.0.2"] {
        assert!(parse_loopback_host(ok).is_ok(), "{ok}");
    }
    for bad in ["0.0.0.0", "::", "192.168.1.10", "example.com", ""] {
        let error = parse_loopback_host(bad).unwrap_err();
        assert!(matches!(error, InlError::Config(_)), "{bad}: {error}");
    }
}

#[test]
fn instance_key_follows_what_changes_the_vectors() {
    let base = InstanceKey {
        model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
        ..InstanceKey::default()
    };
    let key = instance_key(&base).unwrap();
    assert!(key.starts_with("Qwen-Qwen3-Embedding-0.6B-"), "{key}");

    let mut prefixed = base.clone();
    prefixed.device = Some("auto".into());
    prefixed.dtype = Some("auto".into());
    assert_eq!(instance_key(&prefixed).unwrap(), key);
    for change in [
        |k: &mut InstanceKey| k.dtype = Some("f32".into()),
        |k: &mut InstanceKey| k.device = Some("cpu".into()),
        |k: &mut InstanceKey| k.max_length = Some(512),
        |k: &mut InstanceKey| k.attention = Some("eager".into()),
        |k: &mut InstanceKey| k.model = "Other/Model".into(),
    ] {
        let mut other = base.clone();
        change(&mut other);
        assert_ne!(instance_key(&other).unwrap(), key);
    }
    let mut empty = base;
    empty.model = String::new();
    assert!(instance_key(&empty).is_err());
}

// ---- programmatic lifecycle --------------------------------------------------

#[tokio::test]
async fn daemon_spawn_serves_then_stops_and_removes_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let engine = Arc::new(FakeEngine::loaded());
    let daemon = Daemon::spawn(Engine::dense(engine.clone()), opts)
        .await
        .unwrap();
    assert!(matches!(daemon, Daemon::Serving(_)));
    let url = daemon.url();
    let health = daemon.health();
    assert_eq!(health.status, "ok");
    assert_eq!(health.url, url);
    assert_eq!(health.instance, "test-instance");

    let (status, _) = post(&format!("{url}/embed"), json!({"inputs": ["a"]})).await;
    assert_eq!(status, 200);

    let reason = daemon.stop().await.unwrap();
    assert_eq!(reason, StopReason::Signal);
    assert!(!dir.path().join("port").exists());
    assert!(!dir.path().join("lock").is_file() || dir.path().join("lock").exists());
}

#[tokio::test]
async fn daemon_client_discover_probe_wait_unload_and_stop() {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let daemon = Daemon::spawn(Engine::dense(Arc::new(FakeEngine::loaded())), opts)
        .await
        .unwrap();

    let client = DaemonClient::discover(dir.path(), "test-instance").expect("discovery files");
    let ready = client.wait_ready(Duration::from_secs(5)).await.unwrap();
    assert_eq!(ready.status, "ok");
    assert!(client.probe().await.is_some());
    // The wrong instance is not adopted.
    assert!(DaemonClient::discover(dir.path(), "other")
        .unwrap()
        .probe()
        .await
        .is_none());

    assert!(client.unload().await.unwrap());
    assert!(!client.probe().await.unwrap().loaded);

    client.stop().await.unwrap();
    assert_eq!(daemon.wait().await.unwrap(), StopReason::Signal);
    assert!(probe(dir.path(), "test-instance").await.is_none());
}

#[tokio::test]
async fn daemon_spawn_adopts_an_already_serving_instance() {
    struct NeverLoad;
    #[async_trait::async_trait]
    impl DenseEngine for NeverLoad {
        async fn load(&self) -> Result<EngineInfo> {
            panic!("adoption must not load");
        }
        async fn token_counts(&self, _: &[String]) -> Result<Vec<usize>> {
            unreachable!()
        }
        async fn embed(&self, _: Vec<String>) -> Result<Vec<Vec<f32>>> {
            unreachable!()
        }
        async fn unload(&self) -> bool {
            false
        }
        fn is_loaded(&self) -> bool {
            false
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let serving = Daemon::spawn(Engine::dense(Arc::new(FakeEngine::default())), opts.clone())
        .await
        .unwrap();
    let adopted = Daemon::spawn(Engine::dense(Arc::new(NeverLoad)), opts)
        .await
        .unwrap();
    assert!(matches!(adopted, Daemon::Adopted { .. }));
    assert_eq!(adopted.url(), serving.url());
    assert_eq!(adopted.health().model_id, "Fake/Model-1");

    // Stopping the adopted handle shuts the owner down over HTTP.
    adopted.stop().await.unwrap();
    assert_eq!(serving.wait().await.unwrap(), StopReason::Signal);
}

// ---- sparse (SPLADE) engine: POST /embed_sparse ------------------------------

/// Encodes a text as one term per distinct byte: `{byte: 1 + position-in-text-order}`.
#[derive(Default)]
struct FakeSparse {
    seen: StdMutex<Vec<Vec<String>>>,
    timeout: AtomicBool,
    loaded: AtomicBool,
}

#[async_trait::async_trait]
impl SparseEngine for FakeSparse {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: "Fake/Sparse-1".to_string(),
            dimension: 256,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "tiled".to_string(),
            max_length: 4,
            modalities: vec![Modality::Text],
        })
    }

    async fn token_counts(&self, texts: &[String]) -> Result<Vec<usize>> {
        Ok(texts.iter().map(|t| t.split_whitespace().count()).collect())
    }

    async fn encode(&self, texts: Vec<String>) -> Result<Vec<SparseVector>> {
        if self.timeout.load(Ordering::SeqCst) {
            return Err(InlError::Timeout("native sparse batch stuck".into()));
        }
        self.seen.lock().unwrap().push(texts.clone());
        Ok(texts
            .iter()
            .map(|t| {
                SparseVector::from_pairs(
                    t.bytes()
                        .enumerate()
                        .map(|(i, b)| (u32::from(b), 1.0 + i as f32)),
                )
            })
            .collect())
    }

    async fn unload(&self) -> bool {
        self.loaded.swap(false, Ordering::SeqCst)
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }
}

struct SparseHarness {
    url: String,
    engine: Arc<FakeSparse>,
    stop: Arc<tokio::sync::Notify>,
    stopped: tokio::task::JoinHandle<Result<StopReason>>,
    _dir: tempfile::TempDir,
}

async fn spawn_sparse(engine: FakeSparse) -> SparseHarness {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let engine = Arc::new(engine);
    let Started::Serving(server) = start_sparse(engine.clone(), opts).await.unwrap() else {
        panic!("expected a fresh server");
    };
    let url = server.health().url.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let signal = stop.clone();
    let stopped = tokio::spawn(async move {
        server
            .run_until(async move { signal.notified().await })
            .await
    });
    SparseHarness {
        url,
        engine,
        stop,
        stopped,
        _dir: dir,
    }
}

#[tokio::test]
async fn embed_sparse_returns_tei_index_value_rows_in_order() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let (status, body) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["ab", "b"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    // 'a' = 97 weight 1, 'b' = 98 weight 2 ; second text: 'b' weight 1.
    assert_eq!(
        body,
        json!([
            [{"index": 97, "value": 1.0}, {"index": 98, "value": 2.0}],
            [{"index": 98, "value": 1.0}],
        ])
    );
    let (status, body) = post(&format!("{}/embed_sparse", h.url), json!({"inputs": "a"})).await;
    assert_eq!(
        (status, body),
        (200, json!([[{"index": 97, "value": 1.0}]]))
    );
}

#[tokio::test]
async fn embed_sparse_applies_prefixes_only_with_input_type_and_enforces_the_context() {
    let h = spawn_sparse(FakeSparse::default()).await;
    post(&format!("{}/embed_sparse", h.url), json!({"inputs": ["x"]})).await;
    post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["x"], "input_type": "query"}),
    )
    .await;
    post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": ["x"], "input_type": "passage"}),
    )
    .await;
    assert_eq!(
        h.engine.seen.lock().unwrap().clone(),
        vec![
            vec!["x".to_string()],
            vec!["Q: x".to_string()],
            vec!["P: x".to_string()]
        ]
    );

    let long = "w w w w w"; // 5 tokens > max_length 4
    let (status, body) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": [long]}),
    )
    .await;
    assert_eq!(status, 413, "{body}");
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("must have less than"));
    let (status, _) = post(
        &format!("{}/embed_sparse", h.url),
        json!({"inputs": [long], "truncate": true}),
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn embed_sparse_rejects_bad_requests_and_keeps_serving() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let url = format!("{}/embed_sparse", h.url);
    let many: Vec<String> = (0..9).map(|i| i.to_string()).collect();
    assert_eq!(post(&url, json!({"inputs": many})).await.0, 413);
    assert_eq!(post(&url, json!({"inputs": []})).await.0, 422);
    assert_eq!(post(&url, json!({"inputs": [1, 2]})).await.0, 422);
    assert_eq!(
        post(&url, json!({"inputs": ["a"], "prompt_name": "query"}))
            .await
            .0,
        422,
        "prompt_name is unsupported: send prefixed text or input_type"
    );
    assert_eq!(post(&url, json!({"inputs": ["ok"]})).await.0, 200);
}

#[tokio::test]
async fn a_sparse_server_does_not_serve_dense_routes_and_reports_its_kind() {
    let h = spawn_sparse(FakeSparse::default()).await;
    let client = reqwest::Client::new();
    let health: Value = client
        .get(format!("{}/health", h.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["kind"], "sparse");
    assert_eq!(health["model_id"], "Fake/Sparse-1");
    for route in ["/embed", "/v1/embeddings", "/embed_multimodal"] {
        let response = client
            .post(format!("{}{route}", h.url))
            .json(&json!({"inputs": ["a"], "input": ["a"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 404, "{route}");
    }
}

#[tokio::test]
async fn a_dense_server_reports_kind_dense_and_has_no_embed_sparse() {
    let h = spawn().await;
    let health: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["kind"], "dense");
    // axum's default 404 has an empty body, so check the status directly.
    for route in ["/embed_sparse", "/embed_multimodal"] {
        let response = reqwest::Client::new()
            .post(format!("{}{route}", h.url))
            .json(&json!({"inputs": ["a"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 404, "{route}");
    }
}

#[tokio::test]
async fn a_wedged_sparse_pass_shuts_the_server_down_with_the_timeout_exit_code() {
    let engine = FakeSparse::default();
    engine.timeout.store(true, Ordering::SeqCst);
    let h = spawn_sparse(engine).await;
    let (status, body) = post(&format!("{}/embed_sparse", h.url), json!({"inputs": ["a"]})).await;
    assert_eq!(status, 503, "{body}");
    let reason = tokio::time::timeout(Duration::from_secs(5), h.stopped)
        .await
        .expect("server stops")
        .unwrap()
        .unwrap();
    assert_eq!(reason, StopReason::Wedged);
    drop(h.stop);
}

#[test]
fn sparse_instance_key_differs_from_a_dense_one_and_tracks_vector_inputs() {
    let sparse = InstanceKey {
        model: "nlpai-lab/KURE-Sparse-nano".to_string(),
        kind: EngineKind::Sparse,
        ..InstanceKey::default()
    };
    let key = instance_key(&sparse).unwrap();
    assert!(
        key.starts_with("nlpai-lab-KURE-Sparse-nano-sparse-"),
        "{key}"
    );
    let dense = InstanceKey {
        model: "nlpai-lab/KURE-Sparse-nano".to_string(),
        ..InstanceKey::default()
    };
    assert_ne!(key, instance_key(&dense).unwrap());

    let mut other = sparse.clone();
    other.max_length = Some(512);
    assert_ne!(key, instance_key(&other).unwrap());
    // Batching knobs and prefixes do not change the vectors.
    let mut multimodal = sparse.clone();
    multimodal.kind = EngineKind::Multimodal;
    assert!(instance_key(&multimodal).unwrap().contains("-multimodal-"));
}

// ---- multimodal engine: POST /embed_multimodal -------------------------------

/// Embeds each input as `[text len, image count, 1]`; records what it saw.
#[derive(Default)]
struct FakeMultimodal {
    seen: StdMutex<Vec<Vec<MultimodalInput>>>,
    modalities: Vec<Modality>,
    timeout: AtomicBool,
}

impl FakeMultimodal {
    fn text_and_image() -> Self {
        Self {
            modalities: vec![Modality::Text, Modality::Image],
            ..Self::default()
        }
    }

    fn text_only() -> Self {
        Self {
            modalities: vec![Modality::Text],
            ..Self::default()
        }
    }
}

#[async_trait::async_trait]
impl MultimodalEngine for FakeMultimodal {
    async fn load(&self) -> Result<EngineInfo> {
        Ok(EngineInfo {
            model_id: "Fake/Multimodal-1".to_string(),
            dimension: 3,
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            attention: "eager".to_string(),
            max_length: 512,
            modalities: self.modalities.clone(),
        })
    }

    async fn embed(&self, inputs: Vec<MultimodalInput>) -> Result<Vec<Vec<f32>>> {
        if self.timeout.load(Ordering::SeqCst) {
            return Err(InlError::Timeout("native multimodal batch stuck".into()));
        }
        self.seen.lock().unwrap().push(inputs.clone());
        Ok(inputs
            .iter()
            .map(|input| {
                vec![
                    input.text.as_deref().map(str::len).unwrap_or(0) as f32,
                    input.images.len() as f32,
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

struct MultimodalHarness {
    url: String,
    engine: Arc<FakeMultimodal>,
    stop: Arc<tokio::sync::Notify>,
    stopped: tokio::task::JoinHandle<Result<StopReason>>,
    _dir: tempfile::TempDir,
}

async fn spawn_multimodal(engine: FakeMultimodal) -> MultimodalHarness {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let engine = Arc::new(engine);
    let Started::Serving(server) = start_multimodal(engine.clone(), opts).await.unwrap() else {
        panic!("expected a fresh server");
    };
    let url = server.health().url.clone();
    let stop = Arc::new(tokio::sync::Notify::new());
    let signal = stop.clone();
    let stopped = tokio::spawn(async move {
        server
            .run_until(async move { signal.notified().await })
            .await
    });
    MultimodalHarness {
        url,
        engine,
        stop,
        stopped,
        _dir: dir,
    }
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

impl MultimodalHarness {
    /// Signal and await the server so the test leaves nothing running.
    async fn shutdown(self) {
        self.stop.notify_one();
        let _ = self.stopped.await;
    }
}

#[tokio::test]
async fn embed_multimodal_returns_embeddings_model_and_dimensions() {
    let h = spawn_multimodal(FakeMultimodal::text_and_image()).await;
    let image = b64(&[1, 2, 3, 4]);
    let (status, body) = post(
        &format!("{}/embed_multimodal", h.url),
        json!({"inputs": [
            {"text": "hello"},
            {"text": "a photo <|image|>", "images": [image]},
        ]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "Fake/Multimodal-1");
    assert_eq!(body["dimensions"], 3);
    assert_eq!(
        body["embeddings"],
        json!([[5.0, 0.0, 1.0], [17.0, 1.0, 1.0]])
    );
    let seen = h.engine.seen.lock().unwrap().clone();
    assert_eq!(seen[0][1].images.len(), 1);
    assert_eq!(seen[0][1].images[0].bytes, vec![1, 2, 3, 4]);
    // The multimodal kind is reported by `/health`.
    let health: Value = reqwest::get(format!("{}/health", h.url))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["kind"], "multimodal");
    assert_eq!(health["modalities"], json!(["text", "image"]));
    h.shutdown().await;
}

#[tokio::test]
async fn embed_multimodal_validates_base64_modalities_and_limits() {
    let h = spawn_multimodal(FakeMultimodal::text_only()).await;
    let url = format!("{}/embed_multimodal", h.url);

    // Undecodable base64 is a 400.
    let (status, body) = post(&url, json!({"inputs": [{"images": ["not base64!!"]}]})).await;
    assert_eq!(status, 400, "{body}");

    // A modality the engine does not list is a 422.
    let (status, body) = post(&url, json!({"inputs": [{"images": [b64(&[1, 2, 3])]}]})).await;
    assert_eq!(status, 422, "{body}");
    assert!(body["error"].as_str().unwrap().contains("image"));

    // An input with neither text nor media is a 422.
    let (status, _) = post(&url, json!({"inputs": [{}]})).await;
    assert_eq!(status, 422);

    // Too many inputs is a 413.
    let many: Vec<Value> = (0..9).map(|_| json!({"text": "x"})).collect();
    assert_eq!(post(&url, json!({"inputs": many})).await.0, 413);
    // Empty is a 422.
    assert_eq!(post(&url, json!({"inputs": []})).await.0, 422);

    // A text input still works and reports the prefix is applied.
    let url_q = format!("{}/embed_multimodal", h.url);
    let (status, _) = post(
        &url_q,
        json!({"inputs": [{"text": "hi"}], "input_type": "query"}),
    )
    .await;
    assert_eq!(status, 200);

    // The TEI `/embed` route on a multimodal server embeds text and enforces
    // the same limits.
    assert_eq!(
        post(&format!("{}/embed", h.url), json!({"inputs": []}))
            .await
            .0,
        422
    );
    let (status, body) = post(&format!("{}/embed", h.url), json!({"inputs": ["x"]})).await;
    assert_eq!((status, body), (200, json!([[1.0, 0.0, 1.0]])));
    h.shutdown().await;
}

#[tokio::test]
async fn openai_embeddings_accept_media_items_on_a_multimodal_server() {
    let h = spawn_multimodal(FakeMultimodal::text_and_image()).await;
    let image = b64(&[9, 9]);
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Multimodal-1", "input": [
            "plain",
            {"type": "image", "data": image},
        ]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"][0]["embedding"], json!([5.0, 0.0, 1.0]));
    assert_eq!(body["data"][1]["embedding"], json!([0.0, 1.0, 1.0]));

    // An item type the engine does not list is a 400 in the OpenAI shape.
    let (status, body) = post(
        &format!("{}/v1/embeddings", h.url),
        json!({"model": "Fake/Multimodal-1", "input": [{"type": "video", "data": b64(&[1])}]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("video"));
    h.shutdown().await;
}

// ---- discovery without knowing the serve flags --------------------------------

/// A server started with `--device metal --dtype f32` must be found by a
/// client that only knows the model and kind: the lifecycle commands cannot be
/// required to repeat every serve flag.
#[tokio::test]
async fn find_instances_locates_a_server_by_model_and_kind_alone() {
    let root = tempfile::tempdir().unwrap();
    let key = InstanceKey {
        model: "Fake/Multimodal-1".into(),
        device: Some("metal".into()),
        dtype: Some("f32".into()),
        kind: EngineKind::Multimodal,
        ..InstanceKey::default()
    };
    let instance = instance_key(&key).unwrap();
    let dir = root.path().join(&instance);
    let mut opts = options(&dir);
    opts.instance = instance.clone();
    let daemon = Daemon::spawn(
        Engine::multimodal(Arc::new(FakeMultimodal::text_and_image())),
        opts,
    )
    .await
    .unwrap();

    // A different kind and a different model must not match.
    let found = find_instances(root.path(), "Fake/Multimodal-1", EngineKind::Multimodal).await;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].instance, instance);
    assert!(found[0].health.is_ready());
    assert!(
        find_instances(root.path(), "Fake/Multimodal-1", EngineKind::Dense)
            .await
            .is_empty()
    );
    assert!(
        find_instances(root.path(), "Fake/Other", EngineKind::Multimodal)
            .await
            .is_empty()
    );

    daemon.stop().await.unwrap();
    assert!(
        find_instances(root.path(), "Fake/Multimodal-1", EngineKind::Multimodal)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn find_instances_ignores_stale_state_dirs_whose_server_is_gone() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("org-model-aaaaaaaaaaaa");
    std::fs::create_dir_all(&dir).unwrap();
    // A crashed server left its files behind; nothing listens on this port.
    std::fs::write(dir.join("port"), "1").unwrap();
    std::fs::write(dir.join("host"), "127.0.0.1").unwrap();
    assert!(find_instances(root.path(), "org/model", EngineKind::Dense)
        .await
        .is_empty());
}

/// A state directory already names one server, so a client built from it alone
/// must reach that server whatever instance key it was started under: the
/// lifecycle commands do not know (and must not have to repeat) its device,
/// dtype or attention.
#[tokio::test]
async fn client_from_a_state_dir_alone_reaches_the_server_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let opts = options(dir.path());
    let daemon = Daemon::spawn(Engine::dense(Arc::new(FakeEngine::loaded())), opts)
        .await
        .unwrap();

    let client = DaemonClient::from_state_dir(dir.path()).expect("discovery files");
    let health = client.health().await.expect("the server it names");
    assert_eq!(health.instance, "test-instance");
    assert!(client.probe().await.is_some());
    assert!(client.unload().await.unwrap());
    client.stop().await.unwrap();
    assert_eq!(daemon.wait().await.unwrap(), StopReason::Signal);
    assert!(DaemonClient::from_state_dir(dir.path()).is_none());
}
