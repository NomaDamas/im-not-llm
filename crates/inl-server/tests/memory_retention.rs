//! A large request must not leave the server holding memory it no longer uses.
//!
//! Tokenizing a multi-megabyte text allocates hundreds of bytes per token and
//! frees all of it when the request ends. macOS's system allocator keeps the
//! freed pages resident and takes more for the next large request, so the
//! process footprint ratchets up with every one (Qwen3-Embedding-0.6B on
//! Metal: four 14 MB texts one after another, 1.8 GB to 6.3 GB, never
//! plateauing). This runs the real `inl` binary on the tiny checked-in model,
//! on the CPU, so it needs neither a download nor a GPU. Run it with
//! `--release`: unoptimized tokenizing makes it take over a minute.
#![cfg(target_os = "macos")]

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A text of about 2 MB: a million tokens for the fixture's tokenizer, which
/// allocates about 700 MiB while it runs.
const WORDS_PER_TEXT: usize = 1_000_000;
const ROUNDS: usize = 8;
/// Requests to let the allocator settle before measuring growth.
const WARMUP: usize = 3;
/// What the footprint may still grow by over the remaining requests. A
/// plateauing allocator moves by tens of MiB; the system allocator added about
/// 135 MiB per request, 650 MiB over these five.
const GROWTH_LIMIT_MIB: u64 = 250;

struct Server {
    child: Child,
    port: u16,
    _state: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start() -> Server {
    let state = tempfile::tempdir().unwrap();
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inl-text/tests/fixtures/tiny_embeddinggemma2"
    );
    let child = Command::new(env!("CARGO_BIN_EXE_inl"))
        .args([
            "serve",
            "--device",
            "cpu",
            "--idle-timeout",
            "0",
            "--max-request-bytes",
            "67108864",
            "--model",
        ])
        .arg(model)
        .arg("--state-dir")
        .arg(state.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = Server {
        child,
        port: 0,
        _state: state,
    };
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        assert!(Instant::now() < deadline, "the server did not come up");
        if let Ok(port) = std::fs::read_to_string(server._state.path().join("port")) {
            if let Ok(port) = port.trim().parse::<u16>() {
                if get(port, "/health").is_some_and(|body| body["status"] == "ok") {
                    server.port = port;
                    return server;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn http(port: u16, request: &[u8]) -> Option<(u16, Value)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .ok()?;
    stream.write_all(request).ok()?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).ok()?;
    let text = String::from_utf8_lossy(&response);
    let status = text.split_whitespace().nth(1)?.parse().ok()?;
    let body = text.split("\r\n\r\n").nth(1)?;
    Some((status, serde_json::from_str(body).unwrap_or(Value::Null)))
}

fn get(port: u16, path: &str) -> Option<Value> {
    let request = format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    http(port, request.as_bytes()).map(|(_, body)| body)
}

fn embed(port: u16, body: &Value) -> u16 {
    let body = body.to_string();
    let request = format!(
        "POST /embed HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    http(port, request.as_bytes()).expect("response").0
}

fn rss_mib(pid: u32) -> u64 {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .unwrap()
        / 1024
}

#[test]
fn repeated_large_requests_do_not_ratchet_the_footprint_up() {
    let server = start();
    let pid = server.child.id();
    let text = "a b c d e f ".repeat(WORDS_PER_TEXT / 6);
    let mut after = Vec::new();
    for _ in 0..ROUNDS {
        let status = embed(server.port, &json!({ "inputs": [text], "truncate": true }));
        assert_eq!(status, 200);
        after.push(rss_mib(pid));
    }
    eprintln!("rss after each of {ROUNDS} large requests (MiB): {after:?}");
    let settled = after[WARMUP - 1];
    let last = *after.last().unwrap();
    assert!(
        last < settled + GROWTH_LIMIT_MIB,
        "the footprint kept growing: {settled} MiB after {WARMUP} requests, {last} MiB after {ROUNDS} \
         (rss per request, MiB: {after:?}); freed memory is not going back"
    );
}
