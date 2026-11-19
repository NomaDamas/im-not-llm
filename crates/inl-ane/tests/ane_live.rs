//! Live checks of the Apple Neural Engine embedder against the fp32
//! Sentence-Transformers references of `inl-text`
//! (`inl-text/tests/fixtures/*/golden_real.json`).
//!
//! Each test downloads its pinned package and compiles it for the Neural
//! Engine on first use (minutes); run with
//! `cargo test --release -p inl-ane --test ane_live -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use inl_ane::{AneEmbedder, AneOptions, AneSparseEncoder};
use inl_core::DenseEngine;
use inl_sparse::{SparseEngine, SparseVector};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    texts: Vec<String>,
    emb: Vec<Vec<f32>>,
    ids: Vec<Vec<u32>>,
}

fn golden(fixture: &str) -> Golden {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inl-text/tests/fixtures")
        .join(fixture)
        .join("golden_real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// Load `id`, then embed the fixture batched and one text at a time: every
/// vector within `min_cosine` of the reference, a packed text the same as
/// alone, and the same again after an unload.
async fn matches_the_reference(id: &str, fixture: &str, dimension: usize, min_cosine: f32) {
    let golden = golden(fixture);
    let model = AneEmbedder::new(AneOptions {
        id: id.into(),
        ..AneOptions::default()
    })
    .unwrap();
    let started = std::time::Instant::now();
    let info = model.load().await.unwrap();
    eprintln!(
        "{id}: load + self-test {:.1}s",
        started.elapsed().as_secs_f32()
    );
    assert_eq!(info.dimension, dimension);
    assert_eq!(info.device, "ane");
    assert_eq!(info.max_length, 2048);

    let counts = model.token_counts(&golden.texts).await.unwrap();
    for (row, ids) in golden.ids.iter().enumerate() {
        if ids.len() < info.max_length {
            assert_eq!(counts[row], ids.len(), "row {row} token count");
        }
    }

    let started = std::time::Instant::now();
    let batched = model.embed(golden.texts.clone()).await.unwrap();
    eprintln!("batch of {}: {:?}", golden.texts.len(), started.elapsed());
    let mut worst = f32::INFINITY;
    for (row, expected) in golden.emb.iter().enumerate() {
        let norm: f32 = batched[row].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "row {row}: norm {norm}");
        let similarity = cosine(&batched[row], expected);
        let alone = model.embed(vec![golden.texts[row].clone()]).await.unwrap();
        let packed_vs_alone = cosine(&alone[0], &batched[row]);
        eprintln!(
            "row {row}: {} tokens, cosine {similarity:.6}, packed vs alone {packed_vs_alone:.6}",
            golden.ids[row].len()
        );
        assert!(
            packed_vs_alone > 0.9999,
            "row {row}: packed vs alone cosine {packed_vs_alone}"
        );
        worst = worst.min(similarity);
    }
    assert!(worst > min_cosine, "worst cosine to the reference {worst}");

    assert!(model.unload().await);
    assert!(!model.is_loaded());
    let again = model.embed(vec![golden.texts[0].clone()]).await.unwrap();
    assert!(cosine(&again[0], &golden.emb[0]) > min_cosine);
}

/// The fp16 graphs still meet the 0.9999 the candle engine's f32 test holds
/// (measured: at least 0.99996 on every row).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: downloads and compiles the EmbeddingGemma 2 ANE package; run with --ignored"]
async fn embeddinggemma2_matches_the_reference() {
    matches_the_reference("google/embeddinggemma-2", "embeddinggemma2", 768, 0.9999).await;
}

/// Qwen3's 28 layers on the Neural Engine's fp16 matmuls stay below 0.9999:
/// the worst golden row measures 0.99965 on an M4 Max (Core ML's own CPU fp16
/// run of the same graph: 0.9996; candle Metal f16, which accumulates in f32:
/// 0.99999). Retrieval is unchanged (BEIR SciFact nDCG@10 0.7016 vs 0.7011),
/// and packing must still not move a vector (0.9999 above).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: downloads and compiles the Qwen3-Embedding ANE package; run with --ignored"]
async fn qwen3_embedding_matches_the_reference() {
    matches_the_reference("Qwen/Qwen3-Embedding-0.6B", "qwen3embedding", 1024, 0.9995).await;
}

#[derive(Deserialize)]
struct SparseGolden {
    query: String,
    docs: Vec<String>,
    q: SparseRaw,
    d: Vec<SparseRaw>,
}

#[derive(Deserialize)]
struct SparseRaw {
    indices: Vec<u32>,
    values: Vec<f32>,
}

impl SparseRaw {
    fn vector(&self) -> SparseVector {
        SparseVector::new(self.indices.clone(), self.values.clone()).unwrap()
    }
}

fn sparse_cosine(a: &SparseVector, b: &SparseVector) -> f32 {
    a.dot(b) / (a.dot(a).sqrt() * b.dot(b).sqrt())
}

/// KURE-Sparse-nano against the fp32 Sentence-Transformers SparseEncoder
/// (`inl-sparse/tests/fixtures/kure_sparse_nano`): every vector close to the
/// reference, the model-card ranking kept, a packed text the same as alone.
/// The fp32 PyTorch port of the package graph reproduces the reference at
/// cosine 1.000000; the Neural Engine's fp16 run measures 0.99883-0.99904
/// (the same from coremltools and from this crate, so the host pooling adds
/// nothing). `INL_ANE_KURE_PACKAGE` points at a local package instead of the
/// pinned one.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: downloads and compiles the KURE-Sparse-nano ANE package; run with --ignored"]
async fn kure_sparse_nano_matches_the_reference() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inl-sparse/tests/fixtures/kure_sparse_nano/st_f32_golden.json");
    let golden: SparseGolden = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let id = std::env::var("INL_ANE_KURE_PACKAGE")
        .unwrap_or_else(|_| "nlpai-lab/KURE-Sparse-nano".into());
    let encoder = AneSparseEncoder::new(AneOptions {
        id,
        ..AneOptions::default()
    })
    .unwrap();
    let started = std::time::Instant::now();
    let info = encoder.load().await.unwrap();
    eprintln!("load + self-test {:.1}s", started.elapsed().as_secs_f32());
    assert_eq!(info.dimension, 36282);
    assert_eq!(info.device, "ane");

    let mut texts = vec![format!("[Q] {}", golden.query)];
    texts.extend(golden.docs.iter().map(|doc| format!("[D] {doc}")));
    let batched = encoder.encode(texts.clone()).await.unwrap();
    let expected: Vec<SparseVector> = std::iter::once(&golden.q)
        .chain(&golden.d)
        .map(SparseRaw::vector)
        .collect();
    let (mut worst, mut worst_packing) = (f32::INFINITY, f32::INFINITY);
    for (row, (ours, theirs)) in batched.iter().zip(&expected).enumerate() {
        let similarity = sparse_cosine(ours, theirs);
        let alone = encoder.encode(vec![texts[row].clone()]).await.unwrap();
        let packed_vs_alone = sparse_cosine(&alone[0], ours);
        eprintln!(
            "row {row}: nnz {} (ref {}), cosine {similarity:.6}, packed vs alone {packed_vs_alone:.6}",
            ours.len(),
            theirs.len()
        );
        worst = worst.min(similarity);
        worst_packing = worst_packing.min(packed_vs_alone);
    }
    assert!(worst > 0.998, "worst cosine to the reference {worst}");
    assert!(worst_packing > 0.9999, "packed vs alone {worst_packing}");
    let scores: Vec<f32> = batched[1..].iter().map(|d| batched[0].dot(d)).collect();
    let reference: Vec<f32> = expected[1..].iter().map(|d| expected[0].dot(d)).collect();
    eprintln!("scores {scores:?}\nreference {reference:?}");
    assert!(scores[0] > scores[1] && scores[1] > scores[2], "{scores:?}");

    assert!(encoder.unload().await);
    assert!(!encoder.is_loaded());
}

/// Functions load on first use: a cold encoder answers a short query without
/// loading the long functions, and a long text that arrives later loads (and
/// self-tests) its own function and is still right.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: downloads and compiles the KURE-Sparse-nano ANE package; run with --ignored"]
async fn functions_load_on_first_use_and_a_late_long_text_is_still_correct() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inl-sparse/tests/fixtures/kure_sparse_nano/st_f32_golden.json");
    let golden: SparseGolden = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let encoder = AneSparseEncoder::new(AneOptions {
        id: "nlpai-lab/KURE-Sparse-nano".into(),
        ..AneOptions::default()
    })
    .unwrap();

    // `load` must not touch Core ML functions: it is the cold-start cost a
    // `minsync query` pays before its first (short) query.
    let started = std::time::Instant::now();
    encoder.load().await.unwrap();
    let load = started.elapsed().as_secs_f32();
    eprintln!("load {load:.2}s");
    assert!(
        load < 1.5,
        "load took {load:.2}s: functions are loaded eagerly again"
    );

    let query = format!("[Q] {}", golden.query);
    let started = std::time::Instant::now();
    let first = encoder.encode(vec![query.clone()]).await.unwrap();
    let first_query = started.elapsed().as_secs_f32();
    eprintln!("first short query {first_query:.2}s");
    assert!(
        first_query < 1.5,
        "the first short query took {first_query:.2}s: it loaded more than its function"
    );
    assert!(sparse_cosine(&first[0], &SparseRaw::vector(&golden.q)) > 0.998);

    // The longest golden document goes through a larger function, loaded now.
    let longest = golden.docs.iter().max_by_key(|doc| doc.len()).unwrap();
    let index = golden.docs.iter().position(|doc| doc == longest).unwrap();
    let late = encoder
        .encode(vec![format!("[D] {longest}")])
        .await
        .unwrap();
    let similarity = sparse_cosine(&late[0], &SparseRaw::vector(&golden.d[index]));
    eprintln!("late long text cosine {similarity:.6}");
    assert!(
        similarity > 0.998,
        "late-loaded function cosine {similarity}"
    );
    // The first function still answers the same after the others loaded.
    let again = encoder.encode(vec![query]).await.unwrap();
    assert!(sparse_cosine(&first[0], &again[0]) > 0.99999);
}

/// A short query that arrives while a many-call request runs must not wait for
/// the whole request: the Neural Engine gate is taken per call, not per request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "live: downloads and compiles the Qwen3-Embedding ANE package; run with --ignored"]
async fn a_short_query_does_not_wait_for_a_whole_long_request() {
    let model = std::sync::Arc::new(
        AneEmbedder::new(AneOptions {
            id: "NomaDamas/qwen3-embedding-0.6b-ane".into(),
            ..AneOptions::default()
        })
        .unwrap(),
    );
    model.load().await.unwrap();
    model.embed(vec!["warm".into()]).await.unwrap();

    // Many long texts: the planner turns them into many calls.
    let long: Vec<String> = (0..24)
        .map(|i| format!("{i} {}", "retrieval embedding vector ".repeat(300)))
        .collect();
    let started = std::time::Instant::now();
    let big = {
        let model = model.clone();
        tokio::spawn(async move { model.embed(long).await.unwrap().len() })
    };
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let short = std::time::Instant::now();
    model.embed(vec!["a short query".into()]).await.unwrap();
    let short = short.elapsed();
    assert_eq!(big.await.unwrap(), 24);
    let big = started.elapsed();
    eprintln!("short query {short:?} while the long request took {big:?}");
    assert!(
        short < big / 4,
        "the short query took {short:?} of the long request's {big:?}: it queued behind the whole request"
    );
}
