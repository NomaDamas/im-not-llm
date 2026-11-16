//! Live checks of the Apple Neural Engine embedder against the fp32
//! Sentence-Transformers references of `inl-text`
//! (`inl-text/tests/fixtures/*/golden_real.json`).
//!
//! Each test downloads its pinned package and compiles it for the Neural
//! Engine on first use (minutes); run with
//! `cargo test --release -p inl-ane --test ane_live -- --ignored --nocapture`.
#![cfg(target_os = "macos")]

use inl_ane::{AneEmbedder, AneOptions};
use inl_core::DenseEngine;
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
