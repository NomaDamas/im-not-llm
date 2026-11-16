//! Live check of EmbeddingGemma 2 on the Apple Neural Engine against the
//! fp32 Sentence-Transformers reference the candle engine is tested with
//! (`inl-text/tests/fixtures/embeddinggemma2/golden_real.json`).
//!
//! Downloads the pinned package (about 900 MB) and compiles it for the
//! Neural Engine on first use (minutes); run with
//! `cargo test --release -p inl-ane --test gemma2_ane_live -- --ignored --nocapture`.
//!
//! The graphs run in fp16 on the ANE and still meet the 0.9999 cosine the
//! candle engine's f32 test holds (measured: at least 0.99996 on every row).
#![cfg(target_os = "macos")]

use inl_ane::{AneEmbedder, AneOptions};
use inl_core::DenseEngine;
use serde::Deserialize;

const MIN_COSINE: f32 = 0.9999;

#[derive(Deserialize)]
struct Golden {
    texts: Vec<String>,
    emb: Vec<Vec<f32>>,
    ids: Vec<Vec<u32>>,
}

fn golden() -> Golden {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inl-text/tests/fixtures/embeddinggemma2/golden_real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: downloads and compiles the EmbeddingGemma 2 ANE package; run with --ignored"]
async fn matches_the_reference_batched_and_alone() {
    let golden = golden();
    let model = AneEmbedder::new(AneOptions {
        id: "google/embeddinggemma-2".into(),
        ..AneOptions::default()
    })
    .unwrap();
    let started = std::time::Instant::now();
    let info = model.load().await.unwrap();
    eprintln!("load + self-test: {:.1}s", started.elapsed().as_secs_f32());
    assert_eq!(info.dimension, 768);
    assert_eq!(info.device, "ane");
    assert_eq!(info.max_length, 2048);

    let counts = model.token_counts(&golden.texts).await.unwrap();
    for (row, ids) in golden.ids.iter().enumerate() {
        if ids.len() < info.max_length {
            assert_eq!(counts[row], ids.len(), "row {row} token count");
        }
    }

    // Short texts share calls, the long ones use the windowed functions, one
    // is truncated at 2048.
    let started = std::time::Instant::now();
    let batched = model.embed(golden.texts.clone()).await.unwrap();
    eprintln!("batch of {}: {:?}", golden.texts.len(), started.elapsed());
    for (row, expected) in golden.emb.iter().enumerate() {
        let norm: f32 = batched[row].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "row {row}: norm {norm}");
        let similarity = cosine(&batched[row], expected);
        eprintln!(
            "row {row}: {} tokens, cosine {similarity:.6}",
            golden.ids[row].len()
        );
        assert!(
            similarity > MIN_COSINE,
            "batched row {row}: cosine {similarity}"
        );
        let alone = model.embed(vec![golden.texts[row].clone()]).await.unwrap();
        let packed_vs_alone = cosine(&alone[0], &batched[row]);
        assert!(
            packed_vs_alone > 0.9999,
            "row {row}: packed vs alone cosine {packed_vs_alone}"
        );
    }

    assert!(model.unload().await);
    assert!(!model.is_loaded());
    let again = model.embed(vec![golden.texts[0].clone()]).await.unwrap();
    assert!(cosine(&again[0], &golden.emb[0]) > MIN_COSINE);
}
