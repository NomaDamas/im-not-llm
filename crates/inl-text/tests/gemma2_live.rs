//! Live check of the native EmbeddingGemma 2 embedder against the REAL
//! `google/embeddinggemma-2` checkpoint, compared with the reference
//! implementation (Sentence-Transformers, fp32, CPU; `golden_real.json`).
//!
//! Needs the model in the Hugging Face cache (or network): run with
//! `cargo test --release --test gemma2_live -- --ignored --nocapture`.
//! `INL_TEST_DEVICE=cpu|metal` picks the device (default cpu).

use inl_text::{DeviceSpec, DtypeSpec, TextEmbedder, TextOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    texts: Vec<String>,
    emb: Vec<Vec<f32>>,
    emb256: Vec<Vec<f32>>,
}

fn golden() -> Golden {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/embeddinggemma2/golden_real.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn device() -> DeviceSpec {
    let value = std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".into());
    DeviceSpec::parse(Some(&value)).unwrap()
}

fn embedder(dtype: DtypeSpec, output_dim: Option<usize>) -> TextEmbedder {
    TextEmbedder::new(
        TextOptions::default()
            .with_id("google/embeddinggemma-2")
            .with_device(device())
            .with_dtype(Some(dtype))
            .with_output_dim(output_dim),
    )
    .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads google/embeddinggemma-2; run with --ignored"]
async fn matches_the_sentence_transformers_reference_in_f32() {
    let golden = golden();
    let model = embedder(DtypeSpec::F32, None);
    let identity = model.identity().await.unwrap();
    assert_eq!(identity.dimension, 768);
    assert_eq!(identity.dtype, "f32");

    // Every text carries its own task prefix, so no prefix is configured. Two
    // of them exceed the 512-token sliding radius, and one the 2048 context.
    let batched = model.embed(&golden.texts).await.unwrap();
    for (row, expected) in golden.emb.iter().enumerate() {
        let similarity = cosine(&batched[row], expected);
        assert!(
            similarity > 0.9999,
            "batched row {row}: cosine {similarity}"
        );
        let alone = model
            .embed(std::slice::from_ref(&golden.texts[row]))
            .await
            .unwrap();
        let similarity = cosine(&alone[0], expected);
        assert!(similarity > 0.9999, "alone row {row}: cosine {similarity}");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads google/embeddinggemma-2; run with --ignored"]
async fn matryoshka_256_matches_the_reference_truncation() {
    let golden = golden();
    let model = embedder(DtypeSpec::F32, Some(256));
    let identity = model.identity().await.unwrap();
    assert_eq!(identity.dimension, 256);

    let vectors = model.embed(&golden.texts).await.unwrap();
    for (row, expected) in golden.emb256.iter().enumerate() {
        assert_eq!(vectors[row].len(), 256);
        let similarity = cosine(&vectors[row], expected);
        assert!(
            similarity > 0.9999,
            "row {row}: cosine {similarity} against the reference truncation"
        );
    }
}
