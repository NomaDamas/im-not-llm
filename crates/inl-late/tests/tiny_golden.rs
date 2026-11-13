//! The tiny random late-interaction checkpoint against the Hugging Face
//! reference (`scripts/gen_tiny_late_fixture.py`): Qwen3.5 tower with
//! gated-delta-net and bidirectional full-attention layers, `1_Dense`, the
//! document punctuation skiplist and L2 normalization.

use inl_late::{LateEncoder, LateOptions, MultiVector, Task};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Golden {
    dim: usize,
    texts: Vec<String>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    task: String,
    batched: Vec<MultiVector>,
    alone: Vec<MultiVector>,
}

/// Largest absolute difference between two fp32 runs of the same function;
/// the batched reference itself differs from the unbatched one by about 1e-6.
const TOLERANCE: f32 = 1e-4;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_qwen3_5_late")
}

fn golden() -> Golden {
    serde_json::from_slice(&std::fs::read(fixture().join("golden.json")).unwrap()).unwrap()
}

fn encoder(batch_size: usize) -> LateEncoder {
    LateEncoder::new(
        LateOptions::default()
            .with_id(fixture().to_string_lossy())
            .with_device(std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()))
            .with_batch_size(batch_size),
    )
    .unwrap()
}

fn task(name: &str) -> Task {
    match name {
        "query" => Task::Query,
        "document" => Task::Document,
        other => panic!("unknown task {other}"),
    }
}

fn max_diff(ours: &[MultiVector], theirs: &[MultiVector], label: &str) -> f32 {
    assert_eq!(ours.len(), theirs.len(), "{label}: input count");
    let mut worst = 0f32;
    for (index, (a, b)) in ours.iter().zip(theirs).enumerate() {
        assert_eq!(a.len(), b.len(), "{label} input {index}: token vectors");
        for (row_a, row_b) in a.iter().zip(b) {
            assert_eq!(row_a.len(), row_b.len(), "{label} input {index}: dimension");
            for (x, y) in row_a.iter().zip(row_b) {
                worst = worst.max((x - y).abs());
            }
        }
    }
    worst
}

#[tokio::test]
async fn tiny_checkpoint_matches_the_reference_batched_and_alone() {
    let golden = golden();
    for case in &golden.cases {
        // Batched: every input in one right-padded forward pass.
        let batched = encoder(64)
            .encode(golden.texts.clone(), task(&case.task))
            .await
            .unwrap();
        let diff = max_diff(&batched, &case.batched, &format!("{} batched", case.task));
        assert!(diff < TOLERANCE, "{} batched differs by {diff}", case.task);
        // One input per pass: padding must not leak into any row.
        let alone = encoder(1)
            .encode(golden.texts.clone(), task(&case.task))
            .await
            .unwrap();
        let diff = max_diff(&alone, &case.alone, &format!("{} alone", case.task));
        assert!(diff < TOLERANCE, "{} alone differs by {diff}", case.task);
        for vector in batched.iter().flatten() {
            assert_eq!(vector.len(), golden.dim);
        }
    }
}

#[tokio::test]
async fn identity_reports_the_projection_and_the_document_length() {
    let info = encoder(8).identity().await.unwrap();
    assert_eq!(info.dimension, 12);
    assert_eq!(info.max_length, 160);
    assert_eq!(info.dtype, "f32");
}

#[tokio::test]
async fn a_shorter_max_length_caps_both_tasks() {
    let encoder = LateEncoder::new(
        LateOptions::default()
            .with_id(fixture().to_string_lossy())
            .with_device(std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()))
            .with_max_length(10),
    )
    .unwrap();
    let text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12".to_string();
    let query = encoder
        .encode(vec![text.clone()], Task::Query)
        .await
        .unwrap();
    let document = encoder.encode(vec![text], Task::Document).await.unwrap();
    assert_eq!(query[0].len(), 10);
    assert_eq!(document[0].len(), 10);
}
