//! Live check of the native late-interaction encoder against the REAL
//! `perplexity-ai/pplx-embed-v2-late-0.6b` checkpoint, compared with the
//! reference (Sentence-Transformers `MultiVectorEncoder`, fp32, CPU;
//! `scripts/gen_late_golden.py`).
//!
//! Needs the model in the Hugging Face cache (or network): run with
//! `cargo test --release -p inl-late --test late_live -- --ignored --nocapture`.
//! `INL_TEST_DEVICE=cpu|metal|cuda|auto` picks the device (default cpu).

use inl_late::{LateEncoder, LateOptions, MultiVector, Task};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    task: String,
    text: String,
    vectors: MultiVector,
}

/// fp32 against fp32 through the full tower; a wrong mask, rope or recurrence
/// moves single components by 1e-1 or more.
const TOLERANCE: f32 = 2e-3;

fn golden() -> Vec<Case> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pplx_embed_v2_late/st_f32_golden.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn encoder(batch_size: usize) -> LateEncoder {
    LateEncoder::new(
        LateOptions::default()
            .with_device(std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()))
            .with_dtype("f32")
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

fn check(ours: &MultiVector, theirs: &MultiVector, label: &str) {
    assert_eq!(ours.len(), theirs.len(), "{label}: token vectors");
    let mut worst = 0f32;
    for (a, b) in ours.iter().zip(theirs) {
        assert_eq!(a.len(), b.len(), "{label}: dimension");
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((x - y).abs());
        }
    }
    println!("{label}: {} vectors, max |diff| {worst:.2e}", ours.len());
    assert!(worst < TOLERANCE, "{label} differs by {worst}");
}

#[tokio::test]
#[ignore = "live: loads the real perplexity-ai/pplx-embed-v2-late-0.6b checkpoint"]
async fn native_encoder_matches_sentence_transformers_on_the_real_model() {
    let cases = golden();
    for batch_size in [1, 8] {
        let encoder = encoder(batch_size);
        for name in ["query", "document"] {
            // Exercise an actual eight-input forward, not just a limit of eight.
            let group: Vec<&Case> = cases
                .iter()
                .filter(|case| case.task == name)
                .cycle()
                .take(if batch_size == 1 {
                    cases.iter().filter(|case| case.task == name).count()
                } else {
                    8
                })
                .collect();
            let texts = group.iter().map(|case| case.text.clone()).collect();
            let ours = encoder.encode(texts, task(name)).await.unwrap();
            for (vectors, case) in ours.iter().zip(&group) {
                check(
                    vectors,
                    &case.vectors,
                    &format!("batch {batch_size} {name} {} bytes", case.text.len()),
                );
            }
        }
    }
}
