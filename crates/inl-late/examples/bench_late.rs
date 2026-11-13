//! Warm encode latency, including tokenization and host output, excluding loading.
//! cargo run --release -p inl-late --example bench_late -- --device metal --dtype bf16
use inl_late::{LateEncoder, LateEngine, LateOptions, Task};
use std::time::Instant;

#[tokio::main]
async fn main() {
    let mut options = LateOptions::default();
    let mut repeats = 5usize;
    let mut document_repeats = 26usize;
    let mut only = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv.next().expect("flag needs a value");
        match flag.as_str() {
            "--device" => options.device = Some(value),
            "--dtype" => options.dtype = Some(value),
            "--model" => options.id = value,
            "--batch-size" => options.batch_size = value.parse().expect("--batch-size"),
            "--repeats" => repeats = value.parse().expect("--repeats"),
            "--document-repeats" => document_repeats = value.parse().expect("--document-repeats"),
            "--only" => {
                assert!(matches!(
                    value.as_str(),
                    "query" | "document" | "long_document"
                ));
                only = Some(value);
            }
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(repeats > 0);
    let encoder = LateEncoder::new(options).expect("encoder");
    let info = encoder.identity().await.expect("load");
    for (label, text, task) in [
        (
            "query",
            "What is the capital of France?".to_string(),
            Task::Query,
        ),
        (
            "document",
            "Paris is the capital and largest city of France. ".repeat(document_repeats),
            Task::Document,
        ),
        (
            "long_document",
            "Paris is the capital and largest city of France. ".repeat(102),
            Task::Document,
        ),
    ] {
        if only.as_deref().is_some_and(|only| only != label) {
            continue;
        }
        for batch in [1, 8] {
            let texts = vec![text.clone(); batch];
            let tokens = encoder
                .token_counts(&texts, task)
                .await
                .expect("token counts");
            let warm = encoder.encode(texts.clone(), task).await.expect("warmup");
            assert!(warm.iter().flatten().flatten().all(|v| v.is_finite()));
            let mut timings = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                let output = encoder.encode(texts.clone(), task).await.expect("encode");
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(output);
            }
            timings.sort_by(f64::total_cmp);
            let median = timings[repeats / 2];
            println!(
                "{}",
                serde_json::json!({
                    "model": info.model_id, "device": info.device, "dtype": info.dtype,
                    "attention": info.attention, "input": label, "batch": batch,
                    "tokens_each": tokens[0], "median_ms": median,
                    "min_ms": timings[0], "max_ms": timings[repeats-1],
                    "inputs_per_second": batch as f64 * 1000.0 / median,
                    "tokens_per_second": tokens.iter().sum::<usize>() as f64 * 1000.0 / median,
                })
            );
        }
    }
}
