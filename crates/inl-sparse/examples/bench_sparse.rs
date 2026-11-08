//! Throughput benchmark for the in-process sparse encoder.
//!
//! ```text
//! cargo run --release --example bench_sparse -- \
//!     [--device cpu|metal|cuda|auto] [--dtype f32|f16|bf16] \
//!     [--batch-size N] [--max-batch-tokens N] [--n N] [--len N]
//! ```
//!
//! Prints one JSON object with chunks/s and tokens/s. `tokens` counts what the
//! tokenizer actually produced (before any context cut).

use inl_sparse::{SparseEncoder, SparseEngine, SparseOptions, SparseVector};

struct Args {
    device: String,
    dtype: String,
    batch_size: usize,
    max_batch_tokens: usize,
    n: usize,
    len: usize,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            device: "auto".to_string(),
            dtype: "f32".to_string(),
            batch_size: 64,
            max_batch_tokens: 1024,
            n: 128,
            len: 300,
        };
        let mut argv = std::env::args().skip(1);
        while let Some(flag) = argv.next() {
            let mut value = || {
                argv.next()
                    .unwrap_or_else(|| panic!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--device" => args.device = value(),
                "--dtype" => args.dtype = value(),
                "--batch-size" => args.batch_size = value().parse().expect("--batch-size"),
                "--max-batch-tokens" => {
                    args.max_batch_tokens = value().parse().expect("--max-batch-tokens")
                }
                "--n" => args.n = value().parse().expect("--n"),
                "--len" => args.len = value().parse().expect("--len"),
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

fn chunks(n: usize, len: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let mut text = String::with_capacity(len * 10);
            for word in 0..len {
                text.push_str(&format!("어절{word} "));
            }
            format!("{i} {text}")
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let options = SparseOptions {
        device: Some(args.device.clone()),
        dtype: Some(args.dtype.clone()),
        batch_size: args.batch_size,
        max_batch_tokens: args.max_batch_tokens,
        ..SparseOptions::default()
    };
    let encoder = SparseEncoder::new(options).expect("sparse encoder");
    let identity = encoder.identity().await.expect("load model");
    let corpus = chunks(args.n, args.len);

    encoder
        .encode(&corpus[..4.min(corpus.len())])
        .await
        .expect("warmup");
    let started = std::time::Instant::now();
    let vectors = encoder.encode(&corpus).await.expect("encode");
    let seconds = started.elapsed().as_secs_f64();

    let tokens: usize = encoder
        .token_counts(&corpus)
        .await
        .expect("token counts")
        .iter()
        .sum();
    let nnz = vectors.iter().map(SparseVector::len).sum::<usize>() / vectors.len().max(1);

    let report = serde_json::json!({
        "device": identity.device,
        "dtype": identity.dtype,
        "attention": identity.attention,
        "dimension": identity.dimension,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "chunks": corpus.len(),
        "words_per_chunk": args.len,
        "seconds": seconds,
        "chunks_per_second": corpus.len() as f64 / seconds,
        "tokens": tokens,
        "tokens_per_second": tokens as f64 / seconds,
        "mean_nnz": nnz,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
