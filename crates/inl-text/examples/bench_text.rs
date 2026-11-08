//! Timing harness for the dense text embedders: embed `--count` synthetic
//! passages of `--length` words and print chunks/s and tokens/s as JSON.
//!
//!     cargo run --release -p inl-text --example bench_text -- \
//!         --device metal --dtype f16 --batch-size 64 --model Qwen/Qwen3-Embedding-0.6B
//!
//! Used for per-device optimization (gpu01, Windows); it measures, it does
//! not assert.

use inl_text::{DeviceSpec, DtypeSpec, TextEmbedder, TextOptions};
use std::time::Instant;

struct Args {
    model: String,
    device: Option<String>,
    dtype: Option<String>,
    batch_size: usize,
    count: usize,
    length: usize,
    output_dim: Option<usize>,
    max_length: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            device: None,
            dtype: None,
            batch_size: 64,
            count: 256,
            length: 128,
            output_dim: None,
            max_length: 2048,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => args.model = value()?,
            "--device" => args.device = Some(value()?),
            "--dtype" => args.dtype = Some(value()?),
            "--batch-size" => args.batch_size = value()?.parse().map_err(|e| format!("{e}"))?,
            "--count" => args.count = value()?.parse().map_err(|e| format!("{e}"))?,
            "--length" => args.length = value()?.parse().map_err(|e| format!("{e}"))?,
            "--max-length" => args.max_length = value()?.parse().map_err(|e| format!("{e}"))?,
            "--output-dim" => args.output_dim = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn synthetic_passage(index: usize, length: usize) -> String {
    (0..length)
        .map(|word| format!("w{}", (index + word) % 97))
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::main]
async fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            eprintln!(
                "usage: bench_text [--model ID] [--device auto|cpu|metal|cuda] \
                 [--dtype auto|f32|f16|bf16] [--batch-size N] [--count N] \
                 [--length WORDS] [--max-length TOKENS] [--output-dim N]"
            );
            std::process::exit(2);
        }
    };

    let options = TextOptions::default()
        .with_id(args.model.clone())
        .with_device(DeviceSpec::parse(args.device.as_deref()).unwrap())
        .with_dtype(DtypeSpec::parse(args.dtype.as_deref()).unwrap())
        .with_batch_size(args.batch_size)
        .with_max_length(args.max_length)
        .with_output_dim(args.output_dim);
    let embedder = TextEmbedder::new(options).expect("build embedder");
    let identity = embedder.identity().await.expect("load model");

    let texts: Vec<String> = (0..args.count)
        .map(|index| synthetic_passage(index, args.length))
        .collect();
    let tokens: usize = embedder
        .token_counts(&texts)
        .await
        .expect("tokenize")
        .into_iter()
        .sum();

    let started = Instant::now();
    let vectors = embedder.embed(&texts).await.expect("embed");
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(vectors.len(), args.count);

    let report = serde_json::json!({
        "model": identity.id,
        "device": identity.device,
        "dtype": identity.dtype,
        "dimension": identity.dimension,
        "batch_size": args.batch_size,
        "count": args.count,
        "words_per_passage": args.length,
        "tokens": tokens,
        "seconds": seconds,
        "chunks_per_second": args.count as f64 / seconds,
        "tokens_per_second": tokens as f64 / seconds,
    });
    println!("{report}");
}
