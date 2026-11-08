//! Live smoke check of the native Qwen3-Embedding-0.6B embedder: the CPU and
//! accelerator vectors of the same texts must agree.
//!
//! Needs the model in the Hugging Face cache (or network): run with
//! `cargo test --release --test qwen3_live -- --ignored --nocapture`.
//! `INL_TEST_DEVICE=metal|cuda` picks the accelerator (default `auto`, which
//! is the best device this build has; on a CPU-only build the test then only
//! compares the CPU with itself and says so).

use inl_text::{DeviceSpec, TextEmbedder, TextOptions};
use std::str::FromStr;

const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

fn embedder(device: DeviceSpec) -> TextEmbedder {
    TextEmbedder::new(TextOptions::default().with_id(MODEL).with_device(device)).unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads Qwen/Qwen3-Embedding-0.6B; run with --ignored"]
async fn cpu_and_accelerator_agree() {
    let texts = vec![
        "What causes the northern lights?".to_string(),
        "환불 정책에 따라 구매 증빙 인증서를 제출한 고객은 14일 이내에 전액 환불을 받을 수 있습니다."
            .to_string(),
    ];

    let cpu = embedder(DeviceSpec::Cpu);
    let cpu_vectors = cpu.embed(&texts).await.unwrap();
    assert_eq!(cpu_vectors.len(), 2);
    let cpu_identity = cpu.identity().await.unwrap();
    println!(
        "cpu: dimension {} dtype {} device {}",
        cpu_identity.dimension, cpu_identity.dtype, cpu_identity.device
    );

    let spec = std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "auto".to_string());
    let accelerator = embedder(DeviceSpec::from_str(&spec).unwrap());
    let accelerated = accelerator.embed(&texts).await.unwrap();
    let identity = accelerator.identity().await.unwrap();
    println!(
        "{spec}: dimension {} dtype {} device {}",
        identity.dimension, identity.dtype, identity.device
    );
    if identity.device == "cpu" {
        println!("no accelerator in this build: comparing the CPU with itself");
    }

    assert_eq!(cpu_identity.dimension, identity.dimension);
    for (row, (a, b)) in cpu_vectors.iter().zip(&accelerated).enumerate() {
        let similarity = cosine(a, b);
        println!("row {row}: cosine(cpu, {}) = {similarity}", identity.device);
        assert!(similarity > 0.999, "row {row}: cosine {similarity}");
    }
}
