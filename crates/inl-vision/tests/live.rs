//! Live tests: run the real checkpoint on the CPU and (on macOS) Metal and
//! check the two agree. `#[ignore]`d; run with
//! `HF_HOME=... cargo test -p inl-vision --release --test live -- --ignored`.

mod common;

use common::*;
use inl_core::{MediaBytes, MultimodalEngine, MultimodalInput};
use inl_vision::{DeviceSpec, MultimodalEmbedder, MultimodalOptions};

fn media() -> MediaBytes {
    MediaBytes {
        bytes: image_bytes("img_a"),
        mime: None,
    }
}

async fn embed_on(device: DeviceSpec) -> Option<Vec<f32>> {
    let options = MultimodalOptions::default()
        .with_id("google/embeddinggemma-2")
        .with_device(device);
    let embedder = MultimodalEmbedder::new(options);
    match embedder
        .embed(vec![MultimodalInput {
            images: vec![media()],
            ..Default::default()
        }])
        .await
    {
        Ok(mut embeddings) => Some(embeddings.remove(0)),
        Err(error) => {
            eprintln!("{device:?} unavailable: {error}");
            None
        }
    }
}

#[tokio::test]
#[ignore = "loads the real checkpoint on the CPU and the accelerator"]
async fn cpu_and_accelerator_agree() {
    let golden = golden();
    let cpu = embed_on(DeviceSpec::Cpu).await.expect("cpu works");
    let cos = cosine(&cpu, &golden.images[0].embedding);
    eprintln!("cpu image-only cosine vs reference = {cos:.8}");
    assert!(cos > 0.9999);

    // `INL_TEST_DEVICE` picks the accelerator, as the text live tests do; the
    // default is Metal on macOS (the device this test was written for) and CPU
    // elsewhere, where there is no accelerator to compare against.
    let default = if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    };
    let spec = std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| default.to_string());
    let spec = DeviceSpec::parse(Some(&spec)).unwrap();
    if spec == DeviceSpec::Cpu {
        eprintln!(
            "no accelerator selected (INL_TEST_DEVICE={spec:?}): comparing the CPU with itself"
        );
        return;
    }
    let accelerated = embed_on(spec).await.expect("accelerator works");
    assert!(accelerated.iter().all(|value| value.is_finite()));
    let cos = cosine(&accelerated, &golden.images[0].embedding);
    eprintln!("{spec:?} image-only cosine vs reference = {cos:.8}");
    assert!(cos > 0.9999, "{spec:?} image-only cosine {cos}");
    let cos = cosine(&cpu, &accelerated);
    eprintln!("cpu vs {spec:?} cosine = {cos:.8}");
    assert!(cos > 0.9999, "cpu vs {spec:?} cosine {cos}");
}
