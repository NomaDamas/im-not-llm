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
#[ignore = "loads the real checkpoint on the CPU and Metal"]
async fn cpu_and_metal_agree() {
    let golden = golden();
    let cpu = embed_on(DeviceSpec::Cpu).await.expect("cpu works");
    let cos = cosine(&cpu, &golden.images[0].embedding);
    eprintln!("cpu image-only cosine vs reference = {cos:.8}");
    assert!(cos > 0.9999);

    #[cfg(target_os = "macos")]
    {
        let metal = embed_on(DeviceSpec::Metal).await.expect("metal works");
        let cos = cosine(&metal, &golden.images[0].embedding);
        eprintln!("metal image-only cosine vs reference = {cos:.8}");
        assert!(cos > 0.9999);
        let cos = cosine(&cpu, &metal);
        eprintln!("cpu vs metal cosine = {cos:.8}");
        assert!(cos > 0.9999);
        assert!(metal.iter().all(|v| v.is_finite()));
    }
}
