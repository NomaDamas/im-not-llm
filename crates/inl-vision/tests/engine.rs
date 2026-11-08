//! Stage 3 golden: the full multimodal embeddings match the reference
//! (image-only, interleaved text+image, and text-only parity with `inl-text`).
//! Needs the cached checkpoint, so it is `#[ignore]`d; run it with
//! `HF_HOME=... cargo test -p inl-vision --release --test engine -- --ignored`.

mod common;

use common::*;
use inl_core::{MediaBytes, MultimodalEngine, MultimodalInput};
use inl_vision::{DeviceSpec, MultimodalEmbedder, MultimodalOptions};

fn options() -> MultimodalOptions {
    MultimodalOptions::default()
        .with_id("google/embeddinggemma-2")
        .with_device(DeviceSpec::Cpu)
}

fn media(name: &str) -> MediaBytes {
    MediaBytes {
        bytes: image_bytes(name),
        mime: None,
    }
}

#[tokio::test]
#[ignore = "loads the real checkpoint and runs the vision tower"]
async fn multimodal_embeddings_match_the_reference() {
    let golden = golden();
    let embedder = MultimodalEmbedder::new(options());
    let info = embedder.load().await.unwrap();
    assert_eq!(info.dimension, 768);
    assert!(info.modalities.contains(&inl_core::Modality::Image));

    // Image-only (text absent): the item alone.
    let inputs: Vec<MultimodalInput> = golden
        .images
        .iter()
        .map(|image| MultimodalInput {
            images: vec![media(&image.name)],
            ..Default::default()
        })
        .collect();
    let embeddings = embedder.embed(inputs).await.unwrap();
    for (image, embedding) in golden.images.iter().zip(&embeddings) {
        let cos = cosine(embedding, &image.embedding);
        eprintln!("{} image-only cosine = {cos:.8}", image.name);
        assert!(cos > 0.9999, "{} image-only cosine {cos}", image.name);
    }

    // Interleaved text + image.
    let input = MultimodalInput {
        text: Some(golden.text_image.text.clone()),
        images: vec![media(&golden.text_image.image)],
        ..Default::default()
    };
    let embedding = embedder.embed(vec![input]).await.unwrap().remove(0);
    let cos = cosine(&embedding, &golden.text_image.embedding);
    eprintln!("text+image cosine = {cos:.8}");
    assert!(cos > 0.9999, "text+image cosine {cos}");

    // Text-only: parity with the reference and with `inl-text` alone.
    let embedding = embedder
        .embed(vec![MultimodalInput {
            text: Some(golden.query.text.clone()),
            ..Default::default()
        }])
        .await
        .unwrap()
        .remove(0);
    let cos = cosine(&embedding, &golden.query.embedding);
    eprintln!("text-only vs reference cosine = {cos:.8}");
    assert!(cos > 0.9999, "text-only cosine {cos}");

    let text = inl_text::Gemma2Embedder::from_hf(&inl_text::LoadOptions {
        repo_id: "google/embeddinggemma-2",
        default_cache_dir: std::env::temp_dir(),
        device: &candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
        max_length: 8192,
        attention: inl_core::attention::AttentionRequest::Auto,
    })
    .unwrap();
    let reference = text.embed(&[golden.query.text.as_str()]).unwrap().remove(0);
    let cos = cosine(&embedding, &reference);
    eprintln!("text-only vs inl-text cosine = {cos:.8}");
    assert!(cos > 0.9999, "text-only vs inl-text cosine {cos}");
}
