//! Stage 4 golden: a short clip embeds close to the reference. Needs the
//! cached checkpoint and an `ffmpeg` on `PATH`; skipped cleanly without either.

mod common;

use common::*;
use inl_core::{MediaBytes, MultimodalEngine, MultimodalInput};
use inl_vision::video::ffmpeg_binary;
use inl_vision::{DeviceSpec, MultimodalEmbedder, MultimodalOptions};
use std::process::{Command, Stdio};

fn ffmpeg_available() -> bool {
    Command::new(ffmpeg_binary())
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[tokio::test]
#[ignore = "loads the real checkpoint and decodes video with ffmpeg"]
async fn video_embedding_matches_the_reference() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = video_golden();
    let options = MultimodalOptions::default()
        .with_id("google/embeddinggemma-2")
        .with_device(DeviceSpec::Cpu)
        .with_video_fps(1.0)
        .with_video_max_soft_tokens(golden.max_soft_tokens);
    let embedder = MultimodalEmbedder::new(options);
    let input = MultimodalInput {
        videos: vec![MediaBytes {
            bytes: video_bytes(&golden.video),
            mime: Some("video/mp4".to_string()),
        }],
        ..Default::default()
    };
    let embedding = embedder.embed(vec![input]).await.unwrap().remove(0);
    let cos = cosine(&embedding, &golden.embedding);
    eprintln!(
        "video-only cosine vs reference = {cos:.8} ({} frames)",
        golden.frames
    );
    assert!(cos > 0.999, "video-only cosine {cos}");
}
