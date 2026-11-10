//! Video goldens: video-only, interleaved text + video, non-default
//! `video_fps` / `video_max_soft_tokens`, and a 40 s clip past the 32-frame
//! cap, each against the reference processor's own frame sampling. Needs the
//! cached checkpoint and an `ffmpeg` + `ffprobe` (on `PATH` or via
//! `INL_FFMPEG`); skipped cleanly without them.

mod common;

use common::*;
use inl_core::{MediaBytes, MultimodalEngine, MultimodalInput};
use inl_vision::video::{ffmpeg_binary, probe, sample_indices, DEFAULT_MAX_FRAMES};
use inl_vision::{DeviceSpec, MultimodalEmbedder, MultimodalOptions};
use std::path::Path;
use std::process::{Command, Stdio};

fn ffmpeg_available() -> bool {
    Command::new(ffmpeg_binary())
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn ffprobe() -> String {
    Path::new(&ffmpeg_binary())
        .with_file_name("ffprobe")
        .to_string_lossy()
        .into_owned()
}

/// `INL_TEST_DEVICE=cpu|metal|cuda` runs the golden on that device (default
/// CPU): the reference vectors are CPU f32 outputs of the Hugging Face model.
fn options(fps: f64, tokens: usize) -> MultimodalOptions {
    let device = std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    MultimodalOptions::default()
        .with_id("google/embeddinggemma-2")
        .with_device(DeviceSpec::parse(Some(&device)).unwrap())
        .with_video_fps(fps)
        .with_video_max_soft_tokens(tokens)
}

fn input(case: &VideoCase) -> MultimodalInput {
    MultimodalInput {
        text: case.text.clone(),
        videos: vec![MediaBytes {
            bytes: video_bytes(&case.video),
            mime: Some("video/mp4".to_string()),
        }],
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "loads the real checkpoint and decodes video with ffmpeg"]
async fn video_embeddings_match_the_reference() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = video_golden();
    for case in &golden.cases {
        // The engine samples the same frame count the reference did.
        let info = probe(&ffprobe(), &fixture_dir().join(&case.video)).unwrap();
        let sampled = sample_indices(&info, case.fps, DEFAULT_MAX_FRAMES).len();
        assert_eq!(sampled, case.frames, "{}: sampled frames", case.name);
    }

    // Cases sharing the default settings run batched together (padding,
    // mixed sequence lengths); the rest each get their own embedder.
    let defaults: Vec<&VideoCase> = golden
        .cases
        .iter()
        .filter(|case| case.fps == 1.0 && case.max_soft_tokens == 140)
        .collect();
    let mut runs: Vec<(MultimodalOptions, Vec<&VideoCase>)> = vec![(options(1.0, 140), defaults)];
    for case in golden
        .cases
        .iter()
        .filter(|case| case.fps != 1.0 || case.max_soft_tokens != 140)
    {
        runs.push((options(case.fps, case.max_soft_tokens), vec![case]));
    }

    let mut checked = 0;
    for (options, cases) in runs {
        let embedder = MultimodalEmbedder::new(options);
        let embeddings = embedder
            .embed(cases.iter().map(|case| input(case)).collect())
            .await
            .unwrap();
        for (case, embedding) in cases.iter().zip(&embeddings) {
            let cos = cosine(embedding, &case.embedding);
            eprintln!(
                "{}: cosine vs reference = {cos:.8} ({} frames)",
                case.name, case.frames
            );
            assert!(cos > 0.9999, "{}: cosine {cos}", case.name);
            checked += 1;
        }
    }
    assert_eq!(checked, golden.cases.len());
}
