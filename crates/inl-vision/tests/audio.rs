//! Audio goldens against `scripts/gen_audio_golden.py`: the log-mel features
//! and the 30 s cap (need `ffmpeg`), the audio tower output and the engine's
//! embeddings (need the cached checkpoint, so `#[ignore]`d). All skip cleanly
//! without `ffmpeg`.

mod common;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use common::*;
use inl_core::device::DeviceSpec;
use inl_core::hub::ModelSource;
use inl_core::{MediaBytes, Modality, MultimodalEngine, MultimodalInput};
use inl_vision::audio::{self, AudioConfig, AudioTower};
use inl_vision::video::ffmpeg_binary;
use inl_vision::{MultimodalEmbedder, MultimodalOptions};
use std::process::{Command, Stdio};

fn ffmpeg_available() -> bool {
    Command::new(ffmpeg_binary())
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn features(golden: &AudioGolden) -> audio::AudioFeatures {
    let samples =
        audio::decode(&std::fs::read(fixture_dir().join(&golden.audio)).unwrap()).unwrap();
    assert_eq!(samples.len(), golden.samples);
    audio::log_mel(&samples).unwrap()
}

fn rows(data: &[f32], cols: usize) -> Vec<Vec<f32>> {
    data.chunks(cols).map(<[f32]>::to_vec).collect()
}

#[test]
fn log_mel_features_match_the_reference() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = audio_golden();
    let features = features(&golden);
    assert_eq!(features.frames, golden.feature_frames);
    assert_eq!(features.valid, golden.feature_valid);
    assert_eq!(features.tokens(), golden.audio_tokens);
    let (worst, worst_cosine, magnitude) =
        compare_rows(&rows(&features.data, audio::MEL_BINS), &golden.features);
    eprintln!("log-mel: max abs diff = {worst:.3e}, max |ref| = {magnitude:.1}, min row cosine = {worst_cosine:.8}");
    assert!(worst <= 1e-4, "log-mel differs by {worst}");
}

#[test]
#[ignore = "loads the real checkpoint"]
fn audio_tower_matches_the_reference() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = audio_golden();
    let features = features(&golden);
    let device = Device::Cpu;
    let source = ModelSource::resolve("google/embeddinggemma-2", std::env::temp_dir()).unwrap();
    let config = std::fs::read(source.fetch("config.json").unwrap()).unwrap();
    let cfg = AudioConfig::parse(&config)
        .unwrap()
        .expect("checkpoint has an audio tower");
    let weights = source.weight_files().unwrap();
    // SAFETY: the safetensors files are not modified while the tower is used.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &device).unwrap() };
    let tower = AudioTower::new(cfg, 512, vb).unwrap();

    let output = tower.forward(&[&features]).unwrap();
    assert_eq!(output.valid, vec![golden.tower_valid]);
    let hidden: Tensor = output.hidden.squeeze(0).unwrap();
    let actual: Vec<Vec<f32>> = hidden.to_vec2().unwrap();
    assert_eq!(actual.len(), golden.tower_frames);
    // The fixture clip has no padding rows after subsampling, so every
    // sampled row is a real frame.
    assert_eq!(golden.tower_valid, golden.tower_frames);
    let (worst, worst_cosine, magnitude) = compare_rows(&actual, &golden.tower);
    eprintln!(
        "audio_tower: {} rows, max abs diff = {worst:.3e}, max |ref| = {magnitude:.1}, \
         min row cosine = {worst_cosine:.8}",
        actual.len()
    );
    assert!(worst_cosine > 0.9999, "audio_tower cosine {worst_cosine}");
    assert!(
        worst <= 5e-5 * magnitude.max(1.0),
        "audio_tower differs by {worst} over {magnitude}"
    );
}

/// A mono 16-bit PCM WAV of `seconds` of a 440 Hz tone.
fn tone_wav(seconds: f64) -> Vec<u8> {
    let rate = audio::SAMPLE_RATE as u32;
    let samples: Vec<i16> = (0..(seconds * rate as f64) as usize)
        .map(|i| ((i as f64 * 440.0 * std::f64::consts::TAU / rate as f64).sin() * 8000.0) as i16)
        .collect();
    let data = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

#[test]
fn a_clip_over_30_seconds_is_cut_to_the_reference_token_count() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = audio_golden();
    let samples = audio::decode(&tone_wav(golden.long_seconds)).unwrap();
    assert_eq!(samples.len(), audio::MAX_SAMPLES);
    assert_eq!(
        audio::log_mel(&samples).unwrap().tokens(),
        golden.long_tokens
    );
}

fn clip(bytes: Vec<u8>) -> MediaBytes {
    MediaBytes {
        bytes,
        mime: Some("audio/wav".to_string()),
    }
}

/// `INL_TEST_DEVICE=cpu|metal|cuda` runs the golden on that device (default
/// CPU): the reference vectors are CPU f32 outputs of the Hugging Face model.
fn options() -> MultimodalOptions {
    let device = std::env::var("INL_TEST_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    MultimodalOptions::default()
        .with_id("google/embeddinggemma-2")
        .with_device(DeviceSpec::parse(Some(&device)).unwrap())
}

#[tokio::test]
#[ignore = "loads the real checkpoint"]
async fn audio_embeddings_match_the_reference() {
    if !ffmpeg_available() {
        eprintln!("skipping: ffmpeg is not on PATH (set INL_FFMPEG to point at one)");
        return;
    }
    let golden = audio_golden();
    let wav = std::fs::read(fixture_dir().join(&golden.audio)).unwrap();
    let embedder = MultimodalEmbedder::new(options());
    let info = embedder.load().await.unwrap();
    assert!(
        info.modalities.contains(&Modality::Audio),
        "{:?}",
        info.modalities
    );

    let audio_only = MultimodalInput {
        audios: vec![clip(wav.clone())],
        ..Default::default()
    };
    let interleaved = MultimodalInput {
        text: Some(golden.interleaved_text.clone()),
        audios: vec![clip(wav)],
        ..Default::default()
    };
    // Alone, then in one call with a 30 s clip: the short clip is padded to
    // the long one in a shared audio pass, which must not move its embedding.
    for (label, inputs) in [
        ("alone", vec![audio_only.clone(), interleaved.clone()]),
        (
            "batched with a 30 s clip",
            vec![
                MultimodalInput {
                    audios: vec![clip(tone_wav(golden.long_seconds))],
                    ..Default::default()
                },
                audio_only.clone(),
                interleaved.clone(),
            ],
        ),
    ] {
        let embeddings = embedder.embed(inputs).await.unwrap();
        let [.., audio, text_audio] = embeddings.as_slice() else {
            unreachable!("two or three inputs")
        };
        let cos = cosine(audio, &golden.embedding);
        eprintln!("{label}: audio-only cosine = {cos:.8}");
        assert!(cos > 0.9999, "{label}: audio-only cosine {cos}");
        let cos = cosine(text_audio, &golden.interleaved_embedding);
        eprintln!("{label}: text+audio cosine = {cos:.8}");
        assert!(cos > 0.9999, "{label}: text+audio cosine {cos}");
    }
}

#[tokio::test]
#[ignore = "loads the real checkpoint"]
async fn a_modality_left_out_at_load_is_rejected() {
    let embedder =
        MultimodalEmbedder::new(options().with_modalities(vec![Modality::Text, Modality::Image]));
    let info = embedder.load().await.unwrap();
    assert_eq!(info.modalities, vec![Modality::Text, Modality::Image]);
    let error = embedder
        .embed(vec![MultimodalInput {
            audios: vec![clip(tone_wav(1.0))],
            ..Default::default()
        }])
        .await
        .unwrap_err();
    assert_eq!(error.code(), "invalid_input");
}
