//! Shared helpers for the EmbeddingGemma 2 golden tests.

#![allow(dead_code)]

use base64::Engine;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub struct Stage {
    pub rows: Vec<usize>,
    pub cols: usize,
    pub data_b64: String,
}

impl Stage {
    /// The stored rows as `[len(rows), cols]` f32, in the same order as `rows`.
    pub fn to_rows(&self) -> Vec<Vec<f32>> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&self.data_b64)
            .expect("base64");
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values.len(), self.rows.len() * self.cols);
        values.chunks(self.cols).map(|r| r.to_vec()).collect()
    }
}

#[derive(Deserialize)]
pub struct ImageGolden {
    pub name: String,
    pub size: Vec<usize>,
    pub padded_patches: usize,
    pub real_patches: usize,
    pub pixel_values_shape: Vec<usize>,
    pub pixel_values: Stage,
    pub position_ids_head: Vec<Vec<i64>>,
    pub position_ids_tail: Vec<Vec<i64>>,
    pub num_soft_tokens: usize,
    pub vision_tower_shape: Vec<usize>,
    pub vision_tower: Stage,
    pub soft_tokens_shape: Vec<usize>,
    pub soft_tokens: Stage,
    pub embedding: Vec<f32>,
}

#[derive(Deserialize)]
pub struct TextImageGolden {
    pub text: String,
    pub image: String,
    pub embedding: Vec<f32>,
}

#[derive(Deserialize)]
pub struct QueryGolden {
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Deserialize)]
pub struct Golden {
    pub images: Vec<ImageGolden>,
    pub text_image: TextImageGolden,
    pub query: QueryGolden,
}

#[derive(Deserialize)]
pub struct VideoGolden {
    pub video: String,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub max_soft_tokens: usize,
    pub embedding: Vec<f32>,
}

pub fn video_golden() -> VideoGolden {
    serde_json::from_slice(&std::fs::read(fixture_dir().join("golden_video.json")).unwrap())
        .expect("golden_video.json parses")
}

pub fn video_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(name)).unwrap()
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/embeddinggemma2")
}

pub fn golden() -> Golden {
    serde_json::from_slice(&std::fs::read(fixture_dir().join("golden_vision.json")).unwrap())
        .expect("golden_vision.json parses")
}

pub fn image_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture_dir().join(format!("{name}.png"))).unwrap()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}
