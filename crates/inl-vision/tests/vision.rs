//! Stage 2 golden: the vision tower output and the projected soft tokens match
//! the reference. Needs the cached `google/embeddinggemma-2` weights, so it is
//! `#[ignore]`d; run it with
//! `HF_HOME=... cargo test -p inl-vision --release --test vision -- --ignored`.

mod common;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use common::*;
use inl_core::hub::ModelSource;
use inl_vision::preprocess::{self, PreprocessedImage};
use inl_vision::vision::{VisionBatch, VisionConfig, VisionTower};

pub fn load_tower(device: &Device, dtype: DType) -> VisionTower {
    let source = ModelSource::resolve("google/embeddinggemma-2", std::env::temp_dir()).unwrap();
    let config = std::fs::read(source.fetch("config.json").unwrap()).unwrap();
    let cfg = VisionConfig::parse(&config).unwrap();
    let weights = source.weight_files().unwrap();
    // SAFETY: the safetensors files are not modified while the tower is used.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device).unwrap() };
    VisionTower::new(cfg, 512, vb).unwrap()
}

/// Build the `[1, max_patches, ...]` input tensors for one image.
pub fn inputs(pre: &PreprocessedImage, device: &Device) -> (Tensor, Tensor, Tensor) {
    let pixels = Tensor::from_vec(pre.pixel_values.clone(), (1, pre.max_patches, 768), device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let positions =
        Tensor::from_vec(pre.position_ids.clone(), (1, pre.max_patches, 2), device).unwrap();
    let padding: Vec<f32> = (0..pre.max_patches)
        .map(|p| if p < pre.real_patches { 0.0 } else { 1.0 })
        .collect();
    let padding = Tensor::from_vec(padding, (1, pre.max_patches), device).unwrap();
    (pixels, positions, padding)
}

#[test]
#[ignore = "loads the real checkpoint and runs 16 layers over 2520 patches"]
fn vision_tower_matches_the_reference() {
    let golden = golden();
    let device = Device::Cpu;
    let tower = load_tower(&device, DType::F32);
    for image in &golden.images {
        let pre = preprocess::preprocess(&image_bytes(&image.name), 280).unwrap();
        let (pixels, positions, padding) = inputs(&pre, &device);
        let output = tower
            .forward(VisionBatch {
                pixel_values: &pixels,
                position_ids: &positions,
                padding: &padding,
                output_length: 280,
            })
            .unwrap();
        assert_eq!(
            output.tokens_per_row,
            vec![image.num_soft_tokens],
            "{}",
            image.name
        );
        let rows: Vec<Vec<f32>> = output.hidden.to_vec2().unwrap();
        assert_eq!(rows.len(), image.vision_tower_shape[0], "{}", image.name);
        assert_eq!(rows[0].len(), image.vision_tower_shape[1], "{}", image.name);

        let expected = image.vision_tower.to_rows();
        let mut worst = 0.0f32;
        let mut worst_cosine = 1.0f32;
        let mut worst_rel = 0.0f32;
        let mut magnitude = 0.0f32;
        for (i, &row) in image.vision_tower.rows.iter().enumerate() {
            worst = worst.max(max_abs_diff(&rows[row], &expected[i]));
            worst_cosine = worst_cosine.min(cosine(&rows[row], &expected[i]));
            for (a, b) in rows[row].iter().zip(&expected[i]) {
                magnitude = magnitude.max(b.abs());
                worst_rel = worst_rel.max((a - b).abs() / b.abs().max(1e-3));
            }
        }
        eprintln!(
            "{} vision_tower: {} rows, max abs diff = {worst:.3e}, max |ref| = {magnitude:.1}, \
             max rel diff = {worst_rel:.3e}, min row cosine = {worst_cosine:.8}",
            image.name,
            rows.len()
        );
        assert!(
            worst_cosine > 0.9999,
            "{} vision_tower cosine {worst_cosine}",
            image.name
        );
        assert!(
            worst <= 5e-5 * magnitude.max(1.0),
            "{} vision_tower differs by {worst} over magnitude {magnitude}",
            image.name
        );

        // Soft tokens: the projected features.
        let soft = tower.embed_vision(&output.hidden).unwrap();
        let soft_rows: Vec<Vec<f32>> = soft.to_vec2().unwrap();
        assert_eq!(
            soft_rows.len(),
            image.soft_tokens_shape[0],
            "{}",
            image.name
        );
        let expected = image.soft_tokens.to_rows();
        let mut worst = 0.0f32;
        let mut worst_cosine = 1.0f32;
        let mut magnitude = 0.0f32;
        for (i, &row) in image.soft_tokens.rows.iter().enumerate() {
            worst = worst.max(max_abs_diff(&soft_rows[row], &expected[i]));
            worst_cosine = worst_cosine.min(cosine(&soft_rows[row], &expected[i]));
            for value in &expected[i] {
                magnitude = magnitude.max(value.abs());
            }
        }
        eprintln!(
            "{} soft_tokens: max abs diff = {worst:.3e}, max |ref| = {magnitude:.1}, \
             min row cosine = {worst_cosine:.8}",
            image.name
        );
        assert!(
            worst_cosine > 0.9999,
            "{} soft_tokens cosine {worst_cosine}",
            image.name
        );
        assert!(
            worst <= 1e-3,
            "{} soft_tokens differ by {worst}",
            image.name
        );
    }
}
