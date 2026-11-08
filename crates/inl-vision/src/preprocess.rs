//! Image decoding and preprocessing for EmbeddingGemma 2's vision tower.
//!
//! Matches `Gemma4ImageProcessor` (transformers, Apache-2.0) exactly:
//!
//! * decode PNG/JPEG/WebP to RGB8;
//! * aspect-ratio-preserving resize, the largest size that (a) patchifies to at
//!   most `max_patches = max_soft_tokens * 3^2` patches and (b) has both sides
//!   divisible by `3 * 16`, resampled with torchvision's antialiased bicubic;
//! * rescale by `1/255` (no normalize);
//! * patchify to `(num_patches, 3 * 16 * 16)` in patch-major order;
//! * positions `(x, y)` and padding to `max_patches` with `(-1, -1)`.
//!
//! On a CPU where torchvision keeps the uint8 dtype (x86 AVX2, aarch64), the
//! resize runs PyTorch's integer kernel: the Keys cubic (`a = -0.5`) taps,
//! computed in float64, normalized, converted to int16 at a fixed
//! `COEF_PRECISION`, then each separable pass accumulates in int32 and rounds
//! by an arithmetic shift, clamping to `[0, 255]` per pass. This replays that
//! kernel bit-for-bit rather than the float32 path (which differs by up to 16).

use crate::media::RgbImage;
use inl_core::{InlError, Result};

/// Spatial side of one patch, in pixels.
pub const PATCH_SIZE: usize = 16;
/// Side of the square pooling window (patches per soft token side).
pub const POOLING_KERNEL: usize = 3;
/// The soft-token budgets `EmbeddingGemma 2` accepts, from the model card.
pub const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// One image, decoded, resized, patchified and padded to the token budget.
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// Rows in `pixel_values` (the padded patch count).
    pub max_patches: usize,
    /// Patches holding real image content.
    pub real_patches: usize,
    /// Soft tokens this image produces after `3 x 3` pooling.
    pub num_soft_tokens: usize,
    /// Height of the resized image, in pixels.
    pub resized_height: usize,
    /// Width of the resized image, in pixels.
    pub resized_width: usize,
    /// `[max_patches, 3 * PATCH_SIZE * PATCH_SIZE]`, patch-major, padded rows 0.
    pub pixel_values: Vec<f32>,
    /// `[max_patches, 2]` `(x, y)`; padded rows are `(-1, -1)`.
    pub position_ids: Vec<i64>,
}

/// Decode encoded image bytes (PNG / JPEG / WebP) to tightly packed RGB8.
pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    let image = image::load_from_memory(bytes).map_err(|error| {
        InlError::InvalidInput(format!("could not decode image bytes: {error}"))
    })?;
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    RgbImage::new(width, height, rgb.into_raw())
}

/// Decode `bytes` and produce the padded patch tensor for `max_soft_tokens`.
pub fn preprocess(bytes: &[u8], max_soft_tokens: usize) -> Result<PreprocessedImage> {
    let image = decode(bytes)?;
    preprocess_rgb(&image, max_soft_tokens)
}

/// Patchify an already-decoded image for `max_soft_tokens`.
pub fn preprocess_rgb(image: &RgbImage, max_soft_tokens: usize) -> Result<PreprocessedImage> {
    if !SUPPORTED_SOFT_TOKENS.contains(&max_soft_tokens) {
        return Err(InlError::Config(format!(
            "max_soft_tokens {max_soft_tokens} is not one of {SUPPORTED_SOFT_TOKENS:?}"
        )));
    }
    let max_patches = max_soft_tokens * POOLING_KERNEL * POOLING_KERNEL;
    let (target_height, target_width) = aspect_ratio_preserving_size(
        image.height as usize,
        image.width as usize,
        PATCH_SIZE,
        max_patches,
        POOLING_KERNEL,
    )?;

    let resized = resize_rgb(image, target_width, target_height);

    let patch_w = target_width / PATCH_SIZE;
    let patch_h = target_height / PATCH_SIZE;
    let real_patches = patch_w * patch_h;

    let patch_pixels = 3 * PATCH_SIZE * PATCH_SIZE;
    let mut pixel_values = vec![0f32; max_patches * patch_pixels];
    for patch_row in 0..patch_h {
        for patch_col in 0..patch_w {
            let patch = patch_row * patch_w + patch_col;
            let mut cursor = patch * patch_pixels;
            for in_row in 0..PATCH_SIZE {
                let y = patch_row * PATCH_SIZE + in_row;
                for in_col in 0..PATCH_SIZE {
                    let x = patch_col * PATCH_SIZE + in_col;
                    let index = (y * target_width + x) * 3;
                    pixel_values[cursor] = resized[index] as f32 / 255.0;
                    pixel_values[cursor + 1] = resized[index + 1] as f32 / 255.0;
                    pixel_values[cursor + 2] = resized[index + 2] as f32 / 255.0;
                    cursor += 3;
                }
            }
        }
    }

    let mut position_ids = vec![-1i64; max_patches * 2];
    for patch_row in 0..patch_h {
        for patch_col in 0..patch_w {
            let patch = patch_row * patch_w + patch_col;
            position_ids[patch * 2] = patch_col as i64;
            position_ids[patch * 2 + 1] = patch_row as i64;
        }
    }

    Ok(PreprocessedImage {
        max_patches,
        real_patches,
        num_soft_tokens: real_patches / (POOLING_KERNEL * POOLING_KERNEL),
        resized_height: target_height,
        resized_width: target_width,
        pixel_values,
        position_ids,
    })
}

/// The largest `(height, width)` that patchifies to at most `max_patches`
/// patches and has both sides divisible by `pooling_kernel_size * patch_size`,
/// with the source aspect ratio. From `get_aspect_ratio_preserving_size`.
fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let mut target_height =
        ((factor * height as f64) / side_mult as f64).floor() as usize * side_mult;
    let mut target_width =
        ((factor * width as f64) / side_mult as f64).floor() as usize * side_mult;

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 && target_width == 0 {
        return Err(InlError::InvalidInput(
            "image is too small to resize to a non-empty patch grid".to_string(),
        ));
    }
    if target_height == 0 {
        target_height = side_mult;
        target_width =
            ((width as f64 / height as f64).floor() as usize * side_mult).min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height =
            ((height as f64 / width as f64).floor() as usize * side_mult).min(max_side_length);
    }

    if target_height * target_width > max_patches * patch_size * patch_size {
        return Err(InlError::InvalidInput(format!(
            "resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds {max_patches} patches"
        )));
    }
    if target_width == 0 || target_height == 0 {
        return Err(InlError::InvalidInput(
            "image is too small to resize to a non-empty patch grid".to_string(),
        ));
    }
    Ok((target_height, target_width))
}

/// Integer filter taps for one axis, PyTorch's `_compute_index_ranges_int16_weights`.
struct Int16Weights {
    precision: u32,
    /// `taps[i]` is `(input_index, weight)` for output index `i`.
    taps: Vec<Vec<(usize, i16)>>,
}

/// Replay the antialiased-bicubic uint8 kernel on an RGB image: separable
/// horizontal then vertical passes, each in int32 and clamped to uint8.
fn resize_rgb(image: &RgbImage, target_width: usize, target_height: usize) -> Vec<u8> {
    let src_w = image.width as usize;
    let src_h = image.height as usize;

    let mut planes: [Vec<u8>; 3] = std::array::from_fn(|channel| {
        (0..src_w * src_h)
            .map(|index| image.data[index * 3 + channel])
            .collect()
    });

    let mut width_cur = src_w;
    if target_width != src_w {
        let weights = int16_weights(src_w, target_width);
        for plane in planes.iter_mut() {
            *plane = apply_axis(plane, src_h, width_cur, true, target_width, &weights);
        }
        width_cur = target_width;
    }
    if target_height != src_h {
        let weights = int16_weights(src_h, target_height);
        for plane in planes.iter_mut() {
            *plane = apply_axis(plane, src_h, width_cur, false, target_height, &weights);
        }
    }

    // Interleave back to RGB.
    let out_size = width_cur * target_height;
    let mut rgb = vec![0u8; out_size * 3];
    for (channel, plane) in planes.iter().enumerate() {
        for (index, &value) in plane.iter().enumerate() {
            rgb[index * 3 + channel] = value;
        }
    }
    rgb
}

/// Compute per-output-index integer taps exactly as
/// `_compute_index_ranges_weights<double, ...>` + the int16 rescale.
fn int16_weights(input_size: usize, output_size: usize) -> Int16Weights {
    let scale = input_size as f64 / output_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_interp_size = (support.ceil() as usize) * 2 + 1;

    let mut rows: Vec<(usize, Vec<f64>)> = Vec::with_capacity(output_size);
    let mut wt_max = 0.0f64;
    for i in 0..output_size {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(input_size as i64);
        let xsize = (xmax - xmin).clamp(0, max_interp_size as i64) as usize;

        let mut weights = vec![0.0f64; max_interp_size];
        let mut total = 0.0f64;
        for (j, slot) in weights.iter_mut().enumerate().take(xsize) {
            let w = cubic_aa_filter((j as f64 + xmin as f64 - center + 0.5) * invscale);
            *slot = w;
            total += w;
        }
        if total != 0.0 {
            for w in weights.iter_mut().take(xsize) {
                *w /= total;
                wt_max = wt_max.max(*w);
            }
        }
        rows.push((xmin as usize, weights));
    }

    // Weights precision: the smallest p with round(wt_max * 2^(p+1)) >= 2^15.
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * (1u64 << (precision + 1)) as f64) as i64;
        if next >= (1 << 15) {
            break;
        }
        precision += 1;
    }
    let factor = (1u64 << precision) as f64;

    let taps = rows
        .iter()
        .map(|(xmin, weights)| {
            weights
                .iter()
                .enumerate()
                .filter_map(|(j, w)| {
                    let value = w * factor;
                    let quantized = if value < 0.0 {
                        (-0.5 + value) as i64
                    } else {
                        (0.5 + value) as i64
                    };
                    (quantized != 0).then_some((xmin + j, quantized as i16))
                })
                .collect()
        })
        .collect();

    Int16Weights { precision, taps }
}

/// One separable pass. `palette` is the bool for horizontal (W) vs vertical (H).
fn apply_axis(
    plane: &[u8],
    src_h: usize,
    src_w: usize,
    horizontal: bool,
    out_size: usize,
    weights: &Int16Weights,
) -> Vec<u8> {
    let precision = weights.precision;
    let rounding = 1i32 << (precision - 1);
    if horizontal {
        let mut result = vec![0u8; src_h * out_size];
        for y in 0..src_h {
            let row = &plane[y * src_w..y * src_w + src_w];
            let out_row = &mut result[y * out_size..y * out_size + out_size];
            for (i, taps) in weights.taps.iter().enumerate() {
                let mut acc = rounding;
                for &(index, weight) in taps {
                    acc += row[index] as i32 * weight as i32;
                }
                out_row[i] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
        result
    } else {
        let mut result = vec![0u8; out_size * src_w];
        for (i, taps) in weights.taps.iter().enumerate() {
            for x in 0..src_w {
                let mut acc = rounding;
                for &(index, weight) in taps {
                    acc += plane[index * src_w + x] as i32 * weight as i32;
                }
                result[i * src_w + x] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
        result
    }
}

/// The Keys cubic with `a = -0.5` (PIL-compatible), PyTorch's
/// `HelperInterpCubic::aa_filter<double, true>`.
fn cubic_aa_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_matches_reference_for_the_fixtures() {
        // 320x240 -> 57x42 patches (912x672 px), 97x211 -> 33x72 (528x1152 px).
        assert_eq!(
            aspect_ratio_preserving_size(240, 320, 16, 280 * 9, 3).unwrap(),
            (672, 912)
        );
        assert_eq!(
            aspect_ratio_preserving_size(211, 97, 16, 280 * 9, 3).unwrap(),
            (1152, 528)
        );
    }

    #[test]
    fn integer_weights_sum_to_one_at_full_precision() {
        let weights = int16_weights(10, 25);
        assert!(weights.precision >= 12, "precision {}", weights.precision);
        let scale = (1i64 << weights.precision) as f64;
        for taps in &weights.taps {
            let sum: f64 = taps.iter().map(|(_, w)| *w as f64).sum();
            assert!((sum / scale - 1.0).abs() < 1e-3, "sum {}", sum / scale);
        }
    }

    #[test]
    fn cubic_filter_is_the_keys_cubic() {
        assert!((cubic_aa_filter(0.0) - 1.0).abs() < 1e-9);
        assert_eq!(cubic_aa_filter(2.0), 0.0);
    }
}
