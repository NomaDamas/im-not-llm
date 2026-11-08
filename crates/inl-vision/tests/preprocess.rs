//! Stage 1 golden: pure-Rust decode + resize + patchify matches the reference
//! `pixel_values` rows.

mod common;

use common::*;
use inl_vision::preprocess;

#[test]
fn resized_patches_match_the_reference() {
    let golden = golden();
    for image in &golden.images {
        let bytes = image_bytes(&image.name);
        let pre = preprocess::preprocess(&bytes, 280).expect("preprocess");
        assert_eq!(pre.max_patches, image.padded_patches, "{}", image.name);
        assert_eq!(pre.real_patches, image.real_patches, "{}", image.name);
        assert_eq!(pre.num_soft_tokens, image.num_soft_tokens, "{}", image.name);
        assert_eq!(
            pre.resized_width * pre.resized_height,
            image.real_patches * 256
        );

        // Spot-check position ids: the stored head and tail of the real rows.
        let mut positions: Vec<Vec<i64>> = Vec::new();
        for patch in 0..pre.real_patches {
            positions.push(vec![
                pre.position_ids[patch * 2],
                pre.position_ids[patch * 2 + 1],
            ]);
        }
        assert_eq!(
            &positions[..4],
            image.position_ids_head.as_slice(),
            "{}",
            image.name
        );
        let tail_start = pre.real_patches - 3;
        assert_eq!(
            &positions[tail_start..],
            image.position_ids_tail.as_slice(),
            "{}",
            image.name
        );

        // Compare every stored row of the patch tensor.
        let expected = image.pixel_values.to_rows();
        let mut worst = 0.0f32;
        for (i, &row) in image.pixel_values.rows.iter().enumerate() {
            let ours = &pre.pixel_values[row * 768..row * 768 + 768];
            let diff = max_abs_diff(ours, &expected[i]);
            worst = worst.max(diff);
        }
        eprintln!(
            "{} resized {}x{} ({} patches): max |pixel_values diff| = {worst:.3e}",
            image.name, pre.resized_width, pre.resized_height, pre.real_patches
        );
        assert!(
            worst <= 1e-6,
            "{} pixel_values differ by {worst} (expected exact uint8 match)",
            image.name
        );
    }
}
