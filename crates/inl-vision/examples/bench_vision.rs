//! Throughput of the EmbeddingGemma 2 vision path.
//!
//! `cargo run -p inl-vision --release --example bench_vision -- \
//!      --device metal --dtype f32 --n 16 --size 320x240 --budget 280`
//!
//! Prints one JSON line with the device, its effective dtype, the number of
//! images, total seconds and images per second.

use inl_core::engine::{MediaBytes, MultimodalEngine, MultimodalInput};
use inl_vision::engine::{MultimodalEmbedder, MultimodalOptions};
use inl_vision::DeviceSpec;
use std::io::Cursor;
use std::time::Instant;

struct Args {
    device: String,
    dtype: Option<String>,
    n: usize,
    width: u32,
    height: u32,
    budget: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        device: "auto".to_string(),
        dtype: None,
        n: 8,
        width: 320,
        height: 240,
        budget: 280,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().expect("flag needs a value");
        match flag.as_str() {
            "--device" => args.device = value(),
            "--dtype" => args.dtype = Some(value()),
            "--n" => args.n = value().parse().expect("--n"),
            "--size" => {
                let size = value();
                let (w, h) = size.split_once('x').expect("--size WxH");
                args.width = w.parse().expect("width");
                args.height = h.parse().expect("height");
            }
            "--budget" => args.budget = value().parse().expect("--budget"),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

/// A deterministic PNG of `width x height`.
fn synth_png(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut image = image::RgbImage::new(width, height);
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (state >> 33) as u8;
        *pixel = image::Rgb([
            ((x * 255 / width.max(1)) as u8).wrapping_add(noise),
            ((y * 255 / height.max(1)) as u8).wrapping_add(noise),
            noise,
        ]);
    }
    let mut out = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let device: DeviceSpec = args.device.parse().expect("--device");
    let dtype = match args.dtype.as_deref() {
        None | Some("auto") => None,
        Some(name) => Some(name.parse().expect("--dtype")),
    };
    let options = MultimodalOptions::default()
        .with_device(device)
        .with_dtype(dtype)
        .with_max_soft_tokens(args.budget);
    let embedder = MultimodalEmbedder::new(options);
    let info = embedder.load().await.expect("load");

    let inputs: Vec<MultimodalInput> = (0..args.n)
        .map(|i| MultimodalInput {
            images: vec![MediaBytes {
                bytes: synth_png(args.width, args.height, i as u64 + 1),
                mime: Some("image/png".to_string()),
            }],
            ..Default::default()
        })
        .collect();

    // Warm up (kernel selection, allocation).
    embedder.embed(inputs[..1].to_vec()).await.expect("warmup");

    let started = Instant::now();
    let embeddings = embedder.embed(inputs).await.expect("embed");
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(embeddings.len(), args.n);

    println!(
        "{{\"device\":\"{}\",\"dtype\":\"{}\",\"images\":{},\"seconds\":{:.4},\"images_per_second\":{:.3}}}",
        info.device,
        info.dtype,
        args.n,
        seconds,
        args.n as f64 / seconds
    );
}
