//! Video -> frames, via an `ffmpeg` subprocess.
//!
//! The engine does not link a decoder: it writes the encoded bytes to a
//! temporary file, asks `ffprobe` for the frame size, then runs `ffmpeg` at a
//! fixed capture rate (`-vf fps=<n>`) and reads raw `rgb24` frames from stdout.
//! `ffmpeg` is looked up on `PATH` unless `INL_FFMPEG` names a binary.

use crate::media::RgbImage;
use inl_core::{InlError, Result};
use std::path::Path;
use std::process::{Command, Stdio};

/// Default capture rate, in frames per second.
pub const DEFAULT_FPS: f64 = 1.0;

/// The `ffmpeg` binary to invoke: `INL_FFMPEG`, else `ffmpeg`.
pub fn ffmpeg_binary() -> String {
    std::env::var("INL_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string())
}

fn ffprobe_binary() -> String {
    std::env::var("INL_FFMPEG")
        .ok()
        .and_then(|path| {
            let probe = Path::new(&path).with_file_name("ffprobe");
            probe.to_str().map(str::to_string)
        })
        .unwrap_or_else(|| "ffprobe".to_string())
}

pub(crate) fn ensure_binary(name: &str) -> Result<()> {
    match Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(_) => Ok(()),
        Err(error) => Err(InlError::InvalidInput(format!(
            "video decoding needs `{name}` on PATH (set INL_FFMPEG to override): {error}"
        ))),
    }
}

/// Decode `bytes` into RGB frames captured at `fps`, at most `max_frames`.
pub fn extract_frames(bytes: &[u8], fps: f64, max_frames: usize) -> Result<Vec<RgbImage>> {
    if max_frames == 0 {
        return Err(InlError::InvalidInput(
            "the video frame budget is zero".to_string(),
        ));
    }
    let ffmpeg = ffmpeg_binary();
    let ffprobe = ffprobe_binary();
    ensure_binary(&ffmpeg)?;
    ensure_binary(&ffprobe)?;

    let dir = tempfile::tempdir().map_err(InlError::Io)?;
    let input = dir.path().join("input");
    std::fs::write(&input, bytes)?;

    let (width, height) = probe_dimensions(&ffprobe, &input)?;

    let output = Command::new(&ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-vf", &format!("fps={fps}")])
        .args(["-frames:v", &max_frames.to_string()])
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| {
            InlError::InvalidInput(format!(
                "could not run `{ffmpeg}` on the video bytes: {error}"
            ))
        })?;
    if !output.status.success() {
        return Err(InlError::InvalidInput(format!(
            "`{ffmpeg}` failed on the video bytes: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let frame_len = width as usize * height as usize * 3;
    if frame_len == 0 {
        return Err(InlError::InvalidInput(
            "the video has a zero-sized frame".to_string(),
        ));
    }
    let data = output.stdout;
    let mut frames = Vec::with_capacity(data.len() / frame_len);
    for chunk in data.chunks_exact(frame_len) {
        frames.push(RgbImage::new(width, height, chunk.to_vec())?);
    }
    Ok(frames)
}

fn probe_dimensions(ffprobe: &str, input: &Path) -> Result<(u32, u32)> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(input)
        .output()
        .map_err(|error| InlError::InvalidInput(format!("could not run `{ffprobe}`: {error}")))?;
    if !output.status.success() {
        return Err(InlError::InvalidInput(format!(
            "`{ffprobe}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next().unwrap_or("");
    let mut parts = first.split(',');
    let width: u32 = parts
        .next()
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| InlError::InvalidInput(format!("no video dimensions in {first:?}")))?;
    let height: u32 = parts
        .next()
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| InlError::InvalidInput(format!("no video dimensions in {first:?}")))?;
    Ok((width, height))
}
