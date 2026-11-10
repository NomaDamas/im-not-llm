//! Video -> frames, via an `ffmpeg` subprocess.
//!
//! The engine does not link a decoder: it writes the encoded bytes to a
//! temporary file, asks `ffprobe` for the stream's size, frame count, frame
//! rate and duration, picks frame indices the way the reference
//! `EmbeddingGemma2VideoProcessor.sample_frames` does ([`sample_indices`]),
//! then has `ffmpeg` emit exactly those frames as raw `rgb24`. The YUV -> RGB
//! conversion runs with swscale's bit-exact flags so every `ffmpeg` build
//! (7.x on Linux, 9.x on macOS, Windows nightlies) yields identical pixels.
//! `ffmpeg` is looked up on `PATH` unless `INL_FFMPEG` names a binary.

use crate::media::RgbImage;
use inl_core::{InlError, Result};
use std::path::Path;
use std::process::{Command, Stdio};

/// Default capture rate, in frames per second (the checkpoint's
/// `video_processor.fps`).
pub const DEFAULT_FPS: f64 = 1.0;

/// Default cap on frames per video (the checkpoint's
/// `video_processor.max_frames`); longer videos are resampled uniformly.
pub const DEFAULT_MAX_FRAMES: usize = 32;

/// Deterministic YUV -> RGB conversion, identical across `ffmpeg` builds.
const SCALE_FLAGS: &str = "scale=flags=bitexact+accurate_rnd+full_chroma_int";

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

/// What `ffprobe` reports about a video stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    /// Frames in the stream.
    pub frames: usize,
    /// Native (average) frame rate.
    pub fps: f64,
    /// Seconds.
    pub duration: f64,
}

/// The frame indices the reference processor samples: `fps` frames per
/// second of `duration` (at least one), each the native frame at or before
/// that time, then, past `max_frames`, a uniform `linspace` over them.
pub fn sample_indices(info: &VideoInfo, fps: f64, max_frames: usize) -> Vec<usize> {
    let last = info.frames.saturating_sub(1);
    let step = info.fps / fps;
    let count = ((info.duration * fps) as usize).max(1);
    let indices: Vec<usize> = (0..count)
        .map(|i| ((i as f64 * step) as usize).min(last))
        .collect();
    if indices.len() <= max_frames {
        return indices;
    }
    // `np.linspace(0, n - 1, max_frames, dtype=int)`: evenly spaced, truncated.
    let span = (indices.len() - 1) as f64;
    let denominator = (max_frames - 1).max(1) as f64;
    (0..max_frames)
        .map(|k| {
            let position = if max_frames == 1 {
                0.0
            } else {
                k as f64 * span / denominator
            };
            indices[position as usize]
        })
        .collect()
}

/// Decode `bytes` into the RGB frames the reference processor would sample at
/// `fps`, at most `max_frames` of them.
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

    let info = probe(&ffprobe, &input)?;
    let frame_len = info.width as usize * info.height as usize * 3;
    if frame_len == 0 || info.frames == 0 {
        return Err(InlError::InvalidInput(
            "the video has no frames or a zero-sized frame".to_string(),
        ));
    }
    let indices = sample_indices(&info, fps, max_frames);
    // `select` passes each native frame once; a frame sampled twice (a target
    // rate above the native one) is decoded once and repeated below.
    let mut unique = indices.clone();
    unique.dedup();
    let select = unique
        .iter()
        .map(|index| format!("eq(n,{index})"))
        .collect::<Vec<_>>()
        .join("+");

    let output = Command::new(&ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-vf", &format!("select='{select}',{SCALE_FLAGS}")])
        .args(["-fps_mode", "passthrough"])
        .args(["-frames:v", &unique.len().to_string()])
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

    let decoded: Vec<&[u8]> = output.stdout.chunks_exact(frame_len).collect();
    if decoded.len() != unique.len() {
        return Err(InlError::InvalidInput(format!(
            "`{ffmpeg}` decoded {} of the {} sampled frames; the stream reports {} frames",
            decoded.len(),
            unique.len(),
            info.frames
        )));
    }
    let mut slot = 0;
    indices
        .iter()
        .enumerate()
        .map(|(position, index)| {
            if position > 0 && indices[position - 1] != *index {
                slot += 1;
            }
            RgbImage::new(info.width, info.height, decoded[slot].to_vec())
        })
        .collect()
}

/// Probe the first video stream. The frame count is the demuxed packet count
/// (containers such as WebM do not record `nb_frames`); the duration is the
/// stream's, else the container's, else `frames / fps`.
pub fn probe(ffprobe: &str, input: &Path) -> Result<VideoInfo> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets,duration:format=duration",
            "-of",
            "default=noprint_wrappers=1",
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
    parse_probe(&String::from_utf8_lossy(&output.stdout))
}

fn parse_probe(text: &str) -> Result<VideoInfo> {
    // `format=duration` prints after the stream's own `duration`, under the
    // same key, so the first value of each key wins.
    let field = |key: &str| {
        text.lines()
            .filter_map(|line| line.trim().split_once('='))
            .find(|(name, value)| *name == key && *value != "N/A")
            .map(|(_, value)| value)
    };
    let missing = || InlError::InvalidInput(format!("ffprobe found no video stream in {text:?}"));
    let width: u32 = field("width")
        .and_then(|v| v.parse().ok())
        .ok_or_else(missing)?;
    let height: u32 = field("height")
        .and_then(|v| v.parse().ok())
        .ok_or_else(missing)?;
    let frames: usize = field("nb_read_packets")
        .and_then(|v| v.parse().ok())
        .ok_or_else(missing)?;
    let fps = ["avg_frame_rate", "r_frame_rate"]
        .into_iter()
        .filter_map(|key| field(key).and_then(rational))
        .find(|rate| *rate > 0.0)
        .ok_or_else(missing)?;
    let duration = field("duration")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|seconds| *seconds > 0.0)
        .unwrap_or(frames as f64 / fps);
    Ok(VideoInfo {
        width,
        height,
        frames,
        fps,
        duration,
    })
}

/// `30000/1001` -> 29.97; a bare number parses as itself.
fn rational(text: &str) -> Option<f64> {
    match text.split_once('/') {
        Some((numerator, denominator)) => {
            let denominator: f64 = denominator.parse().ok()?;
            (denominator != 0.0).then_some(numerator.parse::<f64>().ok()? / denominator)
        }
        None => text.parse().ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(frames: usize, fps: f64, duration: f64) -> VideoInfo {
        VideoInfo {
            width: 16,
            height: 16,
            frames,
            fps,
            duration,
        }
    }

    // Expected values are the reference `sample_frames` on the same metadata.
    #[test]
    fn indices_follow_the_reference_sampler() {
        // 3 s at 10 fps, sampled at 1 fps: frames 0, 10, 20.
        assert_eq!(
            sample_indices(&info(30, 10.0, 3.0), 1.0, 32),
            vec![0, 10, 20]
        );
        // A fractional duration keeps whole seconds only.
        assert_eq!(
            sample_indices(&info(37, 10.0, 3.7), 1.0, 32),
            vec![0, 10, 20]
        );
        // Sampling faster than the native rate repeats frames, clamped to the last.
        assert_eq!(
            sample_indices(&info(3, 1.0, 3.0), 2.0, 32),
            vec![0, 0, 1, 1, 2, 2]
        );
        // A clip shorter than one sample still gives one frame.
        assert_eq!(sample_indices(&info(4, 10.0, 0.4), 1.0, 32), vec![0]);
        // NTSC rates step by 29.97 frames.
        assert_eq!(
            sample_indices(&info(150, 30000.0 / 1001.0, 5.005), 1.0, 32),
            vec![0, 29, 59, 89, 119]
        );
    }

    #[test]
    fn long_videos_are_resampled_uniformly_to_the_frame_cap() {
        // 40 s sampled at 1 fps is 40 frames; np.linspace(0, 39, 32, dtype=int).
        let indices = sample_indices(&info(200, 5.0, 40.0), 1.0, 32);
        let picks: Vec<usize> = indices.iter().map(|index| index / 5).collect();
        assert_eq!(
            picks,
            vec![
                0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27,
                28, 30, 31, 32, 33, 35, 36, 37, 39
            ]
        );
        assert_eq!(sample_indices(&info(200, 5.0, 40.0), 1.0, 1), vec![0]);
    }

    #[test]
    fn probe_output_parses_with_fallbacks() {
        let mkv = "width=160\nheight=120\nr_frame_rate=25/1\navg_frame_rate=0/0\n\
                   nb_read_packets=50\nduration=N/A\nduration=2.000000\n";
        assert_eq!(
            parse_probe(mkv).unwrap(),
            info_sized(160, 120, 50, 25.0, 2.0)
        );
        let bare = "width=8\nheight=8\navg_frame_rate=10/1\nnb_read_packets=7\n";
        assert_eq!(parse_probe(bare).unwrap(), info_sized(8, 8, 7, 10.0, 0.7));
        assert!(parse_probe("").is_err());
    }

    fn info_sized(width: u32, height: u32, frames: usize, fps: f64, duration: f64) -> VideoInfo {
        VideoInfo {
            width,
            height,
            frames,
            fps,
            duration,
        }
    }
}
