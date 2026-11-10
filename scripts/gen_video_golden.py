"""Reference outputs of google/embeddinggemma-2 for the video path.

Run: HF_HOME=... python scripts/gen_video_golden.py OUT_DIR
Writes OUT_DIR/{video_clip.mp4,video_long.mp4,golden_video.json}.

Every native frame is decoded with the same bit-exact `ffmpeg` conversion the
Rust engine uses (`scale=flags=bitexact+accurate_rnd+full_chroma_int`), and the
reference processor does its own frame sampling from the stream's metadata
(`do_sample_frames=True` with `fps`, `max_frames=32`, `overflow_strategy`
"uniform"), so the golden checks the engine's sampler as well as the model.
Embeddings are the SentenceTransformers mean-pool + L2 norm of the model output.

Cases:
- `video`: the 3 s, 10 fps clip at the defaults (1 fps, 140 tokens/frame).
- `interleaved`: text around a `<|video|>` placeholder, same clip.
- `fps2_tokens70`: non-default `video_fps` 2 and `video_max_soft_tokens` 70.
- `long`: a 40 s clip; 40 frames at 1 fps, resampled uniformly to 32.
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
SCALE = "scale=flags=bitexact+accurate_rnd+full_chroma_int"


def make_clip(name, size, rate, seconds):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", path],
            check=True,
        )
    return path


def decode(path):
    """Every native frame, plus the metadata the reference sampler reads."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets,duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", SCALE,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    total = int(stream["nb_read_packets"])
    assert frames.shape[0] == total, (frames.shape, total)
    metadata = VideoMetadata(
        total_num_frames=total, fps=num / den, duration=float(stream["duration"]),
        width=width, height=height,
    )
    return frames, metadata


model_id = "google/embeddinggemma-2"
hf = AutoModel.from_pretrained(model_id, dtype=torch.float32)
hf.eval()
proc = AutoProcessor.from_pretrained(model_id)


def embed(frames, metadata, fps, tokens, text=None):
    kwargs = dict(
        videos=[frames], video_metadata=[metadata], do_sample_frames=True, fps=fps,
        max_frames=32, max_soft_tokens=tokens, return_tensors="pt",
    )
    if text is not None:
        kwargs["text"] = [text]
    inputs = proc(**kwargs)
    sampled = int(inputs["num_frames_per_video"][0])
    with torch.no_grad():
        outputs = hf(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return sampled, int(mask.sum()), torch.nn.functional.normalize(pooled, dim=-1)[0].tolist()


short = make_clip("video_clip.mp4", "160x120", 10, 3)
long = make_clip("video_long.mp4", "64x48", 2, 40)
short_frames, short_meta = decode(short)
long_frames, long_meta = decode(long)

cases = []
for name, video, frames, meta, fps, tokens, text in [
    ("video", "video_clip.mp4", short_frames, short_meta, 1.0, 140, None),
    ("interleaved", "video_clip.mp4", short_frames, short_meta, 1.0, 140,
     f"A test pattern: {proc.video_token} and some text after it."),
    ("fps2_tokens70", "video_clip.mp4", short_frames, short_meta, 2.0, 70, None),
    ("long", "video_long.mp4", long_frames, long_meta, 1.0, 140, None),
]:
    sampled, length, embedding = embed(frames, meta, fps, tokens, text)
    print(name, "frames", sampled, "sequence", length)
    cases.append({
        "name": name, "video": video, "text": text, "fps": fps,
        "max_soft_tokens": tokens, "frames": sampled, "sequence_length": length,
        "embedding": embedding,
    })

json.dump({"cases": cases}, open(os.path.join(out, "golden_video.json"), "w"))
print("wrote", os.path.join(out, "golden_video.json"))
