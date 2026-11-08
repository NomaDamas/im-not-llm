"""Reference output of google/embeddinggemma-2 for the video path.

Run: HF_HOME=... python scripts/gen_video_golden.py OUT_DIR
Writes OUT_DIR/{video_clip.mp4,golden_video.json}.

The clip is a 3-second `testsrc` clip at 160x120. Frames are captured at 1 fps
with the same `ffmpeg -vf fps=1 -f rawvideo -pix_fmt rgb24` command the Rust
engine runs, so both sides see identical frames. The reference embedding is the
SentenceTransformers mean-pool + L2 norm of the multimodal model run on the
processor's video inputs (max_soft_tokens 140, as the engine's default).
"""
import json
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

out = sys.argv[1]
import os

os.makedirs(out, exist_ok=True)

clip = os.path.join(out, "video_clip.mp4")
if not os.path.exists(clip):
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", clip,
        ],
        check=True,
    )

# Capture frames the way the Rust engine does.
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
     "stream=width,height", "-of", "csv=p=0", clip],
    check=True, capture_output=True, text=True,
)
width, height = (int(x) for x in probe.stdout.strip().split(",")[0:2])
raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", clip, "-vf", "fps=1", "-frames:v", "57",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    check=True, capture_output=True,
).stdout
frame_len = width * height * 3
frames = [
    Image.fromarray(np.frombuffer(raw[i : i + frame_len], dtype=np.uint8).reshape(height, width, 3))
    for i in range(0, len(raw) - frame_len + 1, frame_len)
]
print("captured", len(frames), "frames", width, "x", height)

model_id = "google/embeddinggemma-2"
hf = AutoModel.from_pretrained(model_id, dtype=torch.float32)
hf.eval()
proc = AutoProcessor.from_pretrained(model_id)

inputs = proc(videos=[frames], return_tensors="pt", do_sample_frames=False, max_soft_tokens=140)
print("keys", list(inputs.keys()))
with torch.no_grad():
    outputs = hf(**inputs)
mask = inputs["attention_mask"].unsqueeze(-1).float()
pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
embedding = torch.nn.functional.normalize(pooled, dim=-1)[0].tolist()

golden = {
    "video": "video_clip.mp4",
    "frames": len(frames),
    "width": width,
    "height": height,
    "max_soft_tokens": 140,
    "embedding": embedding,
}
json.dump(golden, open(os.path.join(out, "golden_video.json"), "w"))
print("wrote", os.path.join(out, "golden_video.json"))
