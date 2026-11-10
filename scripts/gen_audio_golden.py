"""Reference outputs of google/embeddinggemma-2 for the audio path.

Run: HF_HOME=... python scripts/gen_audio_golden.py OUT_DIR
Writes OUT_DIR/{audio_clip.wav,golden_audio.json}.

The clip is 3.3 s of 16 kHz mono float32 (a chirp, a tone burst and noise), so
the Rust engine decodes exactly the samples the reference sees. For it the JSON
stores the feature extractor's log-mel frames and mask, the audio tower output
(before `embed_audio`), every-STRIDE-th row as float32 base64, and the final
embeddings (fp32, CPU, mean pooling + L2 norm) of the audio alone and of the
audio interleaved with text. It also records the placeholder count the
processor gives a 31 s clip (the extractor truncates at 30 s).
"""
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

RATE = 16000
STRIDE = 7


def b64(t):
    return base64.b64encode(np.ascontiguousarray(np.asarray(t, dtype=np.float32)).tobytes()).decode()


def rows(t):
    t = np.asarray(t, dtype=np.float32)
    t = t.reshape(-1, t.shape[-1])
    idx = sorted(set(list(range(0, t.shape[0], STRIDE)) + [t.shape[0] - 1]))
    return {"rows": idx, "cols": int(t.shape[-1]), "data_b64": b64(t[idx])}


def synth(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    chirp = 0.4 * np.sin(2 * np.pi * (200 + 900 * t) * t)
    burst = 0.3 * np.sin(2 * np.pi * 1500 * t) * ((t > 1.0) & (t < 1.6))
    noise = 0.05 * rng.standard_normal(t.shape)
    return (chirp + burst + noise).astype(np.float32)


clip = os.path.join(out, "audio_clip.wav")
wave = synth(3.3, 7)
sf.write(clip, wave, RATE, subtype="FLOAT")
wave, rate = sf.read(clip, dtype="float32")
assert rate == RATE

model_id = "google/embeddinggemma-2"
hf = AutoModel.from_pretrained(model_id, dtype=torch.float32)
hf.eval()
proc = AutoProcessor.from_pretrained(model_id)


def embed(inputs):
    with torch.no_grad():
        outputs = hf(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1)[0].tolist()


audio_only = proc(audio=[wave], sampling_rate=RATE, return_tensors="pt")
features = audio_only["input_features"][0]
features_mask = audio_only["input_features_mask"][0]
with torch.no_grad():
    tower = hf.audio_tower(audio_only["input_features"], audio_only["input_features_mask"], return_dict=True)
tower_out = tower.last_hidden_state[0]
tower_mask = tower.attention_mask[0]
audio_token_id = hf.config.audio_token_id
audio_tokens = int((audio_only["input_ids"] == audio_token_id).sum())
print("features", tuple(features.shape), "valid", int(features_mask.sum()), "tokens", audio_tokens)

text = f"Speech sample: {proc.audio_token} and some text after it."
interleaved = proc(text=[text], audio=[wave], sampling_rate=RATE, return_tensors="pt")

long_wave = synth(31.0, 8)
long_inputs = proc(audio=[long_wave], sampling_rate=RATE, return_tensors="pt")
long_tokens = int((long_inputs["input_ids"] == audio_token_id).sum())
print("31 s clip ->", long_tokens, "audio tokens")

golden = {
    "audio": "audio_clip.wav",
    "samples": int(wave.shape[0]),
    "features": rows(features.numpy()),
    "feature_frames": int(features.shape[0]),
    "feature_valid": int(features_mask.sum()),
    "tower": rows(tower_out.numpy()),
    "tower_frames": int(tower_out.shape[0]),
    "tower_valid": int(tower_mask.sum()),
    "audio_tokens": audio_tokens,
    "embedding": embed(audio_only),
    "interleaved_text": text,
    "interleaved_embedding": embed(interleaved),
    "long_seconds": 31.0,
    "long_tokens": long_tokens,
}
json.dump(golden, open(os.path.join(out, "golden_audio.json"), "w"))
print("wrote", os.path.join(out, "golden_audio.json"))
