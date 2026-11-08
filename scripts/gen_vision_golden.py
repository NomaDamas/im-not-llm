"""Reference outputs of google/embeddinggemma-2 for the image path.

Run: HF_HOME=... python scripts/gen_vision_golden.py OUT_DIR
Writes OUT_DIR/{img_a.png,img_b.png,golden_vision.json}. For every image it stores
the processor's patch tensor statistics, vision tower pooled output, the projected
soft tokens, and the final sentence embedding (fp32, CPU, SentenceTransformers
mean pooling + normalize), plus the text-image interleaved embedding.
Intermediates are stored as float32 base64 so the Rust tests can compare stage by stage.
"""
import base64, json, os, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def b64(t):
    return base64.b64encode(np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes()).decode()


STRIDE = 41  # rows kept of each intermediate; the full tensors are ~10 MB per image


def rows(t):
    """Every STRIDE-th row (+ the last) with its index, as float32 base64."""
    t = t.reshape(-1, t.shape[-1])
    idx = sorted(set(list(range(0, t.shape[0], STRIDE)) + [t.shape[0] - 1]))
    return {"rows": idx, "cols": int(t.shape[-1]), "data_b64": b64(t[idx])}


def synth(path, w, h, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
    a[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, 2] = 200
    a[::7, ::5, :] = rng.integers(0, 255, size=a[::7, ::5, :].shape, dtype=np.uint8)
    Image.fromarray(a).save(path)


synth(f"{out}/img_a.png", 320, 240, 1)
synth(f"{out}/img_b.png", 97, 211, 2)  # odd size: exercises the resize rounding

model_id = "google/embeddinggemma-2"
st = SentenceTransformer(model_id, device="cpu", model_kwargs={"torch_dtype": torch.float32})
hf = st[0].model  # EmbeddingGemma2Model
proc = AutoProcessor.from_pretrained(model_id)
hf.eval()

golden = {"images": [], "text_image": None}
for name in ("img_a", "img_b"):
    img = Image.open(f"{out}/{name}.png").convert("RGB")
    feats = proc.image_processor(images=[img], return_tensors="pt")
    pv, pos = feats["pixel_values"], feats["image_position_ids"]
    real = int((pos[0] != -1).all(-1).sum())  # patches that are not padding
    with torch.no_grad():
        vt = hf.vision_tower(pixel_values=pv, pixel_position_ids=pos).last_hidden_state
        soft = hf.embed_vision(inputs_embeds=vt)
    emb = st.encode({"image": img}, normalize_embeddings=True)
    golden["images"].append(
        {
            "name": name,
            "size": list(img.size),
            "padded_patches": int(pv.shape[1]),
            "real_patches": real,
            "pixel_values_shape": [real, int(pv.shape[2])],
            "pixel_values": rows(pv[0, :real]),
            "position_ids_head": pos[0, :4].tolist(),
            "position_ids_tail": pos[0, real - 3 : real].tolist(),
            "num_soft_tokens": int(feats["num_soft_tokens_per_image"][0]),
            "vision_tower_shape": list(vt.shape),
            "vision_tower": rows(vt[0] if vt.dim() == 3 else vt),
            "soft_tokens_shape": list(soft.shape),
            "soft_tokens": rows(soft[0] if soft.dim() == 3 else soft),
            "embedding": emb.astype(np.float32).tolist(),
        }
    )

img = Image.open(f"{out}/img_a.png").convert("RGB")
text = "A gradient picture: <|image|> with a blue box."
e = st.encode({"text": text, "image": [img]}, normalize_embeddings=True)
golden["text_image"] = {"text": text, "image": "img_a", "embedding": e.astype(np.float32).tolist()}
q = st.encode("task: search result | query: a blue box on a gradient", normalize_embeddings=True)
golden["query"] = {"text": "task: search result | query: a blue box on a gradient", "embedding": q.astype(np.float32).tolist()}
json.dump(golden, open(f"{out}/golden_vision.json", "w"))
print("wrote", out)
