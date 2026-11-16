"""Regenerates crates/inl-text/tests/fixtures/qwen3embedding/golden_real.json (dev-only).

Reference vectors of the REAL Qwen/Qwen3-Embedding-0.6B (Sentence-Transformers,
fp32, CPU, last-token pooling, truncated at 2048 tokens like the engines'
default) for crates/inl-ane/tests/ane_live.rs.

    uv venv .venv && uv pip install --python .venv/bin/python transformers torch sentence-transformers
    .venv/bin/python scripts/gen_qwen3_golden.py
"""
import json, torch, numpy as np
from sentence_transformers import SentenceTransformer

REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", revision=REVISION,
                            model_kwargs={"dtype": torch.float32}, device="cpu")
model.eval()
model.max_seq_length = 2048
q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
ko = "환불 정책에 따르면 구매 증빙 인증서를 제출한 고객은 14일 이내에 전액 환불을 받을 수 있습니다. 해외 출장비는 출장 종료 후 5영업일 안에 정산해야 합니다. "
en = "The northern lights are caused by charged particles from the sun colliding with gases in the upper atmosphere. "
texts = [
  q + "환불 받으려면 어떤 인증서가 필요해?",
  q + "What causes the northern lights?",
  "구매 증빙 인증서를 제출하면 환불이 가능합니다.",
  "The northern lights are caused by charged particles from the sun.",
  "fn main() { println!(\"hello 🌍\"); } // ERR-48213",
  ko * 40 + en * 20,              # long: truncated at 2048
  en * 60,                        # long, English
  "",
]
ids = [model.tokenizer(t, truncation=True, max_length=2048)["input_ids"] for t in texts]
print([len(x) for x in ids], ids[0][-3:], ids[-1])
emb = model.encode(texts, batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
print(emb.shape, np.linalg.norm(emb, axis=1))
out = {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": REVISION, "texts": texts, "ids": ids,
       "dim": int(emb.shape[1]), "emb": emb.round(7).tolist()}
json.dump(out, open("crates/inl-text/tests/fixtures/qwen3embedding/golden_real.json", "w"))
