"""Reference vectors of the REAL perplexity-ai/pplx-embed-v2-late-0.6b checkpoint
(Sentence-Transformers MultiVectorEncoder, fp32, CPU) for
`crates/inl-late/tests/late_live.rs`.

    pip install "sentence-transformers>=6" "transformers>=5.4" torch
    python scripts/gen_late_golden.py
"""

import json
import pathlib

import torch
from sentence_transformers import MultiVectorEncoder

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/inl-late/tests/fixtures/pplx_embed_v2_late/st_f32_golden.json"
)

model = MultiVectorEncoder(
    "perplexity-ai/pplx-embed-v2-late-0.6b",
    device="cpu",
    model_kwargs={"dtype": torch.float32},
)
tower = model[0]
cases = {
    "query": ["What is the capital of France?", "서울의 날씨는 어때?"],
    "document": [
        "Paris is the capital and largest city of France.",
        "Hello, world! (a test) -- with punctuation.",
        "Paris is the capital and largest city of France. " * 7,
        "Paris is the capital and largest city of France. " * 26,
    ],
}
golden = []
for task, texts in cases.items():
    prompt = model.prompts.get(task)
    encode = model.encode_query if task == "query" else model.encode_document
    for text, vectors in zip(texts, encode(texts, batch_size=1)):
        ids = tower.preprocess([text], prompt=prompt, task=task)["input_ids"][0].tolist()
        golden.append(
            {
                "task": task,
                "prompt": prompt,
                "text": text,
                "input_ids": ids,
                "vectors": torch.as_tensor(vectors).tolist(),
            }
        )
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(golden))
print("wrote", OUT, len(golden))
