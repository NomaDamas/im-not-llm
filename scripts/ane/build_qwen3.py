#!/usr/bin/env python3
"""Build the Apple Neural Engine (Core ML) package of a Qwen3-Embedding checkpoint for `inl-ane`.

    python build_qwen3.py --hf-dir <Qwen/Qwen3-Embedding-0.6B snapshot> --out <package dir>

Stages (idempotent; --force redoes them):
  convert   one fp16 ML Program per function into <work>/fn: embed_S for S in BUCKETS (packed sequences,
            block-diagonal causal attention, last-token pooling)
  merge     multifunction package Qwen3EmbeddingText.mlpackage (identical weights are stored once)
  assets    embeddings.bf16 (token table, raw bf16 rows), tokenizer.json
  selftest  selftest.json + selftest.bin: one golden test per function, expectations from the fp32
            Hugging Face model (sentence-transformers equivalent: last token, L2 normalize)
  manifest  manifest.json: function signatures, host contract, token table, every file with size + sha256

CPU only. The .mlpackage is compiled on the Mac that runs it (inl-ane does that on first load).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qwen3_ane as q  # noqa: E402

PKG = "Qwen3EmbeddingText"
NAME = "qwen3-embedding-ane"
SELFTEST_TOLERANCE = {"bucket": {"min_cos": 0.999, "zero_rows_max_abs": 1e-3}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def function_name(S):
    return f"embed_{S}"


def specs(W):
    return [(function_name(S), S, [("inputs_embeds", (1, S, W.H)), ("causal_bias", (1, 1, S, S)),
                                    ("pool", (q.k_for(S), S))]) for S in q.BUCKETS]


def stage_convert(a, W):
    import coremltools as ct
    import torch
    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs(f"{a.work}/fn", exist_ok=True)
    for name, S, ins in specs(W):
        path = f"{a.work}/fn/{name}.mlpackage"
        if os.path.exists(path) and not a.force:
            continue
        t = time.time()
        mod = q.Bucket(W, S).eval()
        with torch.no_grad():
            tr = torch.jit.trace(mod, tuple(torch.zeros(*s) for _, s in ins), check_trace=False)
        ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float16) for n, s in ins],
                        outputs=[ct.TensorType(name="embedding", dtype=np.float32)], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                        compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
        tmp = f"{a.work}/fn/_tmp_{name}.mlpackage"
        shutil.rmtree(tmp, ignore_errors=True)
        ml.save(tmp)
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        print(f"[convert] {name} {time.time() - t:.1f}s", flush=True)


def stage_merge(a, W):
    import coremltools as ct
    out = f"{a.out}/{PKG}.mlpackage"
    if os.path.exists(out) and not a.force:
        return
    t = time.time()
    desc = ct.utils.MultiFunctionDescriptor()
    for name, _, _ in specs(W):
        desc.add_function(f"{a.work}/fn/{name}.mlpackage", src_function_name="main", target_function_name=name)
    desc.default_function_name = function_name(512)
    tmp = f"{a.out}/_tmp_{PKG}.mlpackage"
    shutil.rmtree(tmp, ignore_errors=True)
    ct.utils.save_multifunction(desc, tmp)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print(f"[merge] {out} {time.time() - t:.1f}s", flush=True)


def stage_assets(a):
    import torch
    from safetensors import safe_open
    emb = f"{a.out}/embeddings.bf16"
    if not os.path.exists(emb) or a.force:
        f = safe_open(f"{a.hf_dir}/model.safetensors", "pt")
        key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in f.keys() else "embed_tokens.weight"
        t = f.get_tensor(key)
        assert t.dtype == torch.bfloat16, t.dtype
        t.contiguous().view(torch.int16).numpy().astype("<i2").tofile(emb + ".tmp")
        os.replace(emb + ".tmp", emb)
    shutil.copy2(f"{a.hf_dir}/tokenizer.json", f"{a.out}/tokenizer.json")
    for extra in ("LICENSE",):
        if os.path.exists(f"{a.hf_dir}/{extra}"):
            shutil.copy2(f"{a.hf_dir}/{extra}", f"{a.out}/{extra}")
    print("[assets] done", flush=True)


def selftest_doc(tok, n):
    """Deterministic token ids: the Apache-2.0 text shipped in the transformers wheel, tokenized, repeated."""
    import importlib.metadata as md
    src = next(f for f in md.files("transformers") if f.name == "LICENSE")
    text = src.read_text()
    ids = tok.encode(text[text.index("Apache License"):], add_special_tokens=False).ids
    doc = []
    while len(doc) < n:
        doc += ids
    return doc[:n]


def selftest_sequences(doc, S, eos):
    """three packed sequences (one long, two short) that end in the model's <|endoftext|>"""
    lens = {64: (40, 11, 9), 128: (80, 30, 12)}.get(S, (S * 5 // 8, 37, S // 8 + 3))
    seqs, off = [], 0
    for n in lens:
        seqs.append(doc[off:off + n - 1] + [eos])
        off += n
    return seqs


def stage_selftest(a, W):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel
    tok = Tokenizer.from_file(f"{a.out}/tokenizer.json")
    eos = tok.token_to_id("<|endoftext|>")
    table = q.TokenTable(f"{a.out}/embeddings.bf16", W.H)
    hf = AutoModel.from_pretrained(a.hf_dir, dtype=torch.float32).eval()
    doc = selftest_doc(tok, 4096)
    blob = bytearray()

    def put(arr):
        arr = np.ascontiguousarray(arr, "<f4")
        off = len(blob)
        blob.extend(arr.tobytes())
        return {"offset": off, "shape": list(arr.shape)}

    def reference(ids):
        with torch.no_grad():
            h = hf(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].numpy()
        return h / np.linalg.norm(h)

    tests = []
    for name, S, _ in specs(W):
        seqs = selftest_sequences(doc, S, eos)
        expected = np.stack([reference(s) for s in seqs])
        with torch.no_grad():
            feed = q.pack([table.rows(s) for s in seqs], S)
            got = q.Bucket(W, S).eval()(*[torch.from_numpy(np.asarray(v, np.float32)) for v in feed.values()]).numpy()
        worst = min(q.cos(got[k], expected[k]) for k in range(len(seqs)))
        assert worst > 0.9999, f"{name}: the fp32 graph differs from Hugging Face (cos {worst})"
        tests.append({"function": name, "kind": "bucket", "sequences": seqs, "expected": put(expected)})
        print(f"[selftest] {name} lens={[len(s) for s in seqs]} fp32 graph cos {worst:.7f}", flush=True)
    open(f"{a.out}/selftest.bin", "wb").write(bytes(blob))
    json.dump({"format": "aneembed-selftest/1", "bin": "selftest.bin", "dtype": "float32", "byte_order": "little",
               "doc_ids_source": "Apache License 2.0 text of the transformers wheel, tokenized without special tokens",
               "tolerance": SELFTEST_TOLERANCE, "tests": tests}, open(f"{a.out}/selftest.json", "w"), indent=1)


def stage_manifest(a, W):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.out}/tokenizer.json")
    functions = {name: {"inputs": [{"name": n, "shape": list(s), "dtype": "float16"} for n, s in ins],
                        "outputs": [{"name": "embedding", "shape": [q.k_for(S), W.H], "dtype": "float32"}]}
                 for name, S, ins in specs(W)}
    files = []
    for root, _, names in os.walk(a.out):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, a.out)
            if rel == "manifest.json" or ".mlmodelc" in rel:
                continue
            files.append({"path": rel, "size": os.path.getsize(p), "sha256": sha256(p)})
    files.sort(key=lambda f: f["path"])
    rev = None
    meta = os.path.join(a.hf_dir, ".cache/huggingface/download/model.safetensors.metadata")
    if os.path.exists(meta):
        rev = open(meta).read().split()[0]
    elif "/snapshots/" in os.path.realpath(a.hf_dir):
        rev = os.path.basename(os.path.realpath(a.hf_dir))
    manifest = {
        "name": NAME,
        "manifest_version": 1,
        "source_model": {"id": a.model_id, "revision": rev,
                         "model.safetensors_sha256": sha256(f"{a.hf_dir}/model.safetensors")},
        "converter": {"script": "scripts/ane/build_qwen3.py",
                      "precision": "fp16 ML Program (compute_precision FLOAT16), minimum macOS 15",
                      "compute_units": "cpuAndNeuralEngine"},
        "embedding": {"dim": W.H, "pooling": "last token", "normalized": "L2, in graph"},
        "special_tokens": {"eos (<|endoftext|>)": tok.token_to_id("<|endoftext|>")},
        "text_sequence": "tokenizer.encode(text, add_special_tokens=True): the text then <|endoftext|>",
        "token_table": {"file": "embeddings.bf16", "dtype": "bfloat16", "byte_order": "little",
                        "shape": [int(os.path.getsize(f"{a.out}/embeddings.bf16") // (2 * W.H)), W.H],
                        "row_to_input": "fp32(bf16 row), then cast to fp16"},
        "bucket_contract": {
            "inputs_embeds": f"[1,S,{W.H}] fp16: sequences back to back from row 0, zero rows after",
            "causal_bias": "[1,1,S,S] fp16: 0 if rows i and j belong to the same sequence and j <= i, or i == j; "
                           "else -10000",
            "pool": "[K,S] fp16: row k = 1 at the last row of sequence k, 0 elsewhere; unused rows all 0",
            "embedding": f"[K,{W.H}] fp32: row k = L2-normalized embedding of sequence k; unused rows are 0",
            "slots": "K = max(8, S / 32)",
        },
        "packages": {PKG: {"mlpackage": f"{PKG}.mlpackage", "compute_units": "cpuAndNeuralEngine",
                           "functions": functions}},
        "files": files,
    }
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(f"[manifest] {len(files)} files", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="local snapshot of the Qwen3-Embedding checkpoint")
    ap.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--stages", default="convert,merge,assets,selftest,manifest")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out) + ".work")
    os.makedirs(a.out, exist_ok=True)
    W = q.Weights(a.hf_dir)
    stages = a.stages.split(",")
    if "convert" in stages:
        stage_convert(a, W)
    if "merge" in stages:
        stage_merge(a, W)
    if "assets" in stages:
        stage_assets(a)
    if "selftest" in stages:
        stage_selftest(a, W)
    if "manifest" in stages:
        stage_manifest(a, W)


if __name__ == "__main__":
    main()
