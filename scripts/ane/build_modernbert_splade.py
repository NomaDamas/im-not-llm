#!/usr/bin/env python3
"""Build the Apple Neural Engine (Core ML) package of a ModernBERT SPLADE checkpoint
(nlpai-lab/KURE-Sparse-nano) for `inl-ane`.

    python build_modernbert_splade.py --hf-dir <KURE-Sparse-nano snapshot> --out <package dir>

Stages (idempotent; --force redoes them):
  convert   one fp16 ML Program per function into <work>/fn: splade_S for S in BUCKETS (packed texts on
            8-row boundaries, block-diagonal global / windowed attention, block max of relu(logit - shift))
  merge     multifunction package ModernBertSpladeText.mlpackage (identical weights are stored once)
  assets    embeddings.bf16 (token table, raw bf16 rows), tokenizer.json, splade_pooling.json
  selftest  selftest.json + selftest.bin: one golden test per function, expectations from the fp32
            Sentence-Transformers SparseEncoder (folded SPLADE weights, dense)
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
import modernbert_splade_ane as m  # noqa: E402

PKG = "ModernBertSpladeText"
NAME = "kure-sparse-nano-ane"
SELFTEST_TOLERANCE = {"splade": {"min_cos": 0.995}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def function_name(S):
    return f"splade_{S}"


def specs(W):
    return [(function_name(S), S, [("inputs_embeds", (1, S, W.H)), ("global_bias", (1, 1, S, S)),
                                    ("local_bias", (1, 1, S, S)), ("keep", (S, 1))]) for S in m.BUCKETS]


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
        mod = m.Splade(W, S).eval()
        with torch.no_grad():
            tr = torch.jit.trace(mod, tuple(torch.zeros(*s) for _, s in ins), check_trace=False)
        ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float16) for n, s in ins],
                        outputs=[ct.TensorType(name="block_max", dtype=np.float16)], convert_to="mlprogram",
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
        t = safe_open(f"{a.hf_dir}/model.safetensors", "pt").get_tensor("model.embeddings.tok_embeddings.weight")
        assert t.dtype == torch.bfloat16, t.dtype
        t.contiguous().view(torch.int16).numpy().astype("<i2").tofile(emb + ".tmp")
        os.replace(emb + ".tmp", emb)
    shutil.copy2(f"{a.hf_dir}/tokenizer.json", f"{a.out}/tokenizer.json")
    shutil.copy2(f"{a.hf_dir}/1_ShiftedTopKSpladePooling/config.json", f"{a.out}/splade_pooling.json")
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


def selftest_sequences(doc, S, cls, sep, marker):
    """three packed texts (one long, two short) as the tokenizer frames them: [CLS] marker ... [SEP]"""
    lens = {64: (30, 11, 9), 128: (77, 29, 12)}.get(S, (S * 5 // 8 - 3, 37, S // 8 + 3))
    seqs, off = [], 0
    for n in lens:
        seqs.append([cls, marker] + doc[off:off + n - 3] + [sep])
        off += n
    assert sum(m.aligned(len(s)) for s in seqs) <= S
    return seqs


def stage_selftest(a, W):
    import torch
    from sentence_transformers import SparseEncoder
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{a.out}/tokenizer.json")
    tok.no_padding()
    tok.no_truncation()
    cls, doc_marker, sep = tok.encode("[D] ").ids
    assert doc_marker in W.pooling["prefix_token_ids"], doc_marker
    table = m.TokenTable(f"{a.out}/embeddings.bf16", W.H)
    st = SparseEncoder(a.hf_dir, trust_remote_code=True, device="cpu", model_kwargs={"dtype": torch.float32})
    hf = st[0].auto_model
    doc = selftest_doc(tok, 4096)
    prefix = set(W.pooling["prefix_token_ids"])
    blob = bytearray()

    def put(arr):
        arr = np.ascontiguousarray(arr, "<f4")
        off = len(blob)
        blob.extend(arr.tobytes())
        return {"offset": off, "shape": list(arr.shape)}

    def reference(ids):
        features = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        with torch.no_grad():
            features["token_embeddings"] = hf(**features).logits
            return st[1](features)["sentence_embedding"][0].numpy()

    tests = []
    for name, S, _ in specs(W):
        seqs = selftest_sequences(doc, S, cls, sep, doc_marker)
        expected = np.stack([reference(s) for s in seqs])
        feed, offsets = m.pack([table.rows(s) for s in seqs], [[0.0 if t in prefix else 1.0 for t in s] for s in seqs],
                               S, W.half_window)
        with torch.no_grad():
            out = m.Splade(W, S).eval()(*[torch.from_numpy(np.asarray(v, np.float32)) for v in feed.values()]).numpy()
        got = [m.finish(out, offsets[k], len(s), W.pooling) for k, s in enumerate(seqs)]
        worst = min(m.cos(got[k], expected[k]) for k in range(len(seqs)))
        assert worst > 0.9999, f"{name}: the fp32 graph differs from Sentence-Transformers (cos {worst})"
        tests.append({"function": name, "kind": "splade", "sequences": seqs, "expected": put(expected)})
        print(f"[selftest] {name} lens={[len(s) for s in seqs]} fp32 graph cos {worst:.7f}", flush=True)
    open(f"{a.out}/selftest.bin", "wb").write(bytes(blob))
    json.dump({"format": "aneembed-selftest/1", "bin": "selftest.bin", "dtype": "float32", "byte_order": "little",
               "doc_ids_source": "Apache License 2.0 text of the transformers wheel, tokenized without special tokens",
               "tolerance": SELFTEST_TOLERANCE, "tests": tests}, open(f"{a.out}/selftest.json", "w"), indent=1)


def stage_manifest(a, W):
    functions = {name: {"inputs": [{"name": n, "shape": list(s), "dtype": "float16"} for n, s in ins],
                        "outputs": [{"name": "block_max", "shape": [S // m.BLOCK, W.V], "dtype": "float16"}]}
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
    if "/snapshots/" in os.path.realpath(a.hf_dir):
        rev = os.path.basename(os.path.realpath(a.hf_dir))
    canon = W.pooling.get("canonical_ids")
    manifest = {
        "name": NAME,
        "manifest_version": 1,
        "source_model": {"id": a.model_id, "revision": rev,
                         "model.safetensors_sha256": sha256(f"{a.hf_dir}/model.safetensors")},
        "converter": {"script": "scripts/ane/build_modernbert_splade.py",
                      "precision": "fp16 ML Program (compute_precision FLOAT16), minimum macOS 15",
                      "compute_units": "cpuAndNeuralEngine"},
        "embedding": {"dim": (max(canon) + 1) if canon else W.V, "pooling": "SPLADE max (splade_pooling.json)",
                      "vocab": W.V, "block": m.BLOCK, "logit_shift_in_graph": W.shift},
        "text_sequence": "tokenizer.encode(text, add_special_tokens=True): [CLS] text [SEP]",
        "token_table": {"file": "embeddings.bf16", "dtype": "bfloat16", "byte_order": "little",
                        "shape": [int(os.path.getsize(f"{a.out}/embeddings.bf16") // (2 * W.H)), W.H],
                        "row_to_input": "fp32(bf16 row), then cast to fp16"},
        "bucket_contract": {
            "inputs_embeds": f"[1,S,{W.H}] fp16: text k from row offset_k (a multiple of {m.BLOCK}, after text k-1), "
                             "zero rows elsewhere",
            "global_bias": "[1,1,S,S] fp16: 0 if rows i and j belong to the same text, or i == j; else -10000",
            "local_bias": f"[1,1,S,S] fp16: global_bias, and also -10000 where |i - j| > {W.half_window}",
            "keep": "[S,1] fp16: 1 on a text's rows whose token is not in prefix_token_ids, else 0",
            "block_max": f"[S/{m.BLOCK},{W.V}] fp16: row b = max over rows {m.BLOCK}b..{m.BLOCK}b+{m.BLOCK - 1} of "
                         f"relu(logit - {W.shift}) * keep",
            "host": "per text: max over its blocks, masked_vocab_ids -> 0, log1p, max-fold onto canonical_ids",
        },
        "packages": {PKG: {"mlpackage": f"{PKG}.mlpackage", "compute_units": "cpuAndNeuralEngine",
                           "functions": functions}},
        "files": files,
    }
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(f"[manifest] {len(files)} files", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="local snapshot of the ModernBERT SPLADE checkpoint")
    ap.add_argument("--model-id", default="nlpai-lab/KURE-Sparse-nano")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--stages", default="convert,merge,assets,selftest,manifest")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out) + ".work")
    os.makedirs(a.out, exist_ok=True)
    W = m.Weights(a.hf_dir)
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
