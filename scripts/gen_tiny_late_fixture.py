"""Regenerates tests/fixtures/tiny_qwen3_5_late (dev-only; never needed at runtime).

A tiny random late-interaction checkpoint laid out like
perplexity-ai/pplx-embed-v2-late-0.6b: a Qwen3.5 text tower (`language_model.*`
tensors, hybrid gated-delta-net / full-attention layers, nested
`text_config`), then the Sentence-Transformers `MultiVectorEncoder` modules
(`1_Dense` projection, `2_MultiVectorMask` punctuation skiplist for documents,
`3_Normalize`) and the `[Q] ` / `[D] ` prompts. The golden multi-vectors are
computed with the Hugging Face reference (`transformers`
Qwen3_5TextModel, bidirectional full attention) and the same post-processing
the modules apply. The Rust port must reproduce them.

    uv venv .venv && uv pip install --python .venv/bin/python \
        "transformers>=5.4" torch safetensors tokenizers
    .venv/bin/python scripts/gen_tiny_late_fixture.py \
        crates/inl-late/tests/fixtures/tiny_qwen3_5_late
"""

import json, os, sys, torch
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(7)

# ---- tokenizer: <pad>=0 <unk>=1, punctuation, words, then the two prompt tokens
vocab = {"<pad>": 0, "<unk>": 1, ".": 2, ",": 3, "?": 4, "!": 5}
for i in range(54):
    vocab[f"w{i}"] = 6 + i
tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.add_special_tokens([AddedToken("[Q] ", normalized=False), AddedToken("[D] ", normalized=False)])
tok.save(os.path.join(OUT, "tokenizer.json"))
Q_ID, D_ID = tok.token_to_id("[Q] "), tok.token_to_id("[D] ")
VOCAB = 64
assert D_ID < VOCAB

HID, INTER, HEADS, KV, HEAD_DIM = 32, 64, 4, 2, 16
LIN_K_HEADS, LIN_V_HEADS, LIN_K_DIM, LIN_V_DIM = 2, 4, 8, 8
OUT_DIM = 12
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention", "full_attention"]
rope = {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0], "mrope_interleaved": True}
cfg = Qwen3_5TextConfig(
    vocab_size=VOCAB, hidden_size=HID, intermediate_size=INTER, num_hidden_layers=len(LAYER_TYPES),
    num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HEAD_DIM,
    linear_num_key_heads=LIN_K_HEADS, linear_num_value_heads=LIN_V_HEADS,
    linear_key_head_dim=LIN_K_DIM, linear_value_head_dim=LIN_V_DIM, linear_conv_kernel_dim=4,
    layer_types=LAYER_TYPES, rope_parameters=rope, max_position_embeddings=4096,
    rms_norm_eps=1e-6, is_causal=False)
cfg._attn_implementation = "eager"
model = Qwen3_5TextModel(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn_like(p))
        elif p.dim() == 1:
            # zero-centred norms (`1 + w`) and the gated norm's plain weight alike
            p.copy_(0.3 * torch.randn_like(p) + (1.0 if "linear_attn.norm" in name else 0.0))
        elif name.endswith("conv1d.weight"):
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * (1.5 / p.shape[-1] ** 0.5))
dense = torch.randn(OUT_DIM, HID) * (1.0 / HID ** 0.5)

state = {"language_model." + k: v.clone().contiguous() for k, v in model.state_dict().items()}
save_file(state, os.path.join(OUT, "model.safetensors"))
os.makedirs(os.path.join(OUT, "1_Dense"), exist_ok=True)
save_file({"linear.weight": dense.contiguous()}, os.path.join(OUT, "1_Dense", "model.safetensors"))

text = {
    "model_type": "qwen3_5_text", "vocab_size": VOCAB, "hidden_size": HID,
    "intermediate_size": INTER, "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HEAD_DIM,
    "linear_num_key_heads": LIN_K_HEADS, "linear_num_value_heads": LIN_V_HEADS,
    "linear_key_head_dim": LIN_K_DIM, "linear_value_head_dim": LIN_V_DIM,
    "linear_conv_kernel_dim": 4, "layer_types": LAYER_TYPES, "hidden_act": "silu",
    "rms_norm_eps": 1e-6, "attn_output_gate": True, "attention_bias": False, "is_causal": False,
    "rope_parameters": rope, "max_position_embeddings": 4096, "tie_word_embeddings": True,
}
json.dump({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5", "text_config": text},
          open(os.path.join(OUT, "config.json"), "w"), indent=1)
json.dump([
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Dense", "type": "sentence_transformers.base.modules.dense.Dense"},
    {"idx": 2, "name": "2", "path": "2_MultiVectorMask",
     "type": "sentence_transformers.multi_vector_encoder.modules.MultiVectorMask"},
    {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.base.modules.Normalize"},
], open(os.path.join(OUT, "modules.json"), "w"), indent=1)
json.dump({"in_features": HID, "out_features": OUT_DIM, "bias": False,
           "activation_function": "torch.nn.modules.linear.Identity",
           "module_input_name": "token_embeddings", "module_output_name": "token_embeddings",
           "use_residual": False}, open(os.path.join(OUT, "1_Dense", "config.json"), "w"), indent=1)
os.makedirs(os.path.join(OUT, "2_MultiVectorMask"), exist_ok=True)
# "~" is not a vocabulary token: it must be dropped, not turned into <unk>.
SKIP = [".", ",", "!", "~"]
json.dump({"skiplist_words": SKIP, "skiplist_tasks": ["document"], "keep_only_token_ids": None},
          open(os.path.join(OUT, "2_MultiVectorMask", "config.json"), "w"), indent=1)
os.makedirs(os.path.join(OUT, "3_Normalize"), exist_ok=True)
json.dump({}, open(os.path.join(OUT, "3_Normalize", "config.json"), "w"))
json.dump({"model_type": "MultiVectorEncoder", "prompts": {"query": "[Q] ", "document": "[D] "},
           "default_prompt_name": None, "similarity_fn_name": "maxsim"},
          open(os.path.join(OUT, "config_sentence_transformers.json"), "w"), indent=1)
QUERY_LEN, DOC_LEN = 24, 160
json.dump({"query_length": QUERY_LEN, "document_length": DOC_LEN, "module_output_name": "token_embeddings"},
          open(os.path.join(OUT, "sentence_bert_config.json"), "w"), indent=1)

skip_ids = {vocab[w] for w in SKIP if w in vocab}


def embed(texts, task, max_length):
    """MultiVectorEncoder: prompt, forward, Dense, skiplist mask, L2 normalize."""
    t = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    t.enable_truncation(max_length=max_length)
    t.enable_padding(pad_id=0, pad_token="<pad>", direction="right")
    prompt = "[Q] " if task == "query" else "[D] "
    enc = t.encode_batch([prompt + x for x in texts])
    ids = torch.tensor([e.ids for e in enc])
    mask = torch.tensor([e.attention_mask for e in enc])
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        vecs = torch.nn.functional.normalize(hidden @ dense.T, dim=-1)
    out = []
    for row, e in enumerate(enc):
        keep = [i for i, (tid, m) in enumerate(zip(e.ids, e.attention_mask))
                if m and not (task == "document" and tid in skip_ids)]
        out.append(vecs[row, keep].tolist())
    return out, [sum(e.attention_mask) for e in enc]


# The full-attention layers are bidirectional: appending a token moves earlier rows.
with torch.no_grad():
    a = model(input_ids=torch.tensor([[6, 7, 8]])).last_hidden_state[0, :2]
    b = model(input_ids=torch.tensor([[6, 7, 8, 9]])).last_hidden_state[0, :2]
assert (a - b).abs().max() > 1e-4, "reference ran causal"

words = lambda a, b: " ".join(f"w{i % 54}" for i in range(a, b))
texts = [
    words(0, 5) + " .",
    words(3, 20) + ", " + words(20, 24) + " !",
    "",
    words(0, 150) + " . ~ zzz",              # three gated-delta chunks, truncated as a query
    words(10, 75) + " ?",                    # crosses one chunk boundary
]
cases = []
for task, max_length in (("query", QUERY_LEN), ("document", DOC_LEN)):
    batched, lengths = embed(texts, task, max_length)
    alone = [embed([x], task, max_length)[0][0] for x in texts]
    cases.append({"task": task, "batched": batched, "alone": alone, "lengths": lengths})
json.dump({"dim": OUT_DIM, "texts": texts, "cases": cases}, open(os.path.join(OUT, "golden.json"), "w"))
print("ok", {c["task"]: c["lengths"] for c in cases}, "ids", Q_ID, D_ID)
