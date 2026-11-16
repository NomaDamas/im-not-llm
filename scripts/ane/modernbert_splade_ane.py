"""ModernBERT masked-LM + SPLADE pooling (nlpai-lab/KURE-Sparse-nano) re-expressed for Core ML / the
Apple Neural Engine.

Mirrors transformers' ModernBertForMaskedLM (bidirectional; full-attention layers every
`global_attn_every_n_layers`, the rest a sliding window of `local_attention / 2` keys on each side;
GeGLU MLP; head dense + gelu + norm; decoder tied to the token embeddings, with bias) followed by the
checkpoint's ShiftedTopKSpladePooling up to the max over positions. Rewrites the fp16 ANE graph needs,
each exact in real arithmetic:
  * LayerNorm divides each row by its max-abs first (fp16 overflow guard on the variance).
  * RoPE tables are graph constants at absolute positions 0..S-1 (RoPE is relative, so a packed text
    gets the same scores as alone); the packing / window masks are inputs.
  * The token embedding lookup happens on the host (embeddings.bf16): the graph starts at
    inputs_embeds.
  * relu(logit - shift) * keep is max-reduced over aligned blocks of BLOCK rows inside the graph; texts
    start on a BLOCK boundary, so the host finishes the max per text over its own blocks, then
    log1p, the masked vocabulary and the canonical fold (all monotone or per term, so they commute
    with the max).

Splade(S): inputs_embeds [1,S,H], global_bias [1,1,S,S], local_bias [1,1,S,S], keep [S,1]
           -> block_max [S/BLOCK, V] (fp16, >= 0).
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NEG = -1e4
BLOCK = 8
BUCKETS = (64, 128, 256, 512, 768, 1024, 1536, 2048)


def norm(x, w, eps):
    """LayerNorm without bias, max-abs scaled first."""
    s = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=x.dtype))
    y = x / s
    c = y - y.mean(-1, keepdim=True)
    return c * torch.rsqrt((c * c).mean(-1, keepdim=True) + eps / (s * s)) * w


def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(n, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    fr = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos().float(), emb.sin().float()


class Weights(nn.Module):
    def __init__(self, hf_dir):
        super().__init__()
        cfg = json.load(open(f"{hf_dir}/config.json"))
        assert cfg["model_type"] == "modernbert", cfg["model_type"]
        assert not cfg.get("attention_bias") and not cfg.get("mlp_bias") and not cfg.get("norm_bias"), cfg
        assert cfg["hidden_activation"] == "gelu" and cfg["classifier_activation"] == "gelu", cfg
        assert cfg.get("tie_word_embeddings", True) and cfg.get("decoder_bias", True), cfg
        self.cfg = cfg
        self.pooling = json.load(open(f"{hf_dir}/1_ShiftedTopKSpladePooling/config.json"))
        p = self.pooling
        assert p["pooling_strategy"] == "max" and p["activation_function"] == "relu" and not p.get("top_k"), p
        f = safe_open(f"{hf_dir}/model.safetensors", "pt")

        def g(k):
            return nn.Parameter(f.get_tensor(k).float(), requires_grad=False)

        self.H = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.hd = self.H // self.nh
        self.I = cfg["intermediate_size"]
        self.eps = cfg["norm_eps"]
        self.half_window = cfg["local_attention"] // 2
        rope = cfg["rope_parameters"]
        self.theta_global = rope["full_attention"]["rope_theta"]
        self.theta_local = rope["sliding_attention"]["rope_theta"]
        types = cfg.get("layer_types") or [
            "full_attention" if i % cfg["global_attn_every_n_layers"] == 0 else "sliding_attention"
            for i in range(cfg["num_hidden_layers"])]
        self.shift = float(p.get("logit_shift") or 0.0)
        self.emb_norm = g("model.embeddings.norm.weight")
        self.final_norm = g("model.final_norm.weight")
        self.head_dense = g("head.dense.weight")
        self.head_norm = g("head.norm.weight")
        self.decoder = g("model.embeddings.tok_embeddings.weight")
        self.decoder_bias = g("decoder.bias")
        self.V = self.decoder.shape[0]
        self.layers = nn.ModuleList()
        for i, kind in enumerate(types):
            m = nn.Module()
            m.globl = kind == "full_attention"
            key = f"model.layers.{i}.attn_norm.weight"
            m.attn_norm = g(key) if key in f.keys() else None
            m.wqkv = g(f"model.layers.{i}.attn.Wqkv.weight")
            m.wo = g(f"model.layers.{i}.attn.Wo.weight")
            m.mlp_norm = g(f"model.layers.{i}.mlp_norm.weight")
            m.wi = g(f"model.layers.{i}.mlp.Wi.weight")
            m.mlp_wo = g(f"model.layers.{i}.mlp.Wo.weight")
            self.layers.append(m)


def attend(h, m, W, c, s, bias, S):
    qkv = F.linear(h, m.wqkv).reshape(1, S, 3, W.nh, W.hd)
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
    half = W.hd // 2
    q = (q * c + rot_half(q, half) * s) * (W.hd ** -0.5)
    k = k * c + rot_half(k, half) * s
    w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1)
    o = torch.matmul(w, v).transpose(1, 2).reshape(1, S, W.H)
    return F.linear(o, m.wo)


def logits(W, x, tables, global_bias, local_bias, S):
    """inputs_embeds [1,S,H] -> MLM logits [1,S,V]"""
    (gc, gs), (lc, ls) = tables
    x = norm(x, W.emb_norm, W.eps)
    for m in W.layers:
        h = norm(x, m.attn_norm, W.eps) if m.attn_norm is not None else x
        if m.globl:
            x = x + attend(h, m, W, gc, gs, global_bias, S)
        else:
            x = x + attend(h, m, W, lc, ls, local_bias, S)
        a, gate = F.linear(norm(x, m.mlp_norm, W.eps), m.wi).split(W.I, dim=-1)
        x = x + F.linear(F.gelu(a) * gate, m.mlp_wo)
    x = norm(x, W.final_norm, W.eps)
    x = norm(F.gelu(F.linear(x, W.head_dense)), W.head_norm, W.eps)
    return F.linear(x, W.decoder, W.decoder_bias)


class Splade(nn.Module):
    """splade_S: inputs_embeds [1,S,H], global_bias [1,1,S,S], local_bias [1,1,S,S], keep [S,1]
    -> block_max [S/BLOCK, V]"""

    def __init__(self, W, S):
        super().__init__()
        assert S % BLOCK == 0
        self.W, self.S = W, S
        gc, gs = rope_tables(S, W.hd, W.theta_global)
        lc, ls = rope_tables(S, W.hd, W.theta_local)
        for name, t in (("gc", gc), ("gs", gs), ("lc", lc), ("ls", ls)):
            self.register_buffer(name, t, persistent=False)

    def forward(self, inputs_embeds, global_bias, local_bias, keep):
        tables = ((self.gc, self.gs), (self.lc, self.ls))
        z = logits(self.W, inputs_embeds, tables, global_bias, local_bias, self.S)[0]
        z = torch.relu(z - self.W.shift) * keep
        # A halving tree of maxima over the block's rows: reduce_max over a [S/B, B, V] view runs about
        # 3x slower on the Neural Engine (64.7 vs 35.8 ms for the whole 512-row call, M4 Max).
        x = z.reshape(self.S // BLOCK, BLOCK, self.W.V)
        rows = BLOCK
        while rows > 1:
            rows //= 2
            x = torch.maximum(x[:, :rows], x[:, rows:])
        return x[:, 0]


def aligned(n):
    return -(-n // BLOCK) * BLOCK


def pack(rows, keeps, S, half_window):
    """host contract (numpy): rows = fp32 [n_k,H] sequences, keeps = 0/1 per row -> feed of splade_S.
    Text k starts at the BLOCK-aligned offset after text k-1."""
    H = rows[0].shape[1]
    x = np.zeros((1, S, H), np.float32)
    seg = np.full(S, -1)
    keep = np.zeros((S, 1), np.float32)
    off = 0
    offsets = []
    for k, (r, kp) in enumerate(zip(rows, keeps)):
        n = len(r)
        assert off + n <= S
        x[0, off:off + n] = r
        seg[off:off + n] = k
        keep[off:off + n, 0] = kp
        offsets.append(off)
        off += aligned(n)
    i = np.arange(S)
    same = (seg[:, None] == seg[None, :]) & (seg[None, :] >= 0)
    self_ = i[:, None] == i[None, :]
    g = np.where(same | self_, 0.0, NEG)
    near = np.abs(i[:, None] - i[None, :]) <= half_window
    loc = np.where((same & near) | self_, 0.0, NEG)
    feed = {"inputs_embeds": x.astype(np.float16), "global_bias": g.astype(np.float16).reshape(1, 1, S, S),
            "local_bias": loc.astype(np.float16).reshape(1, 1, S, S), "keep": keep.astype(np.float16)}
    return feed, offsets


def finish(block_max, offset, n, pooling):
    """host contract (numpy): one text's blocks -> folded SPLADE weights [dim] (fp32)"""
    pooled = np.asarray(block_max, np.float32)[offset // BLOCK:(offset + aligned(n)) // BLOCK].max(0)
    pooled[np.asarray(pooling.get("masked_vocab_ids") or [], np.int64)] = 0.0
    w = np.log1p(pooled)
    canon = pooling.get("canonical_ids")
    if canon is None:
        return w
    out = np.zeros(max(canon) + 1, np.float32)
    np.maximum.at(out, np.asarray(canon), w)
    return out


class TokenTable:
    def __init__(self, path, H):
        self.raw = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, H)

    def rows(self, ids):
        return (self.raw[np.asarray(ids, np.int64)].astype(np.uint32) << 16).view(np.float32)


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
