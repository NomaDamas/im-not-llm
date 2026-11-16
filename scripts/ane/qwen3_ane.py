"""Qwen3-Embedding text encoder re-expressed for Core ML / the Apple Neural Engine.

Mirrors transformers' Qwen3Model + last-token pooling + L2 normalize (sentence-transformers config of
Qwen/Qwen3-Embedding-*). Differences are numerically neutral rewrites the fp16 ANE graph needs:
  * RMSNorm divides each row by its max-abs first (fp16 overflow guard; exact in real arithmetic).
  * Attention uses the "fold" layout: the query heads sharing one K/V head are folded into the row
    dimension instead of repeat_interleave on K/V.
  * RoPE tables are graph constants at absolute positions 0..S-1; the causal + packing mask is an input.
  * The token embedding lookup happens on the host (embeddings.bf16), so the graph starts at inputs_embeds.
  * Pooling is a [K,S] matrix input (one-hot on each packed sequence's last row), applied before the final
    norm (the norm is per row, so pooling first is exact).

Bucket(S): inputs_embeds [1,S,H], causal_bias [1,1,S,S], pool [K,S] -> embedding [K,H] (L2-normalized).
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NEG = -1e4
BUCKETS = (64, 128, 256, 512, 768, 1024, 1536, 2048)


def k_for(S):
    """packed sequences (pooled rows) per call"""
    return max(8, S // 32)


def rms(x, w, eps):
    s = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=x.dtype))
    y = x / s
    return y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps / (s * s)) * w


def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(n, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    fr = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos().float(), emb.sin().float()


def attend_fold(q, k, v, bias, nh, kvh, S, hd):
    """q [1,nh,S,hd], k/v [1,kvh,S,hd], bias [1,1,S,S] -> [1,nh,S,hd]"""
    sq = sk = S
    g = nh // kvh
    qf = q.reshape(1, kvh, g * sq, hd)
    s = torch.matmul(qf, k.transpose(-1, -2)).reshape(1, nh, sq, sk) + bias
    w = torch.softmax(s, dim=-1).reshape(1, kvh, g * sq, sk)
    return torch.matmul(w, v).reshape(1, nh, sq, hd)


class Weights(nn.Module):
    def __init__(self, hf_dir):
        super().__init__()
        cfg = json.load(open(f"{hf_dir}/config.json"))
        assert cfg["model_type"] == "qwen3" and not cfg.get("use_sliding_window"), cfg
        self.cfg = cfg
        f = safe_open(f"{hf_dir}/model.safetensors", "pt")
        prefix = "model." if "model.norm.weight" in f.keys() else ""

        def g(k):
            return nn.Parameter(f.get_tensor(prefix + k).float(), requires_grad=False)

        self.H = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.kvh = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or self.H // self.nh
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.norm = g("norm.weight")
        self.layers = nn.ModuleList()
        for i in range(cfg["num_hidden_layers"]):
            m = nn.Module()
            m.ln1 = g(f"layers.{i}.input_layernorm.weight")
            m.ln2 = g(f"layers.{i}.post_attention_layernorm.weight")
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                setattr(m, k, g(f"layers.{i}.self_attn.{k}.weight"))
            for k in ("gate_proj", "up_proj", "down_proj"):
                setattr(m, k, g(f"layers.{i}.mlp.{k}.weight"))
            self.layers.append(m)


def encoder(W, x, c, s, bias, S):
    """x [1,S,H] -> final hidden states before the final norm (S static)"""
    hd, nh, kvh, eps = W.hd, W.nh, W.kvh, W.eps
    scale = hd ** -0.5
    for m in W.layers:
        h = rms(x, m.ln1, eps)
        q = rms(F.linear(h, m.q_proj).reshape(1, S, nh, hd).transpose(1, 2), m.q_norm, eps)
        k = rms(F.linear(h, m.k_proj).reshape(1, S, kvh, hd).transpose(1, 2), m.k_norm, eps)
        v = F.linear(h, m.v_proj).reshape(1, S, kvh, hd).transpose(1, 2)
        q = (q * c + rot_half(q, hd // 2) * s) * scale
        k = k * c + rot_half(k, hd // 2) * s
        o = attend_fold(q, k, v, bias, nh, kvh, S, hd).transpose(1, 2).reshape(1, S, nh * hd)
        x = x + F.linear(o, m.o_proj)
        h = rms(x, m.ln2, eps)
        x = x + F.linear(F.silu(F.linear(h, m.gate_proj)) * F.linear(h, m.up_proj), m.down_proj)
    return x


class Bucket(nn.Module):
    """embed_S: inputs_embeds [1,S,H], causal_bias [1,1,S,S], pool [K,S] -> embedding [K,H]"""

    def __init__(self, W, S):
        super().__init__()
        self.W, self.S = W, S
        c, s = rope_tables(S, W.hd, W.theta)
        self.register_buffer("cos", c, persistent=False)
        self.register_buffer("sin", s, persistent=False)

    def forward(self, inputs_embeds, causal_bias, pool):
        x = encoder(self.W, inputs_embeds, self.cos, self.sin, causal_bias, self.S)
        e = rms(torch.matmul(pool, x[0]), self.W.norm, self.W.eps)
        m = torch.maximum(e.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=e.dtype))
        e = e / m
        return e * torch.rsqrt((e * e).sum(-1, keepdim=True) + 1e-6)


def pack(rows, S):
    """host contract (numpy): rows = list of fp32 [n_k,H] sequences -> feed of embed_S"""
    K = k_for(S)
    H = rows[0].shape[1]
    assert len(rows) <= K and sum(len(r) for r in rows) <= S
    x = np.zeros((1, S, H), np.float32)
    seg = np.full(S, -1)
    pool = np.zeros((K, S), np.float32)
    off = 0
    for k, r in enumerate(rows):
        n = len(r)
        x[0, off:off + n] = r
        seg[off:off + n] = k
        pool[k, off + n - 1] = 1.0
        off += n
    i = np.arange(S)
    ok = ((seg[:, None] == seg[None, :]) & (seg[None, :] >= 0) & (i[None, :] <= i[:, None])) | (i[:, None] == i[None, :])
    bias = np.where(ok, 0.0, NEG).astype(np.float16).reshape(1, 1, S, S)
    return {"inputs_embeds": x.astype(np.float16), "causal_bias": bias, "pool": pool.astype(np.float16)}


class TokenTable:
    def __init__(self, path, H):
        self.raw = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, H)

    def rows(self, ids):
        return (self.raw[np.asarray(ids, np.int64)].astype(np.uint32) << 16).view(np.float32)


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
