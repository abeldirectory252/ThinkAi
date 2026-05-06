"""
Primitive layers for PaliGemma (Gemma 1 decoder).
Fully self-contained — no cross-model dependencies.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * norm * (1.0 + self.weight.float())).to(orig_dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=8192, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len, offset=0):
        if offset + seq_len > self.cos_cached.shape[2]:
            self._build_cache(offset + seq_len)
        return self.cos_cached[:, :, offset:offset+seq_len, :], self.sin_cached[:, :, offset:offset+seq_len, :]


class KVCache:
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        if self.k is None: self.k, self.v = k, v
        else: self.k, self.v = torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)
        return self.k, self.v

    @property
    def seq_len(self): return 0 if self.k is None else self.k.shape[2]
    def clear(self): self.k = self.v = None


class GemmaMLP(nn.Module):
    def __init__(self, hidden=2048, intermediate=16384):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))
