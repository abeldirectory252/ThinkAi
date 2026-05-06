"""
Gemma 2B decoder for PaliGemma 3B.
Keys match HF: language_model.model.{embed_tokens,layers,norm}

Config: hidden=2048, intermediate=16384, layers=18, heads=8,
        kv_heads=1, head_dim=256, vocab=257216, RoPE, GeGLU.
"""
import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * norm * (1.0 + self.weight.float())).to(orig_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.shape[2]:
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[:, :, offset:offset+seq_len, :],
            self.sin_cached[:, :, offset:offset+seq_len, :],
        )


class KVCache:
    """Simple key-value cache for autoregressive generation."""
    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def clear(self):
        self.k = self.v = None


class GemmaAttention(nn.Module):
    def __init__(self, hidden: int = 2048, heads: int = 8,
                 kv_heads: int = 1, head_dim: int = 256,
                 use_qk_norm: bool = False, eps: float = 1e-6):
        super().__init__()
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_groups = heads // kv_heads
        self.use_qk_norm = use_qk_norm

        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.rope = RotaryEmbedding(head_dim)

        # Gemma 3: QK-Norm
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps)
            self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None,
                output_attentions: bool = False):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Gemma 3: apply QK-norm before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = cache.seq_len if cache else 0
        cos, sin = self.rope(L, offset)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        # GQA: expand kv heads
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        scale = self.head_dim ** -0.5

        if output_attentions:
            # Manual path: materializes full attention matrix
            attn_w = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn_w = attn_w + mask
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).type_as(q)
            out = (attn_w @ v).transpose(1, 2).reshape(B, L, -1)
        else:
            # Memory-efficient SDPA
            attn_mask = mask if mask is not None else None
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=scale
            )
            out = out.transpose(1, 2).reshape(B, L, -1)
            attn_w = None

        out = self.o_proj(out)

        if output_attentions:
            return out, attn_w
        return out, None


class GemmaMLP(nn.Module):
    def __init__(self, hidden: int = 2048, intermediate: int = 16384):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gemma uses GELU gating (not SiLU) per the official reference
        # JAX nn.gelu defaults to approximate=True (tanh); HF uses "gelu_pytorch_tanh"
        return self.down_proj(F.gelu(self.gate_proj(x), approximate='tanh') * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    def __init__(self, hidden=2048, heads=8, kv_heads=1,
                 head_dim=256, intermediate=16384, eps=1e-6,
                 use_qk_norm=False, use_pre_post_ff_norm=False):
        super().__init__()
        self.use_pre_post_ff_norm = use_pre_post_ff_norm
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = GemmaAttention(hidden, heads, kv_heads, head_dim,
                                        use_qk_norm=use_qk_norm, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = GemmaMLP(hidden, intermediate)

        # Gemma 3: extra layernorms around MLP
        if use_pre_post_ff_norm:
            self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
            self.post_feedforward_layernorm = RMSNorm(hidden, eps)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        r = x
        x = self.input_layernorm(x)
        x, attn = self.self_attn(x, mask, cache, output_attentions)
        x = r + x
        r = x
        x = self.post_attention_layernorm(x)
        if self.use_pre_post_ff_norm:
            x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        if self.use_pre_post_ff_norm:
            x = self.post_feedforward_layernorm(x)
        x = r + x
        return x, attn


class GemmaModel(nn.Module):
    def __init__(self, vocab=257216, hidden=2048, layers=18,
                 heads=8, kv_heads=1, head_dim=256,
                 intermediate=16384, eps=1e-6,
                 use_qk_norm=False, use_pre_post_ff_norm=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(hidden, heads, kv_heads, head_dim, intermediate, eps,
                              use_qk_norm=use_qk_norm,
                              use_pre_post_ff_norm=use_pre_post_ff_norm)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden, eps)
        self.hidden_size = hidden

    def forward(self, input_ids=None, inputs_embeds=None,
                mask=None, caches=None, output_attentions=False):
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
            # Gemma scales embeddings by sqrt(hidden_size)
            h = h * (self.hidden_size ** 0.5)
        else:
            # inputs_embeds are already scaled by the caller (PaliGemma)
            h = inputs_embeds

        all_attn = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            h, attn = layer(h, mask, cache, output_attentions)
            if output_attentions:
                all_attn.append(attn)
        h = self.norm(h)
        return h, all_attn


class GemmaForCausalLM(nn.Module):
    """Wrapper matching HF prefix `language_model.`."""
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GemmaModel(**kwargs)
        # lm_head shares weights with embed_tokens (tied)
        self.vocab_size = kwargs.get("vocab", 257216)

    def forward(self, input_ids=None, inputs_embeds=None,
                mask=None, caches=None, output_attentions=False):
        h, attn = self.model(input_ids, inputs_embeds, mask, caches, output_attentions)
        # tied lm_head
        logits = F.linear(h, self.model.embed_tokens.weight)
        return logits, attn

    def init_caches(self, num_layers: int = 18):
        return [KVCache() for _ in range(num_layers)]

    def clear_caches(self, caches):
        for c in caches:
            c.clear()
