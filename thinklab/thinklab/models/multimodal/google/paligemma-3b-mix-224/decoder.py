"""Gemma 1 decoder for PaliGemma. Self-contained."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RMSNorm, RotaryEmbedding, apply_rope, KVCache, GemmaMLP


class GemmaAttention(nn.Module):
    def __init__(self, hidden=2048, heads=8, kv_heads=1, head_dim=256, eps=1e-6):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.kv_groups = heads // kv_heads
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        offset = cache.seq_len if cache else 0
        cos, sin = self.rope(L, offset)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache: k, v = cache.update(k, v)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        scale = self.head_dim ** -0.5
        if output_attentions:
            aw = (q @ k.transpose(-2, -1)) * scale
            if mask is not None: aw = aw + mask
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).type_as(q)
            out = (aw @ v).transpose(1, 2).reshape(B, L, -1)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            out = out.transpose(1, 2).reshape(B, L, -1)
            aw = None
        return self.o_proj(out), aw


class GemmaDecoderLayer(nn.Module):
    def __init__(self, hidden=2048, heads=8, kv_heads=1, head_dim=256, intermediate=16384, eps=1e-6):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = GemmaAttention(hidden, heads, kv_heads, head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = GemmaMLP(hidden, intermediate)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        r = x; x, a = self.self_attn(self.input_layernorm(x), mask, cache, output_attentions)
        x = r + x; r = x; x = r + self.mlp(self.post_attention_layernorm(x))
        return x, a


class GemmaModel(nn.Module):
    def __init__(self, vocab=257216, hidden=2048, layers=18, heads=8, kv_heads=1,
                 head_dim=256, intermediate=16384, eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.hidden_size = hidden
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(hidden, heads, kv_heads, head_dim, intermediate, eps)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden, eps)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None, caches=None, output_attentions=False):
        h = self.embed_tokens(input_ids) * (self.hidden_size ** 0.5) if inputs_embeds is None else inputs_embeds
        aa = [] if output_attentions else None
        for i, l in enumerate(self.layers):
            h, a = l(h, mask, caches[i] if caches else None, output_attentions)
            if aa is not None: aa.append(a)
        return self.norm(h), aa


class GemmaForCausalLM(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = GemmaModel(**kw)
        self.vocab_size = kw.get("vocab", 257216)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None, caches=None, output_attentions=False):
        h, a = self.model(input_ids, inputs_embeds, mask, caches, output_attentions)
        return F.linear(h, self.model.embed_tokens.weight), a

    def init_caches(self, n=18): return [KVCache() for _ in range(n)]
    def clear_caches(self, c):
        for x in c: x.clear()
