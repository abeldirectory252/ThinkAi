"""Gemma 3 decoder for MedGemma 4B. Sliding window + global attention."""
import torch, torch.nn as nn, torch.nn.functional as F
from .gemma_lm import RMSNorm, RotaryEmbedding, apply_rope, KVCache, GemmaMLP


class Gemma3Attention(nn.Module):
    def __init__(self, hidden=2560, heads=8, kv_heads=4, head_dim=256,
                 is_sliding=False, sliding_window=512, softcap=50.0):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.kv_groups = heads // kv_heads
        self.is_sliding, self.sliding_window, self.softcap = is_sliding, sliding_window, softcap
        # Gemma 3 config: query_pre_attn_scalar = head_dim = 256
        # Scale = head_dim ** -0.5 = 0.0625
        self.query_pre_attn_scalar = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        offset = cache.seq_len if cache else 0
        cos, sin = self.rope(L, offset)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache: k, v = cache.update(k, v)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        attn_w = (q @ k.transpose(-2, -1)) * self.query_pre_attn_scalar
        if self.softcap:
            attn_w = self.softcap * torch.tanh(attn_w / self.softcap)
        if mask is not None: attn_w = attn_w + mask
        if self.is_sliding:
            kv_len = attn_w.shape[-1]
            if kv_len > self.sliding_window:
                sw_mask = torch.zeros_like(attn_w, dtype=torch.bool)
                sw_mask[:, :, :, -self.sliding_window:] = True
                attn_w = attn_w.masked_fill(~sw_mask, float("-inf"))
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).type_as(q)
        out = (attn_w @ v).transpose(1, 2).reshape(B, L, -1)
        return (self.o_proj(out), attn_w) if output_attentions else (self.o_proj(out), None)


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, hidden=2560, heads=8, kv_heads=4, head_dim=256,
                 intermediate=10240, eps=1e-6, is_sliding=False,
                 sliding_window=512, softcap=50.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Gemma3Attention(hidden, heads, kv_heads, head_dim,
                                         is_sliding, sliding_window, softcap)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.mlp = GemmaMLP(hidden, intermediate)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        r = x
        x, attn = self.self_attn(self.input_layernorm(x), mask, cache, output_attentions)
        x = r + self.post_attention_layernorm(x)
        r = x
        x = r + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x, attn


class Gemma3Model(nn.Module):
    def __init__(self, vocab=262144, hidden=2560, layers=34, heads=8,
                 kv_heads=4, head_dim=256, intermediate=10240, eps=1e-6,
                 sliding_window=512, global_every=4, softcap=50.0,
                 final_logit_softcap=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.hidden_size = hidden
        self.final_logit_softcap = final_logit_softcap
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(hidden, heads, kv_heads, head_dim, intermediate,
                               eps, is_sliding=((i+1) % global_every != 0),
                               sliding_window=sliding_window, softcap=softcap)
            for i in range(layers)
        ])
        self.norm = RMSNorm(hidden, eps)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None,
                caches=None, output_attentions=False):
        h = self.embed_tokens(input_ids) * (self.hidden_size**0.5) if inputs_embeds is None else inputs_embeds
        all_attn = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            h, attn = layer(h, mask, caches[i] if caches else None, output_attentions)
            if output_attentions: all_attn.append(attn)
        return self.norm(h), all_attn


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.model = Gemma3Model(**kw)
        self.vocab_size = kw.get("vocab", 262144)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None,
                caches=None, output_attentions=False):
        h, attn = self.model(input_ids, inputs_embeds, mask, caches, output_attentions)
        logits = F.linear(h, self.model.embed_tokens.weight)
        # Apply final logit soft-capping (Gemma 2 uses 30.0, Gemma 3 uses None)
        if self.model.final_logit_softcap is not None:
            logits = logits / self.model.final_logit_softcap
            logits = torch.tanh(logits) * self.model.final_logit_softcap
        return logits, attn

    def init_caches(self, num_layers=None): return [KVCache() for _ in self.model.layers]

    def clear_caches(self, c):
        for x in c: x.clear()
