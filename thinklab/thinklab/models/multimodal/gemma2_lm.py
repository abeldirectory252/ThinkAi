"""
Gemma 2 decoder for PaliGemma2 / MedGemma 4B.

Key differences from Gemma 1:
  - Sliding window + global attention (alternating every 2nd layer)
  - Attention logit soft-capping (50.0)
  - Final logit soft-capping (30.0)
  - Post-attention norm + post-FFW norm
  - query_pre_attn_scalar = embed_dim // num_heads (NOT 1/sqrt(head_dim))

Key differences from Gemma 3:
  - NO QK-norm (Gemma 3 adds QK-norm, Gemma 2 does not)
  - global_every = 2 (Gemma 3 uses 4)
  - query_pre_attn_scalar = embed_dim // num_heads (raw value, not inverted sqrt)
  - final_logit_softcap = 30.0 (Gemma 3 = None)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gemma_lm import RMSNorm, RotaryEmbedding, apply_rope, KVCache, GemmaMLP


class Gemma2Attention(nn.Module):
    def __init__(self, hidden=2304, heads=8, kv_heads=4, head_dim=256,
                 is_sliding=False, sliding_window=4096, softcap=50.0):
        super().__init__()
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_groups = heads // kv_heads
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.softcap = softcap

        # Gemma 2: query_pre_attn_scalar = (embed_dim // num_heads) ** -0.5
        # This is the inverse sqrt, matching HF's implementation
        self.query_pre_attn_scalar = (hidden // heads) ** -0.5

        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.rope = RotaryEmbedding(head_dim)
        # NOTE: Gemma 2 does NOT use QK-norm (that's a Gemma 3 feature)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # No QK-norm for Gemma 2

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

        # Gemma 2: scale attention scores by the pre_attn_scalar
        attn_w = (q @ k.transpose(-2, -1)) * self.query_pre_attn_scalar

        # Attention logit soft-capping
        if self.softcap > 0:
            attn_w = self.softcap * torch.tanh(attn_w / self.softcap)

        if mask is not None:
            attn_w = attn_w + mask

        # Sliding window masking
        if self.is_sliding:
            kv_len = attn_w.shape[-1]
            if kv_len > self.sliding_window:
                sw_mask = torch.zeros_like(attn_w, dtype=torch.bool)
                sw_mask[:, :, :, -self.sliding_window:] = True
                attn_w = attn_w.masked_fill(~sw_mask, float("-inf"))

        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).type_as(q)
        out = (attn_w @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.o_proj(out)

        if output_attentions:
            return out, attn_w
        return out, None


class Gemma2DecoderLayer(nn.Module):
    """Gemma 2 decoder layer with post-attn and post-FFW norms.

    Norm ordering (matches reference Block.__call__):
      1. pre_attention_norm(x)
      2. attn(...)
      3. post_attention_norm(attn_output)  ← norm BEFORE residual
      4. += x                              ← residual add
      5. pre_feedforward_norm(...)
      6. mlp(...)
      7. post_feedforward_norm(mlp_output)  ← norm BEFORE residual
      8. += attn_output                    ← residual add
    """

    def __init__(self, hidden=2304, heads=8, kv_heads=4, head_dim=256,
                 intermediate=9216, eps=1e-6, is_sliding=False,
                 sliding_window=4096, softcap=50.0):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Gemma2Attention(
            hidden, heads, kv_heads, head_dim,
            is_sliding, sliding_window, softcap,
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.mlp = GemmaMLP(hidden, intermediate)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)

    def forward(self, x, mask=None, cache=None, output_attentions=False):
        r = x
        x, attn = self.self_attn(
            self.input_layernorm(x), mask, cache, output_attentions
        )
        # Gemma 2: post_attention_norm applied to attn output BEFORE residual
        x = r + self.post_attention_layernorm(x)
        r = x
        x = r + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        return x, attn


class Gemma2Model(nn.Module):
    def __init__(self, vocab=262144, hidden=2304, layers=26, heads=8,
                 kv_heads=4, head_dim=256, intermediate=9216, eps=1e-6,
                 sliding_window=4096, global_every=2, softcap=50.0,
                 final_logit_softcap=30.0):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.hidden_size = hidden
        self.final_logit_softcap = final_logit_softcap
        self.layers = nn.ModuleList([
            Gemma2DecoderLayer(
                hidden, heads, kv_heads, head_dim, intermediate, eps,
                is_sliding=((i + 1) % global_every != 0),
                sliding_window=sliding_window, softcap=softcap,
            )
            for i in range(layers)
        ])
        self.norm = RMSNorm(hidden, eps)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None,
                caches=None, output_attentions=False):
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids) * (self.hidden_size ** 0.5)
        else:
            # inputs_embeds already scaled by caller (PaliGemma)
            h = inputs_embeds

        all_attn = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            h, attn = layer(h, mask, caches[i] if caches else None,
                            output_attentions)
            if output_attentions:
                all_attn.append(attn)

        return self.norm(h), all_attn


class Gemma2ForCausalLM(nn.Module):
    """Gemma 2 causal LM wrapper. Weight keys match HF prefix `language_model.`."""

    def __init__(self, **kw):
        super().__init__()
        self.model = Gemma2Model(**kw)
        self.vocab_size = kw.get("vocab", 262144)

    def forward(self, input_ids=None, inputs_embeds=None, mask=None,
                caches=None, output_attentions=False):
        h, attn = self.model(input_ids, inputs_embeds, mask, caches,
                             output_attentions)
        logits = F.linear(h, self.model.embed_tokens.weight)
        # Final logit soft-capping (Gemma 2 = 30.0)
        if self.model.final_logit_softcap is not None:
            logits = torch.tanh(logits / self.model.final_logit_softcap) \
                     * self.model.final_logit_softcap
        return logits, attn

    def init_caches(self, num_layers=None):
        return [KVCache() for _ in self.model.layers]

    def clear_caches(self, c):
        for x in c:
            x.clear()
