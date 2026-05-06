"""
SigLIP ViT-So400m/14 — Vision encoder for MedGemma 4B.
Config: hidden=1152, intermediate=4304, layers=27, heads=16,
        image=896, patch=14 → 4096 patches.
Fully self-contained — no cross-model dependencies.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPAttention(nn.Module):
    def __init__(self, hidden: int = 1152, heads: int = 16):
        super().__init__()
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x, output_attentions=False):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if output_attentions:
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out = out.transpose(1, 2).reshape(B, N, C)
            attn_weights = None
        return self.out_proj(out), attn_weights


class SigLIPMLP(nn.Module):
    def __init__(self, hidden: int = 1152, intermediate: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = SigLIPAttention(hidden, heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = SigLIPMLP(hidden, intermediate)

    def forward(self, x, output_attentions=False):
        residual = x
        x, attn = self.self_attn(self.layer_norm1(x), output_attentions)
        x = residual + x
        residual = x
        x = residual + self.mlp(self.layer_norm2(x))
        return x, attn


class SigLIPEncoder(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304, num_layers=27):
        super().__init__()
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(hidden, heads, intermediate) for _ in range(num_layers)
        ])

    def forward(self, x, output_attentions=False, output_hidden_states=False):
        all_hidden = [] if output_hidden_states else None
        all_attn = [] if output_attentions else None
        for layer in self.layers:
            if output_hidden_states: all_hidden.append(x)
            x, attn = layer(x, output_attentions)
            if output_attentions: all_attn.append(attn)
        return {"last_hidden_state": x, "hidden_states": all_hidden, "attentions": all_attn}


class SigLIPEmbeddings(nn.Module):
    def __init__(self, hidden=1152, image_size=896, patch_size=14):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size, bias=True)
        self.position_embedding = nn.Embedding(self.num_patches, hidden)
        self.register_buffer("position_ids", torch.arange(self.num_patches).unsqueeze(0), persistent=False)

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids)


class SigLIPVisionModel(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304,
                 num_layers=27, image_size=896, patch_size=14):
        super().__init__()
        self.embeddings = SigLIPEmbeddings(hidden, image_size, patch_size)
        self.encoder = SigLIPEncoder(hidden, heads, intermediate, num_layers)
        self.post_layernorm = nn.LayerNorm(hidden)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False):
        x = self.embeddings(pixel_values)
        enc_out = self.encoder(x, output_attentions, output_hidden_states)
        last = self.post_layernorm(enc_out["last_hidden_state"])
        return {"last_hidden_state": last, "hidden_states": enc_out["hidden_states"], "attentions": enc_out["attentions"]}


class VisionTower(nn.Module):
    """Wrapper matching HF key prefix `vision_tower.`."""
    def __init__(self, **kwargs):
        super().__init__()
        self.vision_model = SigLIPVisionModel(**kwargs)

    def forward(self, pixel_values, **kw):
        return self.vision_model(pixel_values, **kw)
