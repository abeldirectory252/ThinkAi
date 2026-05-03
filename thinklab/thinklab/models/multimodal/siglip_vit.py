"""
SigLIP ViT-So400m/14 — Vision encoder for PaliGemma 3B.

Architecture (matches HuggingFace weight keys exactly):
  vision_tower.vision_model.embeddings.{patch_embedding, position_embedding}
  vision_tower.vision_model.encoder.layers.{0-26}.{layer_norm1, self_attn, layer_norm2, mlp}
  vision_tower.vision_model.post_layernorm

Config: hidden=1152, intermediate=4304, layers=27, heads=16,
        image=224, patch=14 → 256 patches.
"""
import math
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

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn_weights
        return out, None


class SigLIPMLP(nn.Module):
    def __init__(self, hidden: int = 1152, intermediate: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden: int = 1152, heads: int = 16, intermediate: int = 4304):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = SigLIPAttention(hidden, heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = SigLIPMLP(hidden, intermediate)

    def forward(
        self, x: torch.Tensor, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.layer_norm1(x)
        x, attn = self.self_attn(x, output_attentions)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x, attn


class SigLIPEncoder(nn.Module):
    def __init__(self, hidden: int = 1152, heads: int = 16,
                 intermediate: int = 4304, num_layers: int = 27):
        super().__init__()
        self.layers = nn.ModuleList([
            SigLIPEncoderLayer(hidden, heads, intermediate)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        all_hidden = [] if output_hidden_states else None
        all_attn = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden.append(x)
            x, attn = layer(x, output_attentions)
            if output_attentions:
                all_attn.append(attn)

        return {
            "last_hidden_state": x,
            "hidden_states": all_hidden,
            "attentions": all_attn,
        }


class SigLIPEmbeddings(nn.Module):
    def __init__(self, hidden: int = 1152, image_size: int = 224,
                 patch_size: int = 14):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 256
        self.patch_embedding = nn.Conv2d(
            3, hidden, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.position_embedding = nn.Embedding(self.num_patches, hidden)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).unsqueeze(0),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, 3, 224, 224)
        patches = self.patch_embedding(pixel_values)        # (B, H, 16, 16)
        patches = patches.flatten(2).transpose(1, 2)        # (B, 256, H)
        embeddings = patches + self.position_embedding(self.position_ids)
        return embeddings


class SigLIPVisionModel(nn.Module):
    """Full SigLIP vision model (matches HF key prefix `vision_model.`)."""

    def __init__(self, hidden: int = 1152, heads: int = 16,
                 intermediate: int = 4304, num_layers: int = 27,
                 image_size: int = 224, patch_size: int = 14):
        super().__init__()
        self.embeddings = SigLIPEmbeddings(hidden, image_size, patch_size)
        self.encoder = SigLIPEncoder(hidden, heads, intermediate, num_layers)
        self.post_layernorm = nn.LayerNorm(hidden)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict:
        x = self.embeddings(pixel_values)
        enc_out = self.encoder(x, output_attentions, output_hidden_states)
        last = self.post_layernorm(enc_out["last_hidden_state"])
        return {
            "last_hidden_state": last,       # (B, 256, 1152)
            "hidden_states": enc_out["hidden_states"],
            "attentions": enc_out["attentions"],
        }


class VisionTower(nn.Module):
    """Wrapper matching HF key prefix `vision_tower.`."""

    def __init__(self, **kwargs):
        super().__init__()
        self.vision_model = SigLIPVisionModel(**kwargs)

    def forward(self, pixel_values: torch.Tensor, **kw) -> dict:
        return self.vision_model(pixel_values, **kw)
