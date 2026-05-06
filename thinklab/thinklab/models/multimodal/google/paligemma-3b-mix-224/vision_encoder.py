"""SigLIP ViT for PaliGemma (224px). Self-contained."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPAttention(nn.Module):
    def __init__(self, hidden=1152, heads=16):
        super().__init__()
        self.num_heads, self.head_dim = heads, hidden // heads
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
            aw = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
            out = (aw @ v).transpose(1, 2).reshape(B, N, C)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, N, C)
            aw = None
        return self.out_proj(out), aw


class SigLIPMLP(nn.Module):
    def __init__(self, hidden=1152, intermediate=4304):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(hidden, intermediate), nn.Linear(intermediate, hidden)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304):
        super().__init__()
        self.layer_norm1, self.self_attn = nn.LayerNorm(hidden), SigLIPAttention(hidden, heads)
        self.layer_norm2, self.mlp = nn.LayerNorm(hidden), SigLIPMLP(hidden, intermediate)
    def forward(self, x, output_attentions=False):
        r = x; x, a = self.self_attn(self.layer_norm1(x), output_attentions); x = r + x
        return x + self.mlp(self.layer_norm2(x)), a


class SigLIPEncoder(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304, num_layers=27):
        super().__init__()
        self.layers = nn.ModuleList([SigLIPEncoderLayer(hidden, heads, intermediate) for _ in range(num_layers)])
    def forward(self, x, output_attentions=False, output_hidden_states=False):
        ah, aa = ([] if output_hidden_states else None), ([] if output_attentions else None)
        for l in self.layers:
            if ah is not None: ah.append(x)
            x, a = l(x, output_attentions)
            if aa is not None: aa.append(a)
        return {"last_hidden_state": x, "hidden_states": ah, "attentions": aa}


class SigLIPEmbeddings(nn.Module):
    def __init__(self, hidden=1152, image_size=224, patch_size=14):
        super().__init__()
        np = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size, bias=True)
        self.position_embedding = nn.Embedding(np, hidden)
        self.register_buffer("position_ids", torch.arange(np).unsqueeze(0), persistent=False)
    def forward(self, pv):
        return self.patch_embedding(pv).flatten(2).transpose(1, 2) + self.position_embedding(self.position_ids)


class SigLIPVisionModel(nn.Module):
    def __init__(self, hidden=1152, heads=16, intermediate=4304, num_layers=27, image_size=224, patch_size=14):
        super().__init__()
        self.embeddings = SigLIPEmbeddings(hidden, image_size, patch_size)
        self.encoder = SigLIPEncoder(hidden, heads, intermediate, num_layers)
        self.post_layernorm = nn.LayerNorm(hidden)
    def forward(self, pv, output_attentions=False, output_hidden_states=False):
        e = self.encoder(self.embeddings(pv), output_attentions, output_hidden_states)
        return {"last_hidden_state": self.post_layernorm(e["last_hidden_state"]),
                "hidden_states": e["hidden_states"], "attentions": e["attentions"]}


class VisionTower(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.vision_model = SigLIPVisionModel(**kw)
    def forward(self, pv, **kw): return self.vision_model(pv, **kw)
