"""
SAM3 primitive layers. Self-contained.
Core: Sam3Attention, Sam3MLP, Sam3SinePositionEmbedding, Sam3DecoderMLP.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sam3Attention(nn.Module):
    """Multi-head attention used throughout SAM3 (DETR encoder/decoder, geometry encoder, mask decoder)."""
    def __init__(self, hidden: int = 256, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)

    def forward(self, query, key=None, value=None, attn_mask=None):
        if key is None: key = query
        if value is None: value = key
        B, N, _ = query.shape
        M = key.shape[1]
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        attn_w = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_w = attn_w + attn_mask
        attn_w = F.softmax(attn_w, dim=-1)
        out = (attn_w @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(out),


class Sam3MLP(nn.Module):
    """2-layer MLP with GELU. Used in DETR layers and scoring."""
    def __init__(self, hidden: int = 256, intermediate: int = 2048):
        super().__init__()
        self.layer1 = nn.Linear(hidden, intermediate)
        self.layer2 = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x)))


class Sam3DecoderMLP(nn.Module):
    """Multi-layer MLP used in box/presence/ref_point heads (2 or 3 layers)."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            name = f"layer{i+1}"
            setattr(self, name, nn.Linear(dims[i], dims[i+1]))
            self.layers.append(getattr(self, name))
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class Sam3SinePositionEmbedding(nn.Module):
    """2D sine-cosine positional encoding for spatial features."""
    def __init__(self, hidden: int = 256, temperature: float = 10000.0):
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature

    def forward(self, spatial_shape=None, device=None, dtype=None):
        if spatial_shape is None:
            spatial_shape = (36, 36)
        H, W = spatial_shape if isinstance(spatial_shape, (tuple, list)) else (spatial_shape, spatial_shape)
        half = self.hidden // 2
        dim_t = torch.arange(half, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half)

        y_pos = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
        y_embed = y_pos.unsqueeze(-1) / dim_t
        x_embed = x_pos.unsqueeze(-1) / dim_t
        pe = torch.cat([y_embed.sin(), y_embed.cos(), x_embed.sin(), x_embed.cos()], dim=-1)
        return pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]


class Sam3MaskEmbedder(nn.Module):
    """3-layer MLP for mask embedding projection."""
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
