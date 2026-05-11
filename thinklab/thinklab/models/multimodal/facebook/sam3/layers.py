"""
SAM3 primitive layers — key names match HuggingFace exactly.
sam3-layers.py

SAM3 primitive layers — matches HuggingFace exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sam3MLP(nn.Module):
    """Keys: fc1, fc2, dropout."""
    def __init__(self, hidden_size=256, intermediate_size=2048, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Sam3DecoderMLP(nn.Module):
    """Keys: layer1, layer2, [layer3]."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 2:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
        elif num_layers == 3:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2/3 layers supported, got {num_layers}")

    def forward(self, x):
        x = F.relu(self.layer1(x))
        if self.num_layers == 3:
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x


class Sam3Attention(nn.Module):
    """Multi-head attention. Keys: q_proj, k_proj, v_proj, o_proj.

    attention_mask shape conventions accepted:
      - None
      - [B, 1, 1, K]  (broadcast across heads & queries)
      - [B, num_heads, Q, K]
      - [B, 1, Q, K]
    Mask values: 0 for valid, large-negative for masked.
    """
    def __init__(self, hidden_size=256, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key=None, value=None, attention_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        # q,k,v: [B, num_heads, N, head_dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            # Slice to current key length if needed
            attention_mask = attention_mask[..., :Nk]
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.o_proj(out), attn


class Sam3SinePositionEmbedding(nn.Module):
    """2D sine-cosine positional encoding."""
    def __init__(self, num_pos_feats=128, temperature=10000.0,
                 normalize=False, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize must be True if scale is set")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x, y):
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).to(x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, boxes):
        """boxes: [B, Q, 4] in (cx, cy, w, h)."""
        assert boxes.size(-1) == 4
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).to(boxes.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        x_embed = boxes[:, :, 0] * self.scale
        y_embed = boxes[:, :, 1] * self.scale
        w_embed = boxes[:, :, 2] * self.scale
        h_embed = boxes[:, :, 3] * self.scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

    def forward(self, shape, device, dtype, mask=None):
        B, C, H, W = shape
        if mask is None:
            mask = torch.zeros((B, H, W), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Sam3MaskEmbedder(nn.Module):
    """Keys: layers (ModuleList of 3 Linear)."""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x