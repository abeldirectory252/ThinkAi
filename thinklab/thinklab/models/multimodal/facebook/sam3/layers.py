"""
SAM3 primitive layers — key names match HuggingFace exactly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sam3MLP (used in ViT layers, DETR enc/dec, geometry encoder) ──
class Sam3MLP(nn.Module):
    """2-layer MLP with GELU. Keys: fc1, fc2, dropout."""
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 2048,
                 dropout: float = 0.0):
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


# ── Sam3DecoderMLP (used in box_head, presence_head, ref_point_head, rpb) ──
class Sam3DecoderMLP(nn.Module):
    """2 or 3-layer MLP. Keys: layer1, layer2, layer3 (NO ModuleList)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
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
            raise ValueError(f"Only 2 or 3 layers supported, got {num_layers}")

    def forward(self, x):
        x = F.relu(self.layer1(x))
        if self.num_layers == 3:
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x


# ── Sam3Attention (used in DETR enc/dec, geometry encoder, mask decoder) ──
class Sam3Attention(nn.Module):
    """Multi-head attention. Keys: q_proj, k_proj, v_proj, o_proj."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 8):
        super().__init__()
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
        B, N, _ = query.shape
        M = key.shape[1]
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        attn_w = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_w = attn_w + attention_mask
        attn_w = F.softmax(attn_w, dim=-1).to(v.dtype)
        out = (attn_w @ v).transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(out), attn_w


# ── Sam3SinePositionEmbedding ──
class Sam3SinePositionEmbedding(nn.Module):
    """2D sine-cosine positional encoding. Matches HF num_pos_feats convention."""
    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0,
                 normalize: bool = False, scale: float = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def encode_boxes(self, boxes):
        """Encode 4D box coords [B, Q, 4] → [B, Q, num_pos_feats*4].
        HF order: (pos_y, pos_x, pos_w, pos_h)."""
        input_dtype = boxes.dtype
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        def _encode_coord(c):
            c = c.float() * self.scale
            p = c[:, :, None] / dim_t
            return torch.stack((p[:, :, 0::2].sin(), p[:, :, 1::2].cos()), dim=3).flatten(2)

        pos_x = _encode_coord(boxes[:, :, 0])
        pos_y = _encode_coord(boxes[:, :, 1])
        pos_w = _encode_coord(boxes[:, :, 2])
        pos_h = _encode_coord(boxes[:, :, 3])
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2).to(input_dtype)

    def encode_1d_positions(self, x, y):
        input_dtype = x.dtype
        x_embed = x.float() * self.scale
        y_embed = y.float() * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x.to(input_dtype), pos_y.to(input_dtype)

    def forward(self, shape, device, dtype, mask=None):
        """Generate 2D position encoding [B, C, H, W]."""
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
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ── Sam3MaskEmbedder ──
class Sam3MaskEmbedder(nn.Module):
    """3-layer MLP for mask embeddings. Keys: layers (ModuleList)."""
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
