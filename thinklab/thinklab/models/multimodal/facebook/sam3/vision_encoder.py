"""
SAM3 Vision Encoder: ViT backbone (32 layers) + FPN Neck.
Input:  [B, 3, 1008, 1008]
Output: Multi-scale features via FPN Neck
        ([B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72], [B, 256, 36, 36])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Sam3SinePositionEmbedding


# ── ViT Patch Embeddings ────────────────────────────────────────
class Sam3ViTPatchEmbeddings(nn.Module):
    def __init__(self, hidden: int = 1024, image_size: int = 1008, patch_size: int = 14):
        super().__init__()
        self.projection = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        x = self.projection(pixel_values)  # [B, C, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, N, C]


class Sam3ViTEmbeddings(nn.Module):
    def __init__(self, hidden: int = 1024, image_size: int = 1008, patch_size: int = 14):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 5184
        self.patch_embeddings = Sam3ViTPatchEmbeddings(hidden, image_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, hidden))
        self.dropout = nn.Dropout(0.0)

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        x = x + self.position_embeddings
        return self.dropout(x)


# ── ViT Attention (windowed) ────────────────────────────────────
class Sam3ViTAttention(nn.Module):
    def __init__(self, hidden: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Sam3ViTMLP(nn.Module):
    def __init__(self, hidden: int = 1024, intermediate: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Sam3ViTLayer(nn.Module):
    """Single ViT encoder layer with window attention."""
    def __init__(self, hidden: int = 1024, num_heads: int = 16,
                 intermediate: int = 4096, window_size: int = 8):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(hidden)
        self.attention = Sam3ViTAttention(hidden, num_heads)
        self.layernorm_after = nn.LayerNorm(hidden)
        self.mlp = Sam3ViTMLP(hidden, intermediate)
        self.window_size = window_size

    def _window_partition(self, x, H, W):
        """Partition spatial features into non-overlapping windows."""
        B, N, C = x.shape
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            # Pad if needed
            pH = (ws - H % ws) % ws
            pW = (ws - W % ws) % ws
            x = x.reshape(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pW, 0, pH))
            H, W = H + pH, W + pW
            x = x.reshape(B, H * W, C)
        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        return windows, H, W

    def _window_unpartition(self, windows, H, W, orig_H, orig_W):
        B_windows = windows.shape[0]
        ws = self.window_size
        B = B_windows // ((H // ws) * (W // ws))
        x = windows.reshape(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
        if H != orig_H or W != orig_W:
            x = x.reshape(B, H, W, -1)[:, :orig_H, :orig_W, :].reshape(B, orig_H * orig_W, -1)
        return x

    def forward(self, x, spatial_shape=None):
        H = W = int(x.shape[1] ** 0.5) if spatial_shape is None else spatial_shape[0]
        if spatial_shape is not None:
            H, W = spatial_shape

        residual = x
        x = self.layernorm_before(x)
        # Window partition for efficient attention
        x_win, pH, pW = self._window_partition(x, H, W)
        x_win = self.attention(x_win)
        x = self._window_unpartition(x_win, pH, pW, H, W)
        x = residual + x

        residual = x
        x = self.layernorm_after(x)
        x = self.mlp(x)
        x = residual + x
        return x


# ── ViT Backbone ────────────────────────────────────────────────
class Sam3ViTModel(nn.Module):
    """ViT-Large backbone: 32 layers, hidden=1024, image=1008, patch=14."""
    def __init__(self, hidden: int = 1024, num_heads: int = 16,
                 intermediate: int = 4096, num_layers: int = 32,
                 image_size: int = 1008, patch_size: int = 14,
                 window_size: int = 8):
        super().__init__()
        self.embeddings = Sam3ViTEmbeddings(hidden, image_size, patch_size)
        self.layers = nn.ModuleList([
            Sam3ViTLayer(hidden, num_heads, intermediate, window_size)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden)
        self.grid_size = image_size // patch_size  # 72

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)  # [B, 5184, 1024]
        H = W = self.grid_size
        for layer in self.layers:
            x = layer(x, spatial_shape=(H, W))
        # Reshape to spatial for FPN
        x_spatial = self.layer_norm(x.reshape(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2))
        return {"last_hidden_state": x, "spatial": x_spatial}


# ── FPN Neck ────────────────────────────────────────────────────
class Sam3FPNLayer(nn.Module):
    """Single FPN layer: project + upsample to target scale."""
    def __init__(self, in_channels: int = 1024, out_channels: int = 256,
                 scale_factor: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        if scale_factor > 1:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 2,
                                                stride=2) if scale_factor <= 2 else None
            # For larger scale factors, use sequential upsampling
            if scale_factor > 2:
                layers = []
                for _ in range(int(math.log2(scale_factor))):
                    layers.extend([
                        nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
                    ])
                self.upsample = nn.Sequential(*layers)
        else:
            self.upsample = None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv2(F.gelu(x))
        return x


import math

class Sam3VisionNeck(nn.Module):
    """FPN Neck producing multi-scale features from ViT backbone output.
    Outputs 4 feature maps at different scales:
      [B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72], [B, 256, 36, 36]
    """
    def __init__(self, in_channels: int = 1024, out_channels: int = 256,
                 grid_size: int = 72):
        super().__init__()
        self.position_encoding = Sam3SinePositionEmbedding(out_channels)
        # 4 FPN layers at different scales: 4x, 2x, 1x, 0.5x relative to grid
        self.fpn_layers = nn.ModuleList([
            Sam3FPNLayer(in_channels, out_channels, scale_factor=4),   # → 288
            Sam3FPNLayer(in_channels, out_channels, scale_factor=2),   # → 144
            Sam3FPNLayer(in_channels, out_channels, scale_factor=1),   # → 72
            Sam3FPNLayer(in_channels, out_channels, scale_factor=1),   # → 36 (with pool)
        ])
        self.grid_size = grid_size

    def forward(self, backbone_features):
        """backbone_features: [B, C, H, W] from ViT spatial output."""
        features = []
        for i, fpn in enumerate(self.fpn_layers):
            feat = fpn(backbone_features)
            if i == 3:
                # Downsample for smallest scale
                feat = F.avg_pool2d(feat, 2)
            features.append(feat)
        return tuple(features)


# ── Top-level Vision Encoder ────────────────────────────────────
class Sam3VisionModel(nn.Module):
    """Full vision encoder: ViT backbone + FPN Neck."""
    def __init__(self, hidden: int = 1024, num_heads: int = 16,
                 intermediate: int = 4096, num_layers: int = 32,
                 image_size: int = 1008, patch_size: int = 14,
                 out_channels: int = 256, window_size: int = 8):
        super().__init__()
        self.backbone = Sam3ViTModel(hidden, num_heads, intermediate, num_layers,
                                      image_size, patch_size, window_size)
        grid = image_size // patch_size
        self.neck = Sam3VisionNeck(hidden, out_channels, grid)

    def forward(self, pixel_values):
        backbone_out = self.backbone(pixel_values)
        multi_scale = self.neck(backbone_out["spatial"])
        return {
            "last_hidden_state": backbone_out["last_hidden_state"],
            "multi_scale_features": multi_scale,
        }
