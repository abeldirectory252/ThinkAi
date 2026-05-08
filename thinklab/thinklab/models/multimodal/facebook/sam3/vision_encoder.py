"""
SAM3 Vision Encoder — ViT backbone (RoPE) + FPN Neck.
All key names match HuggingFace exactly.

Fixes vs original:
  1. Sam3ViTRotaryEmbedding.forward() returns raw [seq, dim] tensors (no unsqueeze).
  2. apply_rotary_pos_emb_2d slices cos/sin to actual seq_len before broadcasting.
  3. Sam3ViTModel.__init__ stores self.patch_size.
  4. Sam3ViTModel.forward reshapes THEN layer_norms (matching HF order).
  5. Sam3ViTLayer stores H/W before window_partition for correct unpartition.

thinklab/thinklab/models/multimodal/facebook/sam3/vision_encoder.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Sam3MLP, Sam3SinePositionEmbedding


# ── RoPE helpers ────────────────────────────────────────────────
def rotate_pairwise(x):
    """Pairwise rotation — matches HF exactly."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(q, k, cos, sin):
    """
    Apply 2D RoPE to q and k.

    Args:
        q, k : [B, num_heads, seq_len, head_dim]
        cos, sin : [seq_len_max, head_dim]  — sliced to actual seq_len here.
    """
    seq_len = q.shape[2]
    # Slice to actual sequence length (handles windowed vs global attention)
    cos = cos[:seq_len]   # [seq_len, head_dim]
    sin = sin[:seq_len]   # [seq_len, head_dim]
    # Broadcast: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    q_embed = q.float()
    q_embed = (q_embed * cos) + (rotate_pairwise(q_embed) * sin)
    k_embed = k.float()
    k_embed = (k_embed * cos) + (rotate_pairwise(k_embed) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


# ── ViT Rotary Embedding ───────────────────────────────────────
class Sam3ViTRotaryEmbedding(nn.Module):
    """2D axial RoPE. Buffers: rope_embeddings_cos, rope_embeddings_sin.
    Keys match HF exactly.
    """
    def __init__(self, hidden_size, num_heads, end_x, end_y,
                 rope_theta=10000.0, scale=1.0):
        super().__init__()
        dim = hidden_size // num_heads
        self.end_x, self.end_y = end_x, end_y
        self.dim = dim
        self.rope_theta = rope_theta
        self.scale = scale

        freqs = 1.0 / (rope_theta ** (
            torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))

        flat = torch.arange(end_x * end_y, dtype=torch.long)
        x_pos = (flat % end_x).float() * scale
        y_pos = torch.div(flat, end_x, rounding_mode="floor").float() * scale

        freqs_x = torch.outer(x_pos, freqs).float()
        freqs_y = torch.outer(y_pos, freqs).float()
        inv_freq = torch.cat([freqs_x, freqs_y], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)   # [seq, dim]

        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)

    @torch.no_grad()
    def forward(self):
        # FIX: return [seq_len, head_dim] — NOT [1,1,seq,dim].
        # apply_rotary_pos_emb_2d will slice + broadcast correctly.
        return self.rope_embeddings_cos, self.rope_embeddings_sin


# ── ViT RoPE Attention ─────────────────────────────────────────
class Sam3ViTRoPEAttention(nn.Module):
    """Self-attention with RoPE. Keys: q_proj, k_proj, v_proj, o_proj.
    Input/output: [B, H, W, C] spatial format.
    """
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, position_embeddings):
        B, H, W, _ = hidden_states.shape
        seq_len = H * W
        new_shape = (B, seq_len, self.num_heads, self.head_dim)

        q = self.q_proj(hidden_states).view(*new_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(*new_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*new_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # FIX: apply_rotary_pos_emb_2d now slices cos/sin to seq_len
        q, k = apply_rotary_pos_emb_2d(q, k, cos, sin)

        attn_w = (q @ k.transpose(-2, -1)) * self.scale
        attn_w = F.softmax(attn_w, dim=-1).to(v.dtype)
        out = (attn_w @ v).transpose(1, 2).reshape(B, H, W, -1).contiguous()
        return self.o_proj(out), attn_w


# ── Window partition/unpartition ────────────────────────────────
def window_partition(hidden_state, window_size):
    """[B, H, W, C] → [B*nW, ws, ws, C], (Hp, Wp)."""
    B, H, W, C = hidden_state.shape
    pH = (window_size - H % window_size) % window_size
    pW = (window_size - W % window_size) % window_size
    hidden_state = F.pad(hidden_state, (0, 0, 0, pW, 0, pH))
    Hp, Wp = H + pH, W + pW
    hidden_state = hidden_state.view(
        B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, orig_hw):
    """[B*nW, ws, ws, C] → [B, H, W, C]."""
    Hp, Wp = pad_hw
    H, W = orig_hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


# ── Patch Embeddings ────────────────────────────────────────────
class Sam3ViTPatchEmbeddings(nn.Module):
    """Keys: projection (Conv2d, bias=False)."""
    def __init__(self, hidden_size=1024, patch_size=14, num_channels=3,
                 pretrain_image_size=1008):
        super().__init__()
        ps = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        ims = pretrain_image_size if isinstance(pretrain_image_size, (list, tuple)) \
            else (pretrain_image_size, pretrain_image_size)
        self.num_patches = (ims[0] // ps[0]) * (ims[1] // ps[1])
        self.patch_size = ps
        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=ps, stride=ps, bias=False)

    def forward(self, pixel_values):
        return self.projection(
            pixel_values.to(self.projection.weight.dtype)
        ).flatten(2).transpose(1, 2)


# ── ViT Embeddings ──────────────────────────────────────────────
class Sam3ViTEmbeddings(nn.Module):
    """Keys: patch_embeddings, position_embeddings, dropout."""
    def __init__(self, hidden_size=1024, patch_size=14,
                 pretrain_image_size=1008, num_channels=3, dropout=0.0):
        super().__init__()
        self.patch_embeddings = Sam3ViTPatchEmbeddings(
            hidden_size, patch_size, num_channels, pretrain_image_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]

    def _tile_position_embeddings(self, pos_embed, height, width):
        pretrain_size = int(pos_embed.shape[1] ** 0.5)
        if pretrain_size == height and pretrain_size == width:
            return pos_embed.reshape(1, height * width, -1)
        C = pos_embed.shape[-1]
        pe = pos_embed.reshape(1, pretrain_size, pretrain_size, C).permute(0, 3, 1, 2)
        rh = height // pretrain_size + 1
        rw = width // pretrain_size + 1
        pe = pe.tile([1, 1, rh, rw])[:, :, :height, :width]
        return pe.permute(0, 2, 3, 1).reshape(1, height * width, C)

    def forward(self, pixel_values):
        H, W = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)
        hp = H // self.patch_size
        wp = W // self.patch_size
        pos = self._tile_position_embeddings(self.position_embeddings, hp, wp)
        return self.dropout(embeddings + pos)


# ── ViT Layer ───────────────────────────────────────────────────
class Sam3ViTLayer(nn.Module):
    """Keys: layer_norm1, rotary_emb, attention, layer_norm2, mlp, dropout."""
    def __init__(self, hidden_size=1024, num_heads=16, intermediate_size=4096,
                 image_size=1008, patch_size=14, window_size=0,
                 global_window_size=8, layer_norm_eps=1e-6,
                 dropout=0.0, rope_theta=10000.0):
        super().__init__()
        ims = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        ps = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        input_size = (ims[0] // ps[0], ims[1] // ps[1])

        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # RoPE grid: windowed layers use (ws,ws), global layers use full input_size
        rotary_input_size = (window_size, window_size) if window_size > 0 else input_size
        # Scale so that global layers have rotary_scale=1, windowed layers scale up
        rotary_scale = global_window_size / rotary_input_size[0]

        self.rotary_emb = Sam3ViTRotaryEmbedding(
            hidden_size, num_heads,
            end_x=rotary_input_size[0], end_y=rotary_input_size[1],
            rope_theta=rope_theta, scale=rotary_scale)

        self.attention = Sam3ViTRoPEAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Sam3MLP(hidden_size, intermediate_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            # FIX: save H,W BEFORE partition for correct unpartition
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = window_partition(hidden_states, self.window_size)

        pos = self.rotary_emb()   # [seq, head_dim] — no extra dims
        hidden_states, _ = self.attention(hidden_states, pos)

        if self.window_size > 0:
            hidden_states = window_unpartition(
                hidden_states, self.window_size, pad_hw, (H, W))

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


# ── ViT Backbone ────────────────────────────────────────────────
class Sam3ViTModel(nn.Module):
    """Keys: embeddings, layer_norm, layers."""
    def __init__(self, hidden_size=1024, num_heads=16, intermediate_size=4096,
                 num_layers=32, image_size=1008, patch_size=14,
                 pretrain_image_size=1008, window_size=8,
                 global_attn_indexes=None, layer_norm_eps=1e-6,
                 dropout=0.0, rope_theta=10000.0):
        super().__init__()
        if global_attn_indexes is None:
            global_attn_indexes = []

        # FIX: store patch_size so forward() can compute H, W correctly
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]

        self.embeddings = Sam3ViTEmbeddings(
            hidden_size, patch_size, pretrain_image_size, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layers = nn.ModuleList([
            Sam3ViTLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                image_size=image_size,
                patch_size=patch_size,
                # Global attention layers get window_size=0
                window_size=window_size if i not in global_attn_indexes else 0,
                global_window_size=window_size,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout,
                rope_theta=rope_theta,
            )
            for i in range(num_layers)
        ])

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)       # [B, N, C]
        B = x.shape[0]
        # FIX: use self.patch_size (now correctly stored)
        H = pixel_values.shape[-2] // self.patch_size
        W = pixel_values.shape[-1] // self.patch_size
        C = x.shape[-1]

        # Reshape to spatial for windowed attention: [B, H, W, C]
        x = x.view(B, H, W, C)
        # FIX: layer_norm AFTER reshape to spatial (matches HF)
        x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        # Back to sequence: [B, H*W, C]
        x = x.view(B, H * W, C)
        return {"last_hidden_state": x}


# ── FPN Layer ───────────────────────────────────────────────────
class Sam3FPNLayer(nn.Module):
    """Keys: scale_layers, proj1, proj2."""
    def __init__(self, in_channels, fpn_dim, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_layers = nn.ModuleList()

        if scale_factor == 4.0:
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            intermediate = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            intermediate = in_channels // 2
        elif scale_factor == 1.0:
            intermediate = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate = in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor}")

        self.proj1 = nn.Conv2d(intermediate, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.to(self.proj1.weight.dtype)
        for layer in self.scale_layers:
            x = layer(x)
        x = self.proj1(x)
        x = self.proj2(x)
        return x


# ── Vision Neck ─────────────────────────────────────────────────
class Sam3VisionNeck(nn.Module):
    """Keys: position_encoding, fpn_layers."""
    def __init__(self, backbone_hidden_size=1024, fpn_hidden_size=256,
                 scale_factors=None):
        super().__init__()
        if scale_factors is None:
            scale_factors = [4.0, 2.0, 1.0, 0.5]
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=fpn_hidden_size // 2, normalize=True)
        self.fpn_layers = nn.ModuleList([
            Sam3FPNLayer(backbone_hidden_size, fpn_hidden_size, s)
            for s in scale_factors
        ])

    def forward(self, hidden_states):
        """hidden_states: [B, C, H, W] from backbone."""
        fpn_hidden_states = ()
        fpn_position_encoding = ()
        for fpn in self.fpn_layers:
            out = fpn(hidden_states)
            fpn_hidden_states += (out,)
            pos = self.position_encoding(out.shape, out.device, out.dtype)
            fpn_position_encoding += (pos,)
        return fpn_hidden_states, fpn_position_encoding


# ── Full Vision Encoder ─────────────────────────────────────────
class Sam3VisionModel(nn.Module):
    """Keys: backbone, neck."""
    def __init__(self, hidden_size=1024, num_heads=16, intermediate_size=4096,
                 num_layers=32, image_size=1008, patch_size=14,
                 pretrain_image_size=1008, fpn_hidden_size=256,
                 window_size=8, global_attn_indexes=None,
                 scale_factors=None, layer_norm_eps=1e-6,
                 dropout=0.0, rope_theta=10000.0):
        super().__init__()
        self.backbone = Sam3ViTModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            image_size=image_size,
            patch_size=patch_size,
            pretrain_image_size=pretrain_image_size,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
            rope_theta=rope_theta,
        )
        self.neck = Sam3VisionNeck(hidden_size, fpn_hidden_size, scale_factors)
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]

    def forward(self, pixel_values):
        backbone_out = self.backbone(pixel_values)
        last_hidden = backbone_out["last_hidden_state"]  # [B, N, C]
        B = last_hidden.shape[0]
        H = pixel_values.shape[-2] // self.patch_size
        W = pixel_values.shape[-1] // self.patch_size
        # Reshape to spatial [B, C, H, W] for FPN
        spatial = last_hidden.view(B, H, W, -1).permute(0, 3, 1, 2)
        fpn_hidden, fpn_pos = self.neck(spatial)
        return {
            "last_hidden_state": last_hidden,
            "fpn_hidden_states": fpn_hidden,
            "fpn_position_encoding": fpn_pos,
        }