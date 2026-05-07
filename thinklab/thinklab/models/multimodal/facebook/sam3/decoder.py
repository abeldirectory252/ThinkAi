"""
SAM3 Decoder: DETR Encoder, DETR Decoder, Geometry Encoder, Mask Decoder.
These form the detection + segmentation head of SAM3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Sam3Attention, Sam3MLP, Sam3DecoderMLP, Sam3SinePositionEmbedding, Sam3MaskEmbedder


# ── DETR Encoder ────────────────────────────────────────────────
class Sam3DetrEncoderLayer(nn.Module):
    """DETR encoder layer: self_attn + cross_attn + MLP."""
    def __init__(self, hidden: int = 256, num_heads: int = 8,
                 intermediate: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.cross_attn = Sam3Attention(hidden, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Sam3MLP(hidden, intermediate)
        self.layer_norm3 = nn.LayerNorm(hidden)

    def forward(self, x, pos_embed=None, attn_mask=None):
        # Self-attention with positional encoding
        q = k = self.layer_norm1(x)
        if pos_embed is not None:
            q = q + pos_embed
            k = k + pos_embed
        attn_out = self.self_attn(q, k, self.layer_norm1(x))[0]
        x = x + self.dropout(attn_out)
        # MLP
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.mlp(x)
        x = self.layer_norm3(x)
        return x


class Sam3DetrEncoder(nn.Module):
    """6-layer DETR encoder processing vision features."""
    def __init__(self, hidden: int = 256, num_heads: int = 8,
                 intermediate: int = 2048, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            Sam3DetrEncoderLayer(hidden, num_heads, intermediate)
            for _ in range(num_layers)
        ])

    def forward(self, vision_features, pos_embeds=None):
        x = vision_features
        for layer in self.layers:
            x = layer(x, pos_embeds)
        return {"last_hidden_state": x, "pos_embeds_flattened": pos_embeds}


# ── DETR Decoder ────────────────────────────────────────────────
class Sam3DetrDecoderLayer(nn.Module):
    """DETR decoder layer: self_attn → text_cross_attn → vision_cross_attn → MLP."""
    def __init__(self, hidden: int = 256, num_heads: int = 8,
                 intermediate: int = 2048, dropout: float = 0.0):
        super().__init__()
        # Self-attention
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        # Text cross-attention
        self.text_cross_attn = Sam3Attention(hidden, num_heads)
        self.text_cross_attn_dropout = nn.Dropout(dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden)
        # Vision cross-attention
        self.vision_cross_attn = Sam3Attention(hidden, num_heads)
        self.vision_cross_attn_dropout = nn.Dropout(dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden)
        # MLP
        self.mlp = Sam3MLP(hidden, intermediate)
        self.mlp_layer_norm = nn.LayerNorm(hidden)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, vision_features, text_features,
                vision_pos=None, rpb=None):
        # Self-attention
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        x = residual + self.self_attn_dropout(self.self_attn(x_norm)[0])

        # Text cross-attention
        residual = x
        x_norm = self.text_cross_attn_layer_norm(x)
        x = residual + self.text_cross_attn_dropout(
            self.text_cross_attn(x_norm, text_features, text_features)[0])

        # Vision cross-attention
        residual = x
        x_norm = self.vision_cross_attn_layer_norm(x)
        x = residual + self.vision_cross_attn_dropout(
            self.vision_cross_attn(x_norm, vision_features, vision_features, attn_mask=rpb)[0])

        # MLP
        residual = x
        x_norm = self.mlp_layer_norm(x)
        x = residual + self.mlp_dropout(self.mlp(x_norm))
        return x


class Sam3DetrDecoder(nn.Module):
    """6-layer DETR decoder with query-based detection + mask prediction heads."""
    def __init__(self, hidden: int = 256, num_heads: int = 8,
                 intermediate: int = 2048, num_layers: int = 6,
                 num_queries: int = 200):
        super().__init__()
        self.layers = nn.ModuleList([
            Sam3DetrDecoderLayer(hidden, num_heads, intermediate)
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden)

        # Detection heads
        self.box_head = Sam3DecoderMLP(hidden, hidden, 4, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden)
        self.reference_points = nn.Embedding(num_queries, 4)

        # Presence detection
        self.presence_token = nn.Embedding(1, hidden)
        self.presence_head = Sam3DecoderMLP(hidden, hidden, 1, num_layers=3)
        self.presence_layer_norm = nn.LayerNorm(hidden)

        # Reference point refinement
        self.ref_point_head = Sam3DecoderMLP(hidden * 2, hidden, hidden, num_layers=2)

        # Box relative position bias
        self.box_rpb_embed_x = Sam3DecoderMLP(2, hidden, num_heads, num_layers=2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, hidden, num_heads, num_layers=2)

        self.position_encoding = Sam3SinePositionEmbedding(hidden)
        self.num_queries = num_queries
        self.num_layers = len(self.layers)

    def forward(self, vision_features, text_features, vision_pos=None):
        B = vision_features.shape[0]
        dev = vision_features.device

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_pts = self.reference_points.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)
        presence = self.presence_token.weight.unsqueeze(0).expand(B, -1, -1)

        # Prepend presence token
        x = torch.cat([presence, queries], dim=1)  # [B, 201, 256]

        intermediates = []
        for layer in self.layers:
            x = layer(x, vision_features, text_features, vision_pos)
            intermediates.append(x[:, 1:, :])  # Skip presence token

        # Final outputs
        x_final = self.output_layer_norm(x[:, 1:, :])  # [B, 200, 256]
        presence_out = self.presence_layer_norm(x[:, :1, :])

        # Intermediate stack
        inter_stack = torch.stack(intermediates, dim=0)  # [6, B, 200, 256]

        # Box predictions from all layers
        boxes = self.box_head(inter_stack).sigmoid()  # [6, B, 200, 4]

        # Presence prediction
        presence_logit = self.presence_head(presence_out)  # [B, 1, 1]

        return {
            "intermediate_hidden_states": inter_stack,
            "reference_points": ref_pts,
            "last_hidden_state": x_final,
            "pred_boxes": boxes,
            "presence_logit": presence_logit,
        }


# ── Geometry Encoder ────────────────────────────────────────────
class Sam3GeometryEncoderLayer(nn.Module):
    def __init__(self, hidden: int = 256, num_heads: int = 8,
                 intermediate: int = 2048):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.dropout = nn.Dropout(0.0)
        self.cross_attn = Sam3Attention(hidden, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Sam3MLP(hidden, intermediate)
        self.layer_norm3 = nn.LayerNorm(hidden)

    def forward(self, x, vision_features):
        residual = x
        x = self.layer_norm1(x)
        x = residual + self.self_attn(x)[0]
        residual = x
        x = residual + self.cross_attn(x, vision_features, vision_features)[0]
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.mlp(x)
        return self.layer_norm3(x)


class Sam3GeometryEncoder(nn.Module):
    """Encodes geometric prompts (boxes, points) for conditioning the decoder."""
    def __init__(self, hidden: int = 256, num_layers: int = 3):
        super().__init__()
        self.position_encoding = Sam3SinePositionEmbedding(hidden)
        self.label_embed = nn.Embedding(2, hidden)
        self.cls_embed = nn.Embedding(1, hidden)
        self.boxes_direct_project = nn.Linear(4, hidden)
        self.boxes_pool_project = nn.Conv2d(hidden, hidden, 7, padding=3)
        self.boxes_pos_enc_project = nn.Linear(hidden + 2, hidden)
        self.vision_layer_norm = nn.LayerNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)
        self.prompt_layer_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            Sam3GeometryEncoderLayer(hidden) for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden)

    def forward(self, boxes=None, points=None, vision_features=None):
        """Encode geometric prompts."""
        B = vision_features.shape[0] if vision_features is not None else 1
        dev = vision_features.device if vision_features is not None else boxes.device

        prompts = []
        if boxes is not None:
            box_embed = self.boxes_direct_project(boxes)
            prompts.append(box_embed)
        if points is not None:
            # Point prompts encoded as label embeddings
            point_embed = self.label_embed(points["labels"])
            prompts.append(point_embed)

        if len(prompts) == 0:
            return None

        x = torch.cat(prompts, dim=1) if len(prompts) > 1 else prompts[0]
        x = self.prompt_layer_norm(x)

        if vision_features is not None:
            v = self.vision_layer_norm(vision_features)
            for layer in self.layers:
                x = layer(x, v)

        x = self.output_layer_norm(self.final_proj(x))
        return x


# ── Mask Decoder ────────────────────────────────────────────────
class Sam3PixelDecoder(nn.Module):
    """Progressive upsampling from multi-scale FPN features."""
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(32, hidden),
            nn.GroupNorm(32, hidden),
            nn.GroupNorm(32, hidden),
        ])

    def forward(self, features):
        """features: tuple of multi-scale FPN outputs (high-res first)."""
        # Progressive fusion: smallest → largest
        x = features[2]  # [B, 256, 72, 72]
        # Upsample and fuse with 144
        x = F.interpolate(x, size=features[1].shape[2:], mode="bilinear", align_corners=False)
        x = x + features[1]
        x = self.norms[0](self.conv_layers[0](F.gelu(x)))
        # Upsample and fuse with 288
        x = F.interpolate(x, size=features[0].shape[2:], mode="bilinear", align_corners=False)
        x = x + features[0]
        x = self.norms[1](self.conv_layers[1](F.gelu(x)))
        # Final refinement
        x = self.norms[2](self.conv_layers[2](F.gelu(x)))
        return x  # [B, 256, 288, 288]


class Sam3MaskDecoder(nn.Module):
    """Produces final segmentation masks from decoder outputs + pixel features."""
    def __init__(self, hidden: int = 256, num_heads: int = 8):
        super().__init__()
        self.pixel_decoder = Sam3PixelDecoder(hidden)
        self.mask_embedder = Sam3MaskEmbedder(hidden)
        self.instance_projection = nn.Conv2d(hidden, hidden, 1)
        self.semantic_projection = nn.Conv2d(hidden, 1, 1)

        # Prompt-based cross-attention for text conditioning
        self.prompt_cross_attn = Sam3Attention(hidden, num_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden)
        self.prompt_cross_attn_dropout = nn.Dropout(0.0)

    def forward(self, decoder_output, multi_scale_features, text_features=None):
        """
        Args:
            decoder_output: [B, num_queries, 256] from DETR decoder
            multi_scale_features: tuple of 4 FPN feature maps
            text_features: [B, seq_len, 256] projected text
        Returns:
            pred_masks: [B, num_queries, H, W] instance masks
            semantic_seg: [B, 1, H, W] semantic segmentation
        """
        # Pixel features from FPN
        pixel_features = self.pixel_decoder(multi_scale_features[:3])  # [B, 256, 288, 288]

        # Text conditioning via cross-attention
        if text_features is not None:
            B, N, C = pixel_features.shape[0], pixel_features.shape[2] * pixel_features.shape[3], pixel_features.shape[1]
            pf_flat = pixel_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            pf_flat = pf_flat + self.prompt_cross_attn_dropout(
                self.prompt_cross_attn(
                    self.prompt_cross_attn_norm(pf_flat), text_features, text_features
                )[0]
            )
            pixel_features = pf_flat.transpose(1, 2).reshape(B, C, pixel_features.shape[2], pixel_features.shape[3])

        # Instance masks via dot product
        mask_embed = self.mask_embedder(decoder_output)  # [B, queries, 256]
        instance_feat = self.instance_projection(pixel_features)  # [B, 256, H, W]
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, instance_feat)

        # Semantic segmentation
        semantic_seg = self.semantic_projection(pixel_features)

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
        }


# ── Dot Product Scoring ─────────────────────────────────────────
class Sam3DotProductScoring(nn.Module):
    """Scores query-text alignment for text-conditioned detection."""
    def __init__(self, hidden: int = 256, intermediate: int = 2048):
        super().__init__()
        self.text_mlp = Sam3MLP(hidden, intermediate)
        self.text_mlp_dropout = nn.Dropout(0.0)
        self.text_mlp_out_norm = nn.LayerNorm(hidden)
        self.text_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)

    def forward(self, query_embeds, text_embeds):
        """
        query_embeds: [layers, B, queries, hidden]
        text_embeds: [B, seq, hidden]
        """
        text = self.text_mlp_out_norm(text_embeds + self.text_mlp_dropout(self.text_mlp(text_embeds)))
        text_pooled = text.mean(dim=1)  # [B, hidden]
        text_proj = self.text_proj(text_pooled)  # [B, hidden]
        query_proj = self.query_proj(query_embeds)  # [L, B, Q, hidden]
        # Dot product scoring
        scores = (query_proj * text_proj.unsqueeze(0).unsqueeze(2)).sum(dim=-1, keepdim=True)
        return scores  # [L, B, Q, 1]
