"""
SAM3 Decoder — DETR Encoder/Decoder, Geometry Encoder, Mask Decoder, Scoring.
All key names match HuggingFace exactly.
SAM3 DETR Encoder/Decoder, Geometry Encoder, Mask Decoder, Scoring.
sam3-decoder.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (Sam3Attention, Sam3MLP, Sam3DecoderMLP,
                     Sam3SinePositionEmbedding, Sam3MaskEmbedder)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def box_cxcywh_to_xyxy(x):
    xc, yc, w, h = x.unbind(-1)
    return torch.stack([(xc - 0.5*w), (yc - 0.5*h), (xc + 0.5*w), (yc + 0.5*h)], dim=-1)


def _build_padding_mask(text_mask, dtype, device):
    """Build [B, 1, 1, K] additive mask. text_mask: [B, K] bool, True=valid."""
    if text_mask is None:
        return None
    mask = torch.zeros(
        text_mask.shape[0], 1, 1, text_mask.shape[1],
        device=device, dtype=dtype)
    mask.masked_fill_(~text_mask.unsqueeze(1).unsqueeze(2),
                      torch.finfo(dtype).min)
    return mask


# ── DETR Encoder ────────────────────────────────────────────────
class Sam3DetrEncoderLayer(nn.Module):
    def __init__(self, hidden=256, num_heads=8, intermediate=2048, drop=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.dropout = nn.Dropout(drop)
        self.cross_attn = Sam3Attention(hidden, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Sam3MLP(hidden, intermediate, drop)
        self.layer_norm3 = nn.LayerNorm(hidden)

    def forward(self, vision_feats, prompt_feats, vision_pos,
                prompt_cross_attn_mask=None):
        # Self-attention with positional addition on q/k only
        residual = vision_feats
        h = self.layer_norm1(vision_feats)
        h_pos = h + vision_pos
        attn_out, _ = self.self_attn(h_pos, h_pos, h)
        h = self.dropout(attn_out) + residual
        # Cross-attention to text/prompt (add pos to query for spatial awareness)
        residual = h
        h = self.layer_norm2(h)
        h_pos = h + vision_pos  # pos_enc_at_cross_attn_queries=True
        attn_out, _ = self.cross_attn(h_pos, prompt_feats, prompt_feats,
                                       attention_mask=prompt_cross_attn_mask)
        h = self.dropout(attn_out) + residual
        # MLP
        residual = h
        h = self.layer_norm3(h)
        h = self.mlp(h)
        h = self.dropout(h) + residual
        return h


class Sam3DetrEncoder(nn.Module):
    def __init__(self, hidden=256, num_heads=8, intermediate=2048,
                 num_layers=6, drop=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Sam3DetrEncoderLayer(hidden, num_heads, intermediate, drop)
            for _ in range(num_layers)
        ])
        # Pooled text fusion: project mean-pooled text and add to image features
        self.text_pooling_proj = nn.Linear(hidden, hidden)

    def forward(self, vision_features, text_features, vision_pos_embeds,
                text_mask=None, spatial_shapes=None):
        if isinstance(vision_features, (list, tuple)):
            shapes = []
            feats_flat, pos_flat = [], []
            for f, p in zip(vision_features, vision_pos_embeds):
                H, W = f.shape[-2:]
                shapes.append((H, W))
                feats_flat.append(f.flatten(2).transpose(1, 2))
                pos_flat.append(p.flatten(2).transpose(1, 2))
            hidden = torch.cat(feats_flat, dim=1)
            pos = torch.cat(pos_flat, dim=1)
            spatial = torch.tensor(shapes, dtype=torch.long, device=hidden.device)
        else:
            hidden = vision_features
            pos = vision_pos_embeds
            spatial = spatial_shapes

        # Fuse mean-pooled text into image features (reference: TransformerEncoderFusion)
        if text_mask is not None:
            is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)  # [B, S, 1]
            num_valid = is_valid.sum(dim=1).clamp(min=1.0)  # [B, 1]
            pooled_text = (text_features * is_valid).sum(dim=1) / num_valid  # [B, C]
        else:
            pooled_text = text_features.mean(dim=1)  # [B, C]
        pooled_text = self.text_pooling_proj(pooled_text).unsqueeze(1)  # [B, 1, C]
        hidden = hidden + pooled_text  # broadcast add to all spatial positions

        cross_attn_mask = _build_padding_mask(text_mask, hidden.dtype, hidden.device)

        for layer in self.layers:
            hidden = layer(hidden, text_features, pos,
                           prompt_cross_attn_mask=cross_attn_mask)

        return {
            "last_hidden_state": hidden,
            "pos_embeds_flattened": pos,
            "text_features": text_features,
            "spatial_shapes": spatial,
        }


# ── DETR Decoder ────────────────────────────────────────────────
class Sam3DetrDecoderLayer(nn.Module):
    def __init__(self, hidden=256, num_heads=8, intermediate=2048, drop=0.0):
        super().__init__()
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.self_attn_dropout = nn.Dropout(drop)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)

        self.text_cross_attn = Sam3Attention(hidden, num_heads)
        self.text_cross_attn_dropout = nn.Dropout(drop)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden)

        self.vision_cross_attn = Sam3Attention(hidden, num_heads)
        self.vision_cross_attn_dropout = nn.Dropout(drop)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden)

        self.mlp = Sam3MLP(hidden, intermediate, drop)
        self.mlp_layer_norm = nn.LayerNorm(hidden)
        self.mlp_dropout = nn.Dropout(drop)

    def forward(self, hidden_states, query_pos, text_features, vision_features,
                vision_pos_encoding, text_cross_attn_mask=None,
                vision_cross_attn_mask=None):
        # Pad query_pos for presence token at position 0
        query_pos = F.pad(query_pos, (0, 0, 1, 0), value=0)

        # Self-attention: q,k = h+qpos, v = h
        residual = hidden_states
        q_pos = hidden_states + query_pos
        out, _ = self.self_attn(q_pos, q_pos, hidden_states)
        hidden_states = residual + self.self_attn_dropout(out)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Text cross-attention: q = h+qpos, k,v = text
        residual = hidden_states
        q_pos = hidden_states + query_pos
        out, _ = self.text_cross_attn(q_pos, text_features, text_features,
                                       attention_mask=text_cross_attn_mask)
        hidden_states = residual + self.text_cross_attn_dropout(out)
        hidden_states = self.text_cross_attn_layer_norm(hidden_states)

        # Vision cross-attention: q = h+qpos, k = v+pos, v = vision
        residual = hidden_states
        q_pos = hidden_states + query_pos
        k_pos = vision_features + vision_pos_encoding
        out, _ = self.vision_cross_attn(q_pos, k_pos, vision_features,
                                         attention_mask=vision_cross_attn_mask)
        hidden_states = residual + self.vision_cross_attn_dropout(out)
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states)

        # MLP
        residual = hidden_states
        out = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_dropout(out)
        hidden_states = self.mlp_layer_norm(hidden_states)
        return hidden_states


class Sam3DetrDecoder(nn.Module):
    def __init__(self, hidden=256, num_heads=8, intermediate=2048,
                 num_layers=6, num_queries=200, drop=0.0):
        super().__init__()
        self.hidden_size = hidden
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            Sam3DetrDecoderLayer(hidden, num_heads, intermediate, drop)
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden)
        self.box_head = Sam3DecoderMLP(hidden, hidden, 4, num_layers=3)
        self.query_embed = nn.Embedding(num_queries, hidden)
        self.reference_points = nn.Embedding(num_queries, 4)

        self.presence_token = nn.Embedding(1, hidden)
        self.presence_head = Sam3DecoderMLP(hidden, hidden, 1, num_layers=3)
        self.presence_layer_norm = nn.LayerNorm(hidden)

        self.ref_point_head = Sam3DecoderMLP(2 * hidden, hidden, hidden, num_layers=2)
        self.box_rpb_embed_x = Sam3DecoderMLP(2, hidden, num_heads, num_layers=2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, hidden, num_heads, num_layers=2)

        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=hidden // 2, normalize=False)
        self.clamp_presence_logit_max_val = 10.0

    def _get_rpb_matrix(self, ref_boxes, spatial_shape):
        """ref_boxes: [B, Q, 4] cxcywh in [0,1]. Returns [B, num_heads, Q, H*W]."""
        H, W = int(spatial_shape[0]), int(spatial_shape[1])
        boxes_xyxy = box_cxcywh_to_xyxy(ref_boxes)
        B, Q, _ = boxes_xyxy.shape
        coords_h = torch.arange(0, H, device=ref_boxes.device, dtype=ref_boxes.dtype) / H
        coords_w = torch.arange(0, W, device=ref_boxes.device, dtype=ref_boxes.dtype) / W
        # boxes_xyxy[..., 1:4:2] = [y1, y2]; boxes_xyxy[..., 0:3:2] = [x1, x2]
        dy = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        dy = dy.view(B, Q, -1, 2)
        dx = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        dx = dx.view(B, Q, -1, 2)
        dx_log = dx * 8
        dx_log = torch.sign(dx_log) * torch.log2(torch.abs(dx_log) + 1.0) / math.log2(8)
        dy_log = dy * 8
        dy_log = torch.sign(dy_log) * torch.log2(torch.abs(dy_log) + 1.0) / math.log2(8)
        dx_emb = self.box_rpb_embed_x(dx_log)   # [B, Q, W, num_heads]
        dy_emb = self.box_rpb_embed_y(dy_log)   # [B, Q, H, num_heads]
        rpb = dy_emb.unsqueeze(3) + dx_emb.unsqueeze(2)   # [B, Q, H, W, num_heads]
        rpb = rpb.flatten(2, 3).permute(0, 3, 1, 2).contiguous()  # [B, num_heads, Q, H*W]
        return rpb

    def forward(self, vision_features, text_features, vision_pos_encoding,
                text_mask=None, spatial_shapes=None):
        B = vision_features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_boxes = self.reference_points.weight.unsqueeze(0).expand(B, -1, -1).sigmoid()
        presence = self.presence_token.weight.unsqueeze(0).expand(B, -1, -1)
        hidden_states = torch.cat([presence, queries], dim=1)

        text_cross_attn_mask = _build_padding_mask(
            text_mask, hidden_states.dtype, hidden_states.device)

        intermediates = []
        intermediate_boxes = [ref_boxes]
        intermediate_presence = []

        for layer in self.layers:
            ref_input = ref_boxes.unsqueeze(2)
            query_sine = self.position_encoding.encode_boxes(ref_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine)

            vision_cross_attn_mask = None
            if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
                rpb = self._get_rpb_matrix(
                    ref_boxes,
                    (int(spatial_shapes[0, 0].item()), int(spatial_shapes[0, 1].item())))
                # Pad on Q dimension for presence token (dim=2 of [B,heads,Q,K])
                vision_cross_attn_mask = F.pad(rpb, (0, 0, 1, 0), value=0)

            hidden_states = layer(
                hidden_states, query_pos, text_features,
                vision_features, vision_pos_encoding,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask)

            query_hs = hidden_states[:, 1:]
            ref_before = inverse_sigmoid(ref_boxes)
            delta = self.box_head(self.output_layer_norm(query_hs))
            new_ref = (delta + ref_before).sigmoid()
            ref_boxes = new_ref.detach()

            intermediates.append(self.output_layer_norm(query_hs))
            intermediate_boxes.append(new_ref)

            p_hidden = hidden_states[:, :1]
            p_logit = self.presence_head(self.presence_layer_norm(p_hidden)).squeeze(-1)
            p_logit = p_logit.clamp(-self.clamp_presence_logit_max_val,
                                     self.clamp_presence_logit_max_val)
            intermediate_presence.append(p_logit)

        return {
            "intermediate_hidden_states": torch.stack(intermediates),
            "reference_boxes": torch.stack(intermediate_boxes[:-1]),
            "presence_logits": torch.stack(intermediate_presence),
        }


# ── Geometry Encoder ────────────────────────────────────────────
class Sam3GeometryEncoderLayer(nn.Module):
    def __init__(self, hidden=256, num_heads=8, intermediate=2048, drop=0.0):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = Sam3Attention(hidden, num_heads)
        self.dropout = nn.Dropout(drop)
        self.cross_attn = Sam3Attention(hidden, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = Sam3MLP(hidden, intermediate, drop)
        self.layer_norm3 = nn.LayerNorm(hidden)

    def forward(self, prompt_feats, vision_feats, vision_pos, prompt_mask=None):
        residual = prompt_feats
        h = self.layer_norm1(prompt_feats)
        out, _ = self.self_attn(h, h, h, attention_mask=prompt_mask)
        h = self.dropout(out) + residual
        residual = h
        h = self.layer_norm2(h)
        k = vision_feats + vision_pos
        out, _ = self.cross_attn(h, k, vision_feats)
        h = self.dropout(out) + residual
        residual = h
        h = self.layer_norm3(h)
        h = self.mlp(h)
        h = self.dropout(h) + residual
        return h


class Sam3GeometryEncoder(nn.Module):
    def __init__(self, hidden=256, num_layers=3, roi_size=7, num_heads=8,
                 intermediate=2048, drop=0.0):
        super().__init__()
        self.hidden_size = hidden
        self.roi_size = roi_size
        self.position_encoding = Sam3SinePositionEmbedding(
            num_pos_feats=hidden // 2, normalize=True)
        self.label_embed = nn.Embedding(2, hidden)
        self.cls_embed = nn.Embedding(1, hidden)
        self.boxes_direct_project = nn.Linear(4, hidden)
        self.boxes_pool_project = nn.Conv2d(hidden, hidden, roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden + 2, hidden)
        self.vision_layer_norm = nn.LayerNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)
        self.prompt_layer_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            Sam3GeometryEncoderLayer(hidden, num_heads, intermediate, drop)
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden)


# ── Pixel Decoder ───────────────────────────────────────────────
class Sam3PixelDecoder(nn.Module):
    def __init__(self, hidden=256, num_stages=3):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(num_stages)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, hidden) for _ in range(num_stages)
        ])

    def forward(self, backbone_features):
        prev = backbone_features[-1]
        for i, feat in enumerate(reversed(backbone_features[:-1])):
            prev = F.interpolate(prev, size=feat.shape[-2:], mode="nearest")
            prev = prev + feat
            prev = self.conv_layers[i](prev)
            prev = self.norms[i](prev)
            prev = F.relu(prev)
        return prev


# ── Mask Decoder ────────────────────────────────────────────────
class Sam3MaskDecoder(nn.Module):
    def __init__(self, hidden=256, num_heads=8, num_upsampling_stages=3, drop=0.0):
        super().__init__()
        self.pixel_decoder = Sam3PixelDecoder(hidden, num_upsampling_stages)
        self.mask_embedder = Sam3MaskEmbedder(hidden)
        self.instance_projection = nn.Conv2d(hidden, hidden, 1)
        self.semantic_projection = nn.Conv2d(hidden, 1, 1)
        self.prompt_cross_attn = Sam3Attention(hidden, num_heads)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden)
        self.prompt_cross_attn_dropout = nn.Dropout(drop)

    def forward(self, decoder_queries, backbone_features, encoder_hidden_states,
                prompt_features=None, prompt_mask=None):
        if prompt_features is not None:
            residual = encoder_hidden_states
            normed = self.prompt_cross_attn_norm(encoder_hidden_states)
            cross_attn_mask = _build_padding_mask(
                prompt_mask, normed.dtype, normed.device)
            out, _ = self.prompt_cross_attn(
                normed, prompt_features, prompt_features,
                attention_mask=cross_attn_mask)
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(out)

        feats = [f.clone() for f in backbone_features]
        spatial_dim = feats[-1].shape[-2] * feats[-1].shape[-1]
        enc_vis = encoder_hidden_states[:, :spatial_dim, :]
        B, _, C = enc_vis.shape
        H, W = feats[-1].shape[-2:]
        feats[-1] = enc_vis.transpose(1, 2).reshape(B, C, H, W)

        pixel_embed = self.pixel_decoder(feats)
        instance_embed = self.instance_projection(pixel_embed)
        mask_embed = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, instance_embed)
        semantic_seg = self.semantic_projection(pixel_embed)
        return {"pred_masks": pred_masks, "semantic_seg": semantic_seg}


# ── Dot Product Scoring ─────────────────────────────────────────
class Sam3DotProductScoring(nn.Module):
    def __init__(self, hidden=256, intermediate=2048, drop=0.0):
        super().__init__()
        self.text_mlp = Sam3DecoderMLP(hidden, intermediate, hidden, num_layers=2)
        self.text_mlp_dropout = nn.Dropout(drop)
        self.text_mlp_out_norm = nn.LayerNorm(hidden)
        self.text_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.scale = float(1.0 / (hidden ** 0.5))
        self.clamp_max_val = 12.0

    def forward(self, decoder_hidden_states, text_features, text_mask=None):
        """decoder_hidden_states: [L, B, Q, C], text_features: [B, S, C]."""
        orig = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features = text_features + orig
        text_features = self.text_mlp_out_norm(text_features)
        if text_mask is not None:
            is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)
            num_valid = is_valid.sum(dim=1).clamp(min=1.0)
            pooled = (text_features * is_valid).sum(dim=1) / num_valid
        else:
            pooled = text_features.mean(dim=1)
        proj_text = self.text_proj(pooled).unsqueeze(-1)        # [B, C, 1]
        proj_queries = self.query_proj(decoder_hidden_states)   # [L, B, Q, C]
        scores = torch.matmul(proj_queries, proj_text.unsqueeze(0)) * self.scale
        scores = scores.clamp(-self.clamp_max_val, self.clamp_max_val)
        return scores