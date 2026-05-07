"""
SAM3 — Segment Anything Model 3.

Full architecture: ViT backbone → FPN Neck → CLIP text → DETR encoder/decoder → Mask decoder.
Output: Instance masks [B, 200, 288, 288], boxes [B, 200, 4], semantic seg [B, 1, 288, 288].

This is a SEGMENTATION model — NOT autoregressive text generation.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....core.base_model import BaseModel
from .....weights.huggingface import HuggingFaceDownloader
from .vision_encoder import Sam3VisionModel
from .decoder import (Sam3DetrEncoder, Sam3DetrDecoder, Sam3GeometryEncoder,
                      Sam3MaskDecoder, Sam3DotProductScoring)
from .projector import Sam3TextProjection

logger = logging.getLogger(__name__)


class CLIPTextEncoder(nn.Module):
    """Minimal CLIP text encoder matching HF weight keys.

    Architecture: token_embedding → position_embedding → transformer → layernorm → projection
    HF prefix: text_encoder.text_model.* and text_encoder.text_projection.*
    """
    def __init__(self, vocab_size: int = 49408, hidden: int = 1024,
                 num_heads: int = 16, num_layers: int = 24,
                 intermediate: int = 4096, max_position: int = 32):
        super().__init__()
        self.text_model = CLIPTextModel(vocab_size, hidden, num_heads,
                                         num_layers, intermediate, max_position)
        self.text_projection = nn.Linear(hidden, 512, bias=False)

    def forward(self, input_ids, attention_mask=None):
        text_out = self.text_model(input_ids, attention_mask)
        # Pool: take [EOS] token embedding (last non-padding)
        pooled = text_out["pooler_output"]
        text_embeds = self.text_projection(pooled)
        return {
            "text_embeds": text_embeds,              # [B, 512]
            "last_hidden_state": text_out["last_hidden_state"],  # [B, seq, 1024]
        }


class CLIPTextModel(nn.Module):
    def __init__(self, vocab_size, hidden, num_heads, num_layers,
                 intermediate, max_position):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden, max_position)
        self.encoder = CLIPEncoder(hidden, num_heads, num_layers, intermediate)
        self.final_layer_norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        x = self.encoder(x, attention_mask)["last_hidden_state"]
        x = self.final_layer_norm(x)
        # Pooler: EOS token
        pooled = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
        return {"last_hidden_state": x, "pooler_output": pooled}


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden, max_position):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden)
        self.position_embedding = nn.Embedding(max_position, hidden)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(pos_ids)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, intermediate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden),
        )

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.mlp(x)
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, hidden, num_heads, num_layers, intermediate):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden, num_heads, intermediate)
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        # Build causal mask for CLIP
        seq_len = x.shape[1]
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, attn_mask=causal)
        return {"last_hidden_state": x}


# ── Main SAM3 Model ────────────────────────────────────────────
class Sam3Model(BaseModel):
    """Segment Anything Model 3 — Text-conditioned instance segmentation.

    Architecture:
        ViT backbone (32L, 1024) → FPN Neck (4 scales)
        CLIP text encoder (24L, 1024) → text projection (1024→256)
        Geometry encoder (boxes/points)
        DETR encoder (6L) + decoder (6L, 200 queries)
        Mask decoder → instance masks + semantic seg

    ~840M parameters total.
    """
    model_type = "sam3"

    def __init__(self, vision_cfg=None, text_cfg=None, decoder_cfg=None,
                 dtype=torch.float32):
        super().__init__(dtype=dtype)
        vc = vision_cfg or {}
        tc = text_cfg or {}
        dc = decoder_cfg or {}

        # ── Vision encoder ──
        self.vision_encoder = Sam3VisionModel(
            hidden=vc.get("hidden_size", 1024),
            num_heads=vc.get("num_attention_heads", 16),
            intermediate=vc.get("intermediate_size", 4096),
            num_layers=vc.get("num_hidden_layers", 32),
            image_size=vc.get("image_size", 1008),
            patch_size=vc.get("patch_size", 14),
            out_channels=dc.get("hidden_size", 256),
            window_size=vc.get("window_size", 8),
        )

        # ── CLIP text encoder ──
        self.text_encoder = CLIPTextEncoder(
            vocab_size=tc.get("vocab_size", 49408),
            hidden=tc.get("hidden_size", 1024),
            num_heads=tc.get("num_attention_heads", 16),
            num_layers=tc.get("num_hidden_layers", 24),
            intermediate=tc.get("intermediate_size", 4096),
            max_position=tc.get("max_position_embeddings", 32),
        )

        # ── Text projection (CLIP 1024 → decoder 256) ──
        self.text_projection = Sam3TextProjection(
            text_dim=tc.get("hidden_size", 1024),
            decoder_dim=dc.get("hidden_size", 256),
        )

        decoder_hidden = dc.get("hidden_size", 256)
        decoder_heads = dc.get("num_attention_heads", 8)
        decoder_intermediate = dc.get("intermediate_size", 2048)
        num_queries = dc.get("num_queries", 200)

        # ── Geometry encoder ──
        self.geometry_encoder = Sam3GeometryEncoder(
            hidden=decoder_hidden,
            num_layers=dc.get("geometry_layers", 3),
        )

        # ── DETR encoder (processes flattened vision features) ──
        self.detr_encoder = Sam3DetrEncoder(
            hidden=decoder_hidden,
            num_heads=decoder_heads,
            intermediate=decoder_intermediate,
            num_layers=dc.get("encoder_layers", 6),
        )

        # ── DETR decoder (query-based detection) ──
        self.detr_decoder = Sam3DetrDecoder(
            hidden=decoder_hidden,
            num_heads=decoder_heads,
            intermediate=decoder_intermediate,
            num_layers=dc.get("decoder_layers", 6),
            num_queries=num_queries,
        )

        # ── Mask decoder ──
        self.mask_decoder = Sam3MaskDecoder(
            hidden=decoder_hidden,
            num_heads=decoder_heads,
        )

        # ── Dot-product scoring ──
        self.dot_product_scoring = Sam3DotProductScoring(
            hidden=decoder_hidden,
            intermediate=decoder_intermediate,
        )

        # Store config
        self.image_size = vc.get("image_size", 1008)
        self.num_queries = num_queries

    def load_weights(self, path: Path, debug: bool = False):
        """Load weights from HuggingFace checkpoint."""
        dl = HuggingFaceDownloader(repo_id="", save_dir=path)
        state = dl.load_state_dict()
        if debug:
            ckpt_keys = set(state.keys())
            model_keys = set(self.state_dict().keys())
            print(f"\n📦 Checkpoint: {len(ckpt_keys)} | Model: {len(model_keys)}")
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            for k in sorted(unexpected)[:20]:
                print(f"  ? {k}")
            for k in sorted(missing)[:20]:
                print(f"  ✗ {k}")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing %d keys", len(missing))
        if unexpected:
            logger.warning("Unexpected %d keys", len(unexpected))
        if not missing and not unexpected:
            logger.info("✅ All %d weights loaded from %s", len(state), path)

    def forward(self, pixel_values, input_ids, attention_mask=None,
                boxes=None, points=None, original_sizes=None):
        """Full forward pass.

        Args:
            pixel_values: [B, 3, 1008, 1008]
            input_ids: [B, 32] CLIP token IDs
            attention_mask: [B, 32]
            boxes: [B, N, 4] optional box prompts (normalized)
            points: dict with coords/labels for point prompts
            original_sizes: [B, 2] for mask resizing

        Returns:
            dict with pred_masks, pred_boxes, scores, semantic_seg
        """
        # ── Vision ──
        vis_out = self.vision_encoder(pixel_values)
        multi_scale = vis_out["multi_scale_features"]
        vis_hidden = vis_out["last_hidden_state"]  # [B, 5184, 1024]

        # ── Text ──
        text_out = self.text_encoder(input_ids, attention_mask)
        text_hidden = text_out["last_hidden_state"]  # [B, 32, 1024]
        text_proj = self.text_projection(text_hidden)  # [B, 32, 256]

        # ── Flatten vision features for DETR ──
        # Use the 72×72 scale (index 2)
        vis_flat = multi_scale[2].flatten(2).transpose(1, 2)  # [B, 5184, 256]

        # ── DETR encoder ──
        enc_out = self.detr_encoder(vis_flat)
        encoded_vision = enc_out["last_hidden_state"]  # [B, 5184, 256]

        # ── Geometry encoding (optional) ──
        if boxes is not None or points is not None:
            geo_out = self.geometry_encoder(
                boxes=boxes, points=points,
                vision_features=encoded_vision,
            )
            # Could condition decoder, but SAM3 uses text primarily
        else:
            geo_out = None

        # ── DETR decoder ──
        dec_out = self.detr_decoder(encoded_vision, text_proj)
        inter_hidden = dec_out["intermediate_hidden_states"]  # [6, B, 200, 256]
        pred_boxes = dec_out["pred_boxes"][-1]  # [B, 200, 4] from last layer
        last_hidden = dec_out["last_hidden_state"]  # [B, 200, 256]

        # ── Mask decoder ──
        mask_out = self.mask_decoder(last_hidden, multi_scale, text_proj)
        pred_masks = mask_out["pred_masks"]      # [B, 200, 288, 288]
        semantic_seg = mask_out["semantic_seg"]   # [B, 1, 288, 288]

        # ── Text-query scoring ──
        scores = self.dot_product_scoring(inter_hidden, text_proj)  # [6, B, 200, 1]
        final_scores = scores[-1].squeeze(-1)  # [B, 200]

        return {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "scores": final_scores,
            "semantic_seg": semantic_seg,
            "all_pred_boxes": dec_out["pred_boxes"],
            "presence_logit": dec_out["presence_logit"],
        }

    @torch.no_grad()
    def segment(self, pixel_values, input_ids, attention_mask=None,
                boxes=None, threshold: float = 0.5,
                original_sizes=None) -> Dict[str, Any]:
        """High-level segmentation API.

        Returns filtered masks and boxes above confidence threshold.
        """
        out = self.forward(pixel_values, input_ids, attention_mask, boxes=boxes)

        scores = out["scores"].sigmoid()  # [B, 200]
        pred_masks = out["pred_masks"]     # [B, 200, 288, 288]
        pred_boxes = out["pred_boxes"]     # [B, 200, 4]

        results = []
        for b in range(scores.shape[0]):
            keep = scores[b] > threshold
            masks_b = pred_masks[b, keep]  # [K, 288, 288]
            boxes_b = pred_boxes[b, keep]  # [K, 4]
            scores_b = scores[b, keep]     # [K]

            # Binarize masks
            binary_masks = (masks_b.sigmoid() > 0.5).float()

            # Resize to original if needed
            if original_sizes is not None:
                oh, ow = original_sizes[b].tolist()
                binary_masks = F.interpolate(
                    binary_masks.unsqueeze(1),
                    size=(oh, ow),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            results.append({
                "masks": binary_masks,
                "boxes": boxes_b,
                "scores": scores_b,
                "num_detections": keep.sum().item(),
            })

        return results[0] if len(results) == 1 else results
