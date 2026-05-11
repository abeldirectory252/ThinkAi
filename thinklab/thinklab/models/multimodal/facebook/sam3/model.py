"""
SAM3 — Segment Anything Model 3. Key names match HuggingFace exactly.

Top-level keys (after stripping 'detector_model.' from checkpoint):
  vision_encoder.backbone.*, vision_encoder.neck.*
  text_encoder.text_model.*, text_encoder.text_projection.*
  text_projection.*
  geometry_encoder.*, detr_encoder.*, detr_decoder.*
  mask_decoder.*, dot_product_scoring.*

  SAM3 — Segment Anything Model 3.
"""

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....core.base_model import BaseModel
from .....weights.huggingface import HuggingFaceDownloader
from .vision_encoder import Sam3VisionModel
from .decoder import (Sam3DetrEncoder, Sam3DetrDecoder, Sam3GeometryEncoder,
                      Sam3MaskDecoder, Sam3DotProductScoring,
                      inverse_sigmoid, box_cxcywh_to_xyxy)

logger = logging.getLogger(__name__)


# ── CLIP Text Encoder ───────────────────────────────────────────
class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=49408, hidden_size=1024, max_position=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position, hidden_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(pos_ids)


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attn_mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_w = attn_w + attn_mask
        attn_w = F.softmax(attn_w, dim=-1).to(v.dtype)
        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # quick_gelu
        h = self.fc1(x)
        return self.fc2(h * torch.sigmoid(1.702 * h))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, intermediate_size=4096):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.layer_norm1(x)
        x = residual + self.self_attn(x, attn_mask)
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.mlp(x)
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        B, seq_len, _ = x.shape
        dtype = x.dtype
        device = x.device
        min_val = torch.finfo(dtype).min

        # Causal mask: [seq, seq] with -inf above diagonal
        causal = torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1)
        # → [1, 1, seq, seq] for broadcasting
        causal = causal.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # attention_mask: [B, seq], 1=valid, 0=pad
            # Build [B, 1, 1, seq] additive mask
            expanded = attention_mask[:, None, None, :].to(dtype)
            expanded = (1.0 - expanded) * min_val
            combined = causal + expanded   # [B, 1, seq, seq]
        else:
            combined = causal

        for layer in self.layers:
            x = layer(x, combined)
        return x


class CLIPTextModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers,
                 intermediate_size, max_position):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position)
        self.encoder = CLIPEncoder(hidden_size, num_heads, num_layers, intermediate_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        x = self.encoder(x, attention_mask)
        x = self.final_layer_norm(x)
        # Pool at the EOT (highest token id) position
        pooled = x[torch.arange(x.shape[0], device=x.device),
                    input_ids.argmax(dim=-1)]
        return {"last_hidden_state": x, "pooler_output": pooled}


class CLIPTextModelWithProjection(nn.Module):
    def __init__(self, vocab_size=49408, hidden_size=1024, num_heads=16,
                 num_layers=24, intermediate_size=4096, max_position=32,
                 projection_dim=512):
        super().__init__()
        self.text_model = CLIPTextModel(vocab_size, hidden_size, num_heads,
                                         num_layers, intermediate_size, max_position)
        self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, input_ids, attention_mask=None):
        out = self.text_model(input_ids, attention_mask)
        return {
            "last_hidden_state": out["last_hidden_state"],
            "pooler_output": out["pooler_output"],
            "text_embeds": self.text_projection(out["pooler_output"]),
        }


# ── Main SAM3 Model ────────────────────────────────────────────
class Sam3Model(BaseModel):
    model_type = "sam3"

    def __init__(self, vision_cfg=None, text_cfg=None, decoder_cfg=None,
                 dtype=torch.float32):
        super().__init__(dtype=dtype)
        vc = vision_cfg or {}
        tc = text_cfg or {}
        dc = decoder_cfg or {}

        decoder_hidden = dc.get("hidden_size", 256)
        decoder_heads = dc.get("num_attention_heads", 8)
        decoder_intermediate = dc.get("intermediate_size", 2048)
        decoder_drop = dc.get("dropout", 0.0)

        self.vision_encoder = Sam3VisionModel(
            hidden_size=vc.get("hidden_size", 1024),
            num_heads=vc.get("num_attention_heads", 16),
            intermediate_size=vc.get("intermediate_size", 4736),
            num_layers=vc.get("num_hidden_layers", 32),
            image_size=vc.get("image_size", 1008),
            patch_size=vc.get("patch_size", 14),
            pretrain_image_size=vc.get("pretrain_image_size", 336), 
            fpn_hidden_size=decoder_hidden,
            window_size=vc.get("window_size", 24),  # FIXED: was 8
            global_attn_indexes=vc.get("global_attn_indexes", [7, 15, 23, 31]),
            scale_factors=vc.get("scale_factors", [4.0, 2.0, 1.0, 0.5]),
            layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
            dropout=vc.get("dropout", 0.0),
            rope_theta=vc.get("rope_theta", 10000.0),
        )

        self.text_encoder = CLIPTextModelWithProjection(
            vocab_size=tc.get("vocab_size", 49408),
            hidden_size=tc.get("hidden_size", 1024),
            num_heads=tc.get("num_attention_heads", 16),
            num_layers=tc.get("num_hidden_layers", 24),
            intermediate_size=tc.get("intermediate_size", 4096),
            max_position=tc.get("max_position_embeddings", 32),
            projection_dim=tc.get("projection_dim", 512),
        )

        # Project CLIP text hidden (1024) → decoder hidden (256)
        self.text_projection = nn.Linear(tc.get("hidden_size", 1024), decoder_hidden)

        self.geometry_encoder = Sam3GeometryEncoder(
            hidden=decoder_hidden, num_layers=dc.get("geometry_layers", 3),
            num_heads=decoder_heads, intermediate=decoder_intermediate,
            drop=decoder_drop)

        self.detr_encoder = Sam3DetrEncoder(
            hidden=decoder_hidden, num_heads=decoder_heads,
            intermediate=decoder_intermediate,
            num_layers=dc.get("encoder_layers", 6), drop=decoder_drop)

        self.detr_decoder = Sam3DetrDecoder(
            hidden=decoder_hidden, num_heads=decoder_heads,
            intermediate=decoder_intermediate,
            num_layers=dc.get("decoder_layers", 6),
            num_queries=dc.get("num_queries", 200), drop=decoder_drop)

        self.mask_decoder = Sam3MaskDecoder(
            hidden=decoder_hidden, num_heads=decoder_heads,
            num_upsampling_stages=dc.get("num_upsampling_stages", 3),
            drop=decoder_drop)

        self.dot_product_scoring = Sam3DotProductScoring(
            hidden=decoder_hidden, intermediate=decoder_intermediate,
            drop=decoder_drop)

        self.image_size = vc.get("image_size", 1008)

    def load_weights(self, path: Path, debug: bool = False):
        dl = HuggingFaceDownloader(repo_id="", save_dir=path)
        state_dict = dl.load_state_dict()

        cleaned = {}
        for k, v in state_dict.items():
            new_key = re.sub(r'^detector_model\.', '', k)
            cleaned[new_key] = v

        cleaned = {k: v for k, v in cleaned.items()
                    if not k.startswith("tracker_model.")
                    and not k.startswith("tracker_neck.")}

        if debug:
            ckpt_keys = set(cleaned.keys())
            model_keys = set(self.state_dict().keys())
            matched = ckpt_keys & model_keys
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            print(f"\n📦 Checkpoint: {len(ckpt_keys)} | Model: {len(model_keys)} | Matched: {len(matched)}")
            if unexpected:
                print(f"  ? Unexpected ({len(unexpected)}):")
                for k in sorted(unexpected)[:20]:
                    print(f"    ? {k}")
            if missing:
                print(f"  ✗ Missing ({len(missing)}):")
                for k in sorted(missing)[:20]:
                    print(f"    ✗ {k}")

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("Missing %d keys: %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected %d keys: %s", len(unexpected), unexpected[:5])

    def forward(self, pixel_values, input_ids=None, attention_mask=None,
                text_embeds=None, boxes=None, original_sizes=None,
                debug=False):
        def _ts(name, t):
            """Print tensor stats for debugging."""
            if not debug or t is None:
                return
            if t.numel() == 0:
                print(f"  {name}: EMPTY {t.shape}")
                return
            t_f = t.float()
            print(f"  {name}: shape={list(t.shape)} "
                  f"min={t_f.min().item():.4f} max={t_f.max().item():.4f} "
                  f"mean={t_f.mean().item():.4f} std={t_f.std().item():.4f}")

        if debug:
            print(f"\n{'='*60}")
            print(f"SAM3 FORWARD DEBUG — pixel_values: {list(pixel_values.shape)}")
            print(f"{'='*60}")

        # Vision
        vis_out = self.vision_encoder(pixel_values)
        # Drop coarsest FPN level (index 3, 0.5×) — match HF
        fpn_hidden = vis_out["fpn_hidden_states"][:-1]
        fpn_pos = vis_out["fpn_position_encoding"][:-1]

        if debug:
            print("\n[1] VISION ENCODER:")
            _ts("backbone_last_hidden", vis_out["last_hidden_state"])
            for i, f in enumerate(fpn_hidden):
                _ts(f"fpn_hidden[{i}]", f)

        # Text — match HF: text_features = text_projection(last_hidden_state)
        if text_embeds is None:
            text_out = self.text_encoder(input_ids, attention_mask)
            text_hidden = text_out["last_hidden_state"]
        else:
            text_hidden = text_embeds
        text_features = self.text_projection(text_hidden)

        text_mask = attention_mask.bool() if attention_mask is not None else None

        if debug:
            print("\n[2] TEXT ENCODER:")
            _ts("text_hidden", text_hidden)
            _ts("text_features (projected)", text_features)
            if text_mask is not None:
                print(f"  text_mask: {text_mask.shape} valid_count={text_mask.sum().item()}")

        # DETR encoder — uses ONLY the finest remaining FPN level
        enc_out = self.detr_encoder(
            vision_features=[fpn_hidden[-1]],
            text_features=text_features,
            vision_pos_embeds=[fpn_pos[-1]],
            text_mask=text_mask)

        if debug:
            print("\n[3] DETR ENCODER:")
            _ts("encoder_output", enc_out["last_hidden_state"])
            print(f"  spatial_shapes: {enc_out['spatial_shapes'].tolist()}")

        # DETR decoder
        dec_out = self.detr_decoder(
            vision_features=enc_out["last_hidden_state"],
            text_features=enc_out["text_features"],
            vision_pos_encoding=enc_out["pos_embeds_flattened"],
            text_mask=text_mask,
            spatial_shapes=enc_out["spatial_shapes"])

        inter_hs = dec_out["intermediate_hidden_states"]
        ref_boxes = dec_out["reference_boxes"]

        if debug:
            print("\n[4] DETR DECODER:")
            _ts("inter_hs (all layers)", inter_hs)
            _ts("ref_boxes", ref_boxes)
            _ts("presence_logits (all layers)", dec_out["presence_logits"])

        # Box predictions across all layers
        all_box_offsets = self.detr_decoder.box_head(inter_hs)
        ref_inv = inverse_sigmoid(ref_boxes)
        all_pred_boxes_cxcywh = (ref_inv + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        # Scoring across all layers
        all_pred_logits = self.dot_product_scoring(
            inter_hs, enc_out["text_features"], text_mask).squeeze(-1)

        # Take last layer for outputs
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hs = inter_hs[-1]
        presence_logits = dec_out["presence_logits"][-1]

        if debug:
            print("\n[5] SCORING (last layer):")
            _ts("pred_logits (raw)", pred_logits)
            scores_sig = pred_logits.sigmoid()
            _ts("pred_logits (sigmoid)", scores_sig)
            _ts("presence_logits (raw)", presence_logits)
            pres_sig = presence_logits.sigmoid()
            _ts("presence_logits (sigmoid)", pres_sig)
            combined = scores_sig * pres_sig
            _ts("combined_scores", combined)
            n_above = (combined > 0.3).sum().item()
            n_above_01 = (combined > 0.1).sum().item()
            top5 = combined.flatten().topk(min(5, combined.numel())).values
            print(f"  detections >0.3: {n_above}, >0.1: {n_above_01}")
            print(f"  top-5 scores: {[f'{v:.4f}' for v in top5.tolist()]}")
            _ts("pred_boxes", pred_boxes)

        # Mask decoder uses all 3 FPN levels (0,1,2)
        mask_out = self.mask_decoder(
            decoder_queries=decoder_hs,
            backbone_features=list(fpn_hidden),
            encoder_hidden_states=enc_out["last_hidden_state"],
            prompt_features=text_features,
            prompt_mask=text_mask)

        if debug:
            print("\n[6] MASK DECODER:")
            _ts("pred_masks", mask_out["pred_masks"])
            print(f"{'='*60}\n")

        return {
            "pred_masks": mask_out["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_out["semantic_seg"],
        }

    @torch.no_grad()
    def segment(self, pixel_values, input_ids, attention_mask=None,
                threshold=0.3, original_sizes=None):
        out = self.forward(pixel_values, input_ids, attention_mask)
        scores = out["pred_logits"].sigmoid()
        if out["presence_logits"] is not None:
            scores = scores * out["presence_logits"].sigmoid()
        masks = out["pred_masks"].sigmoid()
        boxes = out["pred_boxes"]
        results = []
        for b in range(scores.shape[0]):
            keep = scores[b] > threshold
            m = (masks[b, keep] > 0.5).float()
            if original_sizes is not None:
                oh, ow = original_sizes[b].tolist()
                m = F.interpolate(m.unsqueeze(1), size=(oh, ow),
                                   mode="bilinear", align_corners=False).squeeze(1)
            results.append({"masks": m, "boxes": boxes[b, keep],
                             "scores": scores[b, keep],
                             "num_detections": keep.sum().item()})
        return results[0] if len(results) == 1 else results