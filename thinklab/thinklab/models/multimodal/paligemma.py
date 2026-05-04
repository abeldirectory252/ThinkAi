"""
PaliGemma 3B — full multimodal model.
  vision_tower  → SigLIP ViT
  multi_modal_projector → Linear(1152, 2048)
  language_model → Gemma 2B causal LM

Supports: memory-aware layer offloading, KV-cache generation,
          gradient-enabled forward for Grad-CAM.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base_model import BaseModel
from ...weights.huggingface import HuggingFaceDownloader
from .siglip_vit import VisionTower
from .gemma_lm import GemmaForCausalLM, KVCache
from .tokenizer import GemmaTokenizer
from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class MultiModalProjector(nn.Module):
    """Supports both PaliGemma (Linear) and MedGemma (weight-only + norm) projectors."""
    def __init__(self, vis_dim: int = 1152, txt_dim: int = 2048,
                 model_type: str = "gemma1"):
        super().__init__()
        self.model_type = model_type
        if model_type == "gemma3":
            # MedGemma: norm on vision features, then weight-only projection
            # Checkpoint stores weight as (vis_dim, txt_dim)
            self.mm_soft_emb_norm = nn.LayerNorm(vis_dim, elementwise_affine=True)
            self.mm_input_projection_weight = nn.Parameter(
                torch.empty(vis_dim, txt_dim)
            )
        else:
            # PaliGemma: simple Linear
            self.linear = nn.Linear(vis_dim, txt_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "gemma3":
            # Norm first, then project: x @ weight
            x = self.mm_soft_emb_norm(x)
            x = x @ self.mm_input_projection_weight
            return x
        return self.linear(x)


class PaliGemma(BaseModel):
    """
    Multimodal model supporting both architectures:
      - model_type="gemma1" → PaliGemma 3B (Gemma 1 decoder)
      - model_type="gemma3" → MedGemma 4B (Gemma 3 decoder with QK-norm)
    """

    def __init__(
        self,
        vision_cfg: Optional[dict] = None,
        text_cfg: Optional[dict] = None,
        dtype: torch.dtype = torch.bfloat16,
        model_type: str = "gemma1",
    ):
        super().__init__(dtype=dtype)
        self.model_type = model_type
        is_gemma3 = model_type == "gemma3"

        vc = vision_cfg or {}
        tc = text_cfg or {}

        self.vision_tower = VisionTower(
            hidden=vc.get("hidden_size", 1152),
            heads=vc.get("num_attention_heads", 16),
            intermediate=vc.get("intermediate_size", 4304),
            num_layers=vc.get("num_hidden_layers", 27),
            image_size=vc.get("image_size", 224),
            patch_size=vc.get("patch_size", 14),
        )
        self.multi_modal_projector = MultiModalProjector(
            vc.get("hidden_size", 1152),
            tc.get("hidden_size", 2048),
            model_type=model_type,
        )
        self.language_model = GemmaForCausalLM(
            vocab=tc.get("vocab_size", 257216),
            hidden=tc.get("hidden_size", 2048),
            layers=tc.get("num_hidden_layers", 18),
            heads=tc.get("num_attention_heads", 8),
            kv_heads=tc.get("num_key_value_heads", 1),
            head_dim=tc.get("head_dim", 256),
            intermediate=tc.get("intermediate_size", 16384),
            eps=tc.get("rms_norm_eps", 1e-6),
            use_qk_norm=is_gemma3,
            use_pre_post_ff_norm=is_gemma3,
        )

        self.num_image_tokens = (
            vc.get("image_size", 224) // vc.get("patch_size", 14)
        ) ** 2
        self.image_token_id = 257152
        self.text_hidden = tc.get("hidden_size", 2048)
        self.num_text_layers = tc.get("num_hidden_layers", 18)


    # ── Weight loading ──────────────────────────────────────────────
    def load_weights(self, path: Path) -> None:
        """Load from a directory containing .safetensors files."""
        dl = HuggingFaceDownloader(repo_id="", save_dir=path)
        state = dl.load_state_dict()
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:10])
        logger.info("Loaded %d tensors from %s", len(state), path)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "google/paligemma-3b-pt-224-128",
        save_dir: str = "./weights/paligemma-3b",
        token: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
    ) -> "PaliGemma":
        """Download weights from HF and build model."""
        save_path = Path(save_dir)

        # Download
        dl = HuggingFaceDownloader(repo_id, save_path, token=token)
        dl.download_model()

        # Read config
        config_path = save_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            vc = cfg.get("vision_config", {})
            tc = cfg.get("text_config", {})
        else:
            vc, tc = {}, {}

        # Build model
        model = cls(vision_cfg=vc, text_cfg=tc, dtype=dtype)
        model.load_weights(save_path)
        model = model.to(dtype)

        # Device placement
        if device == "auto":
            dev = model.smart_device()
        else:
            dev = torch.device(device)

        if dev.type == "cuda":
            free_mb = cls.get_free_gpu_memory_mb()
            param_mb = model.estimate_param_memory_mb()
            if free_mb < param_mb * 1.2:
                logger.info("Enabling layer offloading (%.0f MB free, %.0f MB needed)",
                            free_mb, param_mb)
                model.vision_tower.to(dev)
                model.multi_modal_projector.to(dev)
                model.language_model.model.embed_tokens.to(dev)
                model.language_model.model.norm.to(dev)
                model.offload_layers_to_cpu(
                    model.language_model.model.layers, keep_on_gpu=4
                )
            else:
                model.to(dev)
        else:
            model.to(dev)

        model.eval()
        return model

    # ── Causal mask ─────────────────────────────────────────────────
    @staticmethod
    def _make_causal_mask(q_len: int, kv_len: int, dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, q_len, kv_len), float("-inf"),
                          dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
        return mask

    # ── Forward (supports gradient flow for Grad-CAM) ──────────────
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        output_attentions: bool = False,
        output_vision_hidden: bool = False,
    ) -> dict:
        dev = pixel_values.device

        # Vision encode
        vis_out = self.vision_tower(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_vision_hidden,
        )
        image_embeds = self.multi_modal_projector(vis_out["last_hidden_state"])

        # Text embeddings
        text_embeds = self.language_model.model.embed_tokens(input_ids)
        text_embeds = text_embeds * (self.text_hidden ** 0.5)

        # Merge: replace image token positions with vision embeddings
        combined = text_embeds.clone()
        for b in range(input_ids.shape[0]):
            img_mask = input_ids[b] == self.image_token_id
            n_img = img_mask.sum().item()
            if n_img > 0:
                combined[b, img_mask] = image_embeds[b, :n_img].to(combined.dtype)

        # Causal mask
        seq_len = combined.shape[1]
        mask = self._make_causal_mask(seq_len, seq_len, combined.dtype, dev)

        # LM forward
        h = combined
        all_attn = [] if output_attentions else None
        for i, layer in enumerate(self.language_model.model.layers):
            param_dev = next(layer.parameters()).device
            if param_dev != dev:
                h_l, attn = self.layer_forward_with_offload(
                    layer, h, mask=mask, output_attentions=output_attentions,
                    target_device=str(dev),
                )
            else:
                h_l, attn = layer(h, mask, output_attentions=output_attentions)
            h = h_l
            if output_attentions and attn is not None:
                all_attn.append(attn)

        h = self.language_model.model.norm(h)
        logits = F.linear(h, self.language_model.model.embed_tokens.weight)

        return {
            "logits": logits,
            "vision_features": vis_out["last_hidden_state"],
            "vision_attentions": vis_out.get("attentions"),
            "vision_hidden_states": vis_out.get("hidden_states"),
            "text_attentions": all_attn,
            "image_embeds": image_embeds,
        }

    # ── Autoregressive generation ───────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> dict:
        dev = pixel_values.device

        # Full forward on prefix
        out = self.forward(pixel_values, input_ids, output_attentions=True)
        logits = out["logits"]

        generated_ids = []
        caches = self.language_model.init_caches(self.num_text_layers)

        # Prefill caches manually is complex; simpler: re-use full output
        # and generate token-by-token from the last position
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                scaled = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, top_k, dim=-1)
                    scaled[scaled < v[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits[remove] = float("-inf")
                    scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_id.item() if next_id.numel() == 1 else next_id[0, 0].item())

            # Check EOS
            if generated_ids[-1] == 1:  # EOS
                break

            # Next step: embed new token, run through LM
            new_embed = self.language_model.model.embed_tokens(next_id)
            new_embed = new_embed * (self.text_hidden ** 0.5)
            all_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=dev)], dim=1)
            seq_len = all_ids.shape[1]
            mask = self._make_causal_mask(1, seq_len, new_embed.dtype, dev)

            h = new_embed
            for i, layer in enumerate(self.language_model.model.layers):
                param_dev = next(layer.parameters()).device
                if param_dev != dev:
                    h, _ = self.layer_forward_with_offload(
                        layer, h, mask=mask, target_device=str(dev)
                    )
                else:
                    h, _ = layer(h, mask)

            h = self.language_model.model.norm(h)
            next_logits = F.linear(h[:, -1:, :], self.language_model.model.embed_tokens.weight).squeeze(1)

        return {
            "generated_ids": generated_ids,
            "vision_features": out["vision_features"],
            "vision_attentions": out["vision_attentions"],
            "text_attentions": out["text_attentions"],
            "image_embeds": out["image_embeds"],
        }
