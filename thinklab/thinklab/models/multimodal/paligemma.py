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
from .gemma3_lm import Gemma3ForCausalLM
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
            from .gemma_lm import RMSNorm
            self.mm_soft_emb_norm = RMSNorm(vis_dim)
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
        if is_gemma3:
            # Handle both 'sliding_window' and 'sliding_window_size' config keys
            sw = tc.get("sliding_window", tc.get("sliding_window_size", 4096))
            self.language_model = Gemma3ForCausalLM(
                vocab=tc.get("vocab_size", 262144),
                hidden=tc.get("hidden_size", 2560),
                layers=tc.get("num_hidden_layers", 34),
                heads=tc.get("num_attention_heads", 8),
                kv_heads=tc.get("num_key_value_heads", 4),
                head_dim=tc.get("head_dim", 256),
                intermediate=tc.get("intermediate_size", 10240),
                eps=tc.get("rms_norm_eps", 1e-6),
                sliding_window=sw,
                global_every=tc.get("global_every", 4),
                softcap=tc.get("attn_logit_softcapping", 50.0),
                final_logit_softcap=tc.get("final_logit_softcapping", None),
            )
        else:
            self.language_model = GemmaForCausalLM(
                vocab=tc.get("vocab_size", 257216),
                hidden=tc.get("hidden_size", 2048),
                layers=tc.get("num_hidden_layers", 18),
                heads=tc.get("num_attention_heads", 8),
                kv_heads=tc.get("num_key_value_heads", 1),
                head_dim=tc.get("head_dim", 256),
                intermediate=tc.get("intermediate_size", 16384),
                eps=tc.get("rms_norm_eps", 1e-6),
            )

        self.num_image_tokens = (
            vc.get("image_size", 224) // vc.get("patch_size", 14)
        ) ** 2
        self.image_token_id = 257152
        self.text_hidden = tc.get("hidden_size", 2048)
        self.num_text_layers = tc.get("num_hidden_layers", 18)
        # Store final logit softcap for generate() (Gemma 2 = 30.0, Gemma 3 = None)
        self.final_logit_softcap = tc.get("final_logit_softcapping", None)


    # ── Weight loading ──────────────────────────────────────────────
    def load_weights(self, path: Path, debug: bool = False) -> None:
        """Load from a directory containing .safetensors files."""
        dl = HuggingFaceDownloader(repo_id="", save_dir=path)
        state = dl.load_state_dict()

        if debug:
            ckpt_keys = set(state.keys())
            model_keys = set(self.state_dict().keys())
            print(f"\n📦 Checkpoint: {len(ckpt_keys)} tensors | Model: {len(model_keys)} params")
            only_ckpt = sorted(ckpt_keys - model_keys)
            only_model = sorted(model_keys - ckpt_keys)
            if only_ckpt:
                print(f"  ⚠️  In checkpoint only ({len(only_ckpt)}):")
                for k in only_ckpt:
                    print(f"    ? {k:55s} {list(state[k].shape)}")
            if only_model:
                print(f"  ⚠️  In model only ({len(only_model)}):")
                for k in only_model:
                    print(f"    ✗ {k}")
            if not only_ckpt and not only_model:
                print("  ✅ Perfect key alignment")

        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing %d keys: %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected %d keys: %s", len(unexpected), unexpected[:5])
        if not missing and not unexpected:
            logger.info("✅ All %d weights loaded perfectly from %s", len(state), path)
        else:
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
                          device: torch.device,
                          num_image_tokens: int = 0) -> torch.Tensor:
        """Build causal mask with bidirectional attention for image tokens.

        PaliGemma's input is [IMG_0, IMG_1, ..., IMG_255, <bos>, text...].
        Image tokens must attend to each other bidirectionally.
        Text tokens are causal (attend to all image + prior text tokens).
        """
        # Start with standard causal mask
        mask = torch.full((1, 1, q_len, kv_len), float("-inf"),
                          dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

        # Open up bidirectional attention in the image token region
        if num_image_tokens > 0 and num_image_tokens <= q_len:
            mask[:, :, :num_image_tokens, :num_image_tokens] = 0.0

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

        # Causal mask (bidirectional for image tokens, causal for text)
        seq_len = combined.shape[1]
        mask = self._make_causal_mask(seq_len, seq_len, combined.dtype, dev,
                                      num_image_tokens=self.num_image_tokens)

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
        output_attentions: bool = False,
    ) -> dict:
        dev = pixel_values.device

        # ── Phase 1: Prefill ─────────────────────────────────────────
        # Vision encode
        vis_out = self.vision_tower(
            pixel_values, output_attentions=output_attentions,
        )
        image_embeds = self.multi_modal_projector(vis_out["last_hidden_state"])

        # Text embeddings
        text_embeds = self.language_model.model.embed_tokens(input_ids)
        text_embeds = text_embeds * (self.text_hidden ** 0.5)

        # Merge vision + text
        combined = text_embeds.clone()
        for b in range(input_ids.shape[0]):
            img_mask = input_ids[b] == self.image_token_id
            n_img = img_mask.sum().item()
            if n_img > 0:
                combined[b, img_mask] = image_embeds[b, :n_img].to(combined.dtype)

        # Causal mask for prefill (bidirectional for image tokens, causal for text)
        prefix_len = combined.shape[1]
        mask = self._make_causal_mask(prefix_len, prefix_len, combined.dtype, dev,
                                      num_image_tokens=self.num_image_tokens)

        # Initialize KV caches
        caches = self.language_model.init_caches(self.num_text_layers)

        # Run prefill through all decoder layers (populates caches)
        h = combined
        for i, layer in enumerate(self.language_model.model.layers):
            param_dev = next(layer.parameters()).device
            if param_dev != dev:
                h, _ = self.layer_forward_with_offload(
                    layer, h, mask=mask, cache=caches[i],
                    target_device=str(dev),
                )
            else:
                h, _ = layer(h, mask, cache=caches[i])

        h = self.language_model.model.norm(h)
        logits = F.linear(h, self.language_model.model.embed_tokens.weight)
        if self.final_logit_softcap is not None:
            logits = torch.tanh(logits / self.final_logit_softcap) * self.final_logit_softcap
        next_logits = logits[:, -1, :]

        # ── Phase 2: Decode (token-by-token with KV cache) ───────────
        generated_ids = []

        for _ in range(max_new_tokens):
            # Sample next token
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

            token_id = next_id.item() if next_id.numel() == 1 else next_id[0, 0].item()
            generated_ids.append(token_id)

            # Check EOS
            if token_id == 1:
                break

            # Embed new token
            new_embed = self.language_model.model.embed_tokens(next_id)
            new_embed = new_embed * (self.text_hidden ** 0.5)

            # Run single token through all layers with cache
            # No mask needed: 1 query token attending to all cached K,V
            h = new_embed
            for i, layer in enumerate(self.language_model.model.layers):
                param_dev = next(layer.parameters()).device
                if param_dev != dev:
                    h, _ = self.layer_forward_with_offload(
                        layer, h, mask=None, cache=caches[i],
                        target_device=str(dev),
                    )
                else:
                    h, _ = layer(h, mask=None, cache=caches[i])

            h = self.language_model.model.norm(h)
            next_logits = F.linear(
                h, self.language_model.model.embed_tokens.weight
            ).squeeze(1)
            if self.final_logit_softcap is not None:
                next_logits = torch.tanh(next_logits / self.final_logit_softcap) * self.final_logit_softcap

        return {
            "generated_ids": generated_ids,
            "vision_features": vis_out.get("last_hidden_state"),
            "vision_attentions": vis_out.get("attentions"),
            "text_attentions": None,
            "image_embeds": image_embeds,
        }
