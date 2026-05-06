"""PaliGemma 3B — full multimodal model. Self-contained."""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.base_model import BaseModel
from ....weights.huggingface import HuggingFaceDownloader
from .vision_encoder import VisionTower
from .decoder import GemmaForCausalLM
from .projector import MultiModalProjector

logger = logging.getLogger(__name__)


class PaliGemma(BaseModel):
    model_type = "gemma1"

    def __init__(self, vision_cfg=None, text_cfg=None, dtype=torch.bfloat16):
        super().__init__(dtype=dtype)
        vc, tc = vision_cfg or {}, text_cfg or {}

        self.vision_tower = VisionTower(
            hidden=vc.get("hidden_size", 1152), heads=vc.get("num_attention_heads", 16),
            intermediate=vc.get("intermediate_size", 4304), num_layers=vc.get("num_hidden_layers", 27),
            image_size=vc.get("image_size", 224), patch_size=vc.get("patch_size", 14),
        )
        self.multi_modal_projector = MultiModalProjector(
            vis_dim=vc.get("hidden_size", 1152), txt_dim=tc.get("hidden_size", 2048),
        )
        self.language_model = GemmaForCausalLM(
            vocab=tc.get("vocab_size", 257216), hidden=tc.get("hidden_size", 2048),
            layers=tc.get("num_hidden_layers", 18), heads=tc.get("num_attention_heads", 8),
            kv_heads=tc.get("num_key_value_heads", 1), head_dim=tc.get("head_dim", 256),
            intermediate=tc.get("intermediate_size", 16384), eps=tc.get("rms_norm_eps", 1e-6),
        )

        self.num_image_tokens = (vc.get("image_size", 224) // vc.get("patch_size", 14)) ** 2
        self.image_token_id = 257152
        self.text_hidden = tc.get("hidden_size", 2048)
        self.num_text_layers = tc.get("num_hidden_layers", 18)
        self.final_logit_softcap = None

    def load_weights(self, path: Path, debug=False):
        dl = HuggingFaceDownloader(repo_id="", save_dir=path)
        state = dl.load_state_dict()
        if debug:
            ck, mk = set(state.keys()), set(self.state_dict().keys())
            print(f"\n📦 Checkpoint: {len(ck)} | Model: {len(mk)}")
            for k in sorted(ck - mk): print(f"  ? {k}")
            for k in sorted(mk - ck): print(f"  ✗ {k}")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing: logger.warning("Missing %d keys", len(missing))
        if unexpected: logger.warning("Unexpected %d keys", len(unexpected))
        if not missing and not unexpected:
            logger.info("✅ All %d weights loaded from %s", len(state), path)

    @staticmethod
    def _make_causal_mask(q_len, kv_len, dtype, device, image_positions=None):
        mask = torch.full((1, 1, q_len, kv_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
        if image_positions is not None and len(image_positions) > 0:
            idx = image_positions.long()
            mask[:, :, idx[:, None], idx[None, :]] = 0.0
        return mask

    def forward(self, pixel_values, input_ids, output_attentions=False, output_vision_hidden=False):
        dev = pixel_values.device
        vis_out = self.vision_tower(pixel_values, output_attentions=output_attentions,
                                     output_hidden_states=output_vision_hidden)
        image_embeds = self.multi_modal_projector(vis_out["last_hidden_state"])
        text_embeds = self.language_model.model.embed_tokens(input_ids) * (self.text_hidden ** 0.5)
        combined = text_embeds.clone()
        for b in range(input_ids.shape[0]):
            m = input_ids[b] == self.image_token_id
            n = m.sum().item()
            if n > 0: combined[b, m] = image_embeds[b, :n].to(combined.dtype)

        seq_len = combined.shape[1]
        ip = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        mask = self._make_causal_mask(seq_len, seq_len, combined.dtype, dev, ip)

        h, all_attn = combined, ([] if output_attentions else None)
        for i, layer in enumerate(self.language_model.model.layers):
            pd = next(layer.parameters()).device
            if pd != dev:
                h, a = self.layer_forward_with_offload(layer, h, mask=mask,
                    output_attentions=output_attentions, target_device=str(dev))
            else:
                h, a = layer(h, mask, output_attentions=output_attentions)
            if output_attentions and a is not None: all_attn.append(a)

        h = self.language_model.model.norm(h)
        logits = F.linear(h, self.language_model.model.embed_tokens.weight)
        return {"logits": logits, "vision_features": vis_out["last_hidden_state"],
                "vision_attentions": vis_out.get("attentions"),
                "vision_hidden_states": vis_out.get("hidden_states"),
                "text_attentions": all_attn, "image_embeds": image_embeds}

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, max_new_tokens=200,
                 temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1,
                 do_sample=True, output_attentions=False, stop_token_ids=None):
        dev = pixel_values.device
        eos_ids = set(stop_token_ids or [1])

        vis_out = self.vision_tower(pixel_values, output_attentions=output_attentions)
        image_embeds = self.multi_modal_projector(vis_out["last_hidden_state"])
        text_embeds = self.language_model.model.embed_tokens(input_ids) * (self.text_hidden ** 0.5)
        combined = text_embeds.clone()
        for b in range(input_ids.shape[0]):
            m = input_ids[b] == self.image_token_id
            n = m.sum().item()
            if n > 0: combined[b, m] = image_embeds[b, :n].to(combined.dtype)

        plen = combined.shape[1]
        ip = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        mask = self._make_causal_mask(plen, plen, combined.dtype, dev, ip)
        caches = self.language_model.init_caches(self.num_text_layers)

        h = combined
        for i, layer in enumerate(self.language_model.model.layers):
            pd = next(layer.parameters()).device
            if pd != dev:
                h, _ = self.layer_forward_with_offload(layer, h, mask=mask,
                    cache=caches[i], target_device=str(dev))
            else:
                h, _ = layer(h, mask, cache=caches[i])
        h = self.language_model.model.norm(h)
        logits = F.linear(h, self.language_model.model.embed_tokens.weight)
        next_logits = logits[:, -1, :]

        generated_ids = []
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and generated_ids:
                for pid in set(generated_ids):
                    if next_logits[0, pid] > 0: next_logits[0, pid] /= repetition_penalty
                    else: next_logits[0, pid] *= repetition_penalty

            lf = next_logits.float()
            lf = torch.where(torch.isfinite(lf), lf, torch.full_like(lf, -1e9))
            if do_sample and temperature > 0:
                scaled = lf / temperature
                if top_k > 0:
                    tk, _ = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1)
                    scaled[scaled < tk[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sl, si = torch.sort(scaled, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cp > top_p; rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
                    sl[rm] = float("-inf")
                    scaled = torch.zeros_like(scaled).scatter(1, si, sl)
                probs = F.softmax(scaled, dim=-1).clamp(min=0.0)
                nid = torch.multinomial(probs, 1) if torch.isfinite(probs).all() and probs.sum() > 0 else lf.argmax(-1, keepdim=True)
            else:
                nid = lf.argmax(-1, keepdim=True)

            tid = nid.item() if nid.numel() == 1 else nid[0, 0].item()
            generated_ids.append(tid)
            if tid in eos_ids: break

            ne = self.language_model.model.embed_tokens(nid) * (self.text_hidden ** 0.5)
            h = ne
            for i, layer in enumerate(self.language_model.model.layers):
                pd = next(layer.parameters()).device
                if pd != dev:
                    h, _ = self.layer_forward_with_offload(layer, h, mask=None,
                        cache=caches[i], target_device=str(dev))
                else:
                    h, _ = layer(h, mask=None, cache=caches[i])
            h = self.language_model.model.norm(h)
            next_logits = F.linear(h.float(),
                self.language_model.model.embed_tokens.weight.float()).squeeze(1)

        return {"generated_ids": generated_ids, "vision_features": vis_out.get("last_hidden_state"),
                "vision_attentions": vis_out.get("attentions"), "text_attentions": None,
                "image_embeds": image_embeds}
