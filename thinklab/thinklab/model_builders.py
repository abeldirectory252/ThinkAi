"""Auto-register all supported model builders."""
import logging
import json
import torch
from pathlib import Path

logger = logging.getLogger("thinklab.builders")


# ── PaliGemma builder ───────────────────────────────────────────────
def build_paligemma(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    from .models.multimodal.paligemma import PaliGemma
    from .weights import HuggingFaceDownloader

    vc = config.get("vision_config", {})
    tc = config.get("text_config", {})

    model = PaliGemma(vision_cfg=vc, text_cfg=tc, dtype=dtype, model_type="gemma1")
    _load_and_place(model, save_dir, dtype, device, max_memory_gb)
    return model


# ── MedGemma builder ───────────────────────────────────────────────
def build_medgemma(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    from .models.multimodal.paligemma import PaliGemma

    vc = config.get("vision_config", {})
    tc = config.get("text_config", {})

    model = PaliGemma(vision_cfg=vc, text_cfg=tc, dtype=dtype, model_type="gemma3")
    _load_and_place(model, save_dir, dtype, device, max_memory_gb)
    return model


# ── Shared weight loading + device placement ────────────────────────
def _load_and_place(model, save_dir, dtype, device, max_memory_gb):
    from .weights import HuggingFaceDownloader

    model.load_weights(Path(save_dir))
    model = model.to(dtype)

    if device == "auto":
        dev = model.smart_device()
    else:
        dev = torch.device(device)

    if dev.type == "cuda":
        free = model.get_free_gpu_memory_mb()
        needed = model.estimate_param_memory_mb()
        budget = max_memory_gb * 1024 if max_memory_gb else free

        if budget < needed * 1.2:
            logger.info("Offloading layers (%.0f MB free, %.0f MB needed)", free, needed)
            model.vision_tower.to(dev)
            model.multi_modal_projector.to(dev)
            model.language_model.model.embed_tokens.to(dev)
            model.language_model.model.norm.to(dev)
            model.offload_layers_to_cpu(model.language_model.model.layers, keep_on_gpu=4)
        else:
            model.to(dev)
    else:
        model.to(dev)

    model.eval()
    return model


# ── Register all models on import ──────────────────────────────────
from .registry import register_model

register_model("paligemma", build_paligemma, arch="gemma1",
               defaults={"model_type": "gemma1"})
register_model("medgemma",  build_medgemma,  arch="gemma3",
               defaults={"model_type": "gemma3"})
