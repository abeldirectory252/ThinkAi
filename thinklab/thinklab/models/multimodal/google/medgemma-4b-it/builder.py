"""
MedGemma builder — constructs and loads the MedGemma 4B model.
Auto-registers with ThinkLab on import via REGISTRY_PATTERN.
"""
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "google/medgemma-4b-it"
ARCH = "gemma3"
REGISTRY_PATTERN = "medgemma"


def build_medgemma(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    from .model import MedGemma
    vc = config.get("vision_config", {})
    tc = config.get("text_config", {})
    debug = kw.pop("debug", False)

    model = MedGemma(vision_cfg=vc, text_cfg=tc, dtype=dtype)
    model.load_weights(Path(save_dir), debug=debug)
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
