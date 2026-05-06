"""PaliGemma builder. Auto-registers via REGISTRY_PATTERN."""
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "google/paligemma-3b-mix-224"
ARCH = "gemma1"
REGISTRY_PATTERN = "paligemma"


def build_paligemma(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    from .model import PaliGemma
    vc, tc = config.get("vision_config", {}), config.get("text_config", {})
    debug = kw.pop("debug", False)

    model = PaliGemma(vision_cfg=vc, text_cfg=tc, dtype=dtype)
    model.load_weights(Path(save_dir), debug=debug)
    model = model.to(dtype)

    dev = model.smart_device() if device == "auto" else torch.device(device)
    if dev.type == "cuda":
        free, needed = model.get_free_gpu_memory_mb(), model.estimate_param_memory_mb()
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
