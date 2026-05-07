"""
SAM3 builder — constructs and loads the Segment Anything Model 3.
Auto-registers with ThinkLab on import via REGISTRY_PATTERN.

Usage:
    model = thinklab.load_llm("facebook/sam3", token="hf_xxx", device="auto")
    result = model.inference(prompt="segment the red object", image_path="photo.jpg")
"""
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/sam3"
ARCH = "sam3"
REGISTRY_PATTERN = "sam3"


def build_sam3(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    """Build and load the SAM3 model."""
    from .model import Sam3Model

    vc = config.get("vision_config", {})
    tc = config.get("text_config", {})
    dc = config.get("decoder_config", {})
    debug = kw.pop("debug", False)

    # Default configs for SAM3 if not provided
    vc.setdefault("hidden_size", 1024)
    vc.setdefault("num_attention_heads", 16)
    vc.setdefault("intermediate_size", 4096)
    vc.setdefault("num_hidden_layers", 32)
    vc.setdefault("image_size", 1008)
    vc.setdefault("patch_size", 14)
    vc.setdefault("window_size", 8)

    tc.setdefault("vocab_size", 49408)
    tc.setdefault("hidden_size", 1024)
    tc.setdefault("num_attention_heads", 16)
    tc.setdefault("num_hidden_layers", 24)
    tc.setdefault("intermediate_size", 4096)
    tc.setdefault("max_position_embeddings", 32)

    dc.setdefault("hidden_size", 256)
    dc.setdefault("num_attention_heads", 8)
    dc.setdefault("intermediate_size", 2048)
    dc.setdefault("encoder_layers", 6)
    dc.setdefault("decoder_layers", 6)
    dc.setdefault("geometry_layers", 3)
    dc.setdefault("num_queries", 200)

    model = Sam3Model(vision_cfg=vc, text_cfg=tc, decoder_cfg=dc, dtype=dtype)
    model.load_weights(Path(save_dir), debug=debug)
    model = model.to(dtype)

    # Device placement
    if device == "auto":
        dev = model.smart_device()
    else:
        dev = torch.device(device)

    if dev.type == "cuda":
        free = model.get_free_gpu_memory_mb()
        needed = model.estimate_param_memory_mb()
        budget = max_memory_gb * 1024 if max_memory_gb else free
        if budget < needed * 1.2:
            logger.info("Offloading SAM3 layers (%.0f MB free, %.0f needed)", free, needed)
            # Keep vision encoder + text encoder on GPU, offload DETR layers
            model.vision_encoder.to(dev)
            model.text_encoder.to(dev)
            model.text_projection.to(dev)
            model.mask_decoder.to(dev)
            model.dot_product_scoring.to(dev)
            model.offload_layers_to_cpu(
                list(model.detr_encoder.layers) + list(model.detr_decoder.layers),
                keep_on_gpu=4,
            )
        else:
            model.to(dev)
    else:
        model.to(dev)

    model.eval()
    return model
