"""
SAM3 builder — constructs and loads the Segment Anything Model 3.
Auto-registers with ThinkLab on import via REGISTRY_PATTERN.

Usage:
    model = thinklab.load_llm("facebook/sam3", token="hf_xxx", device="auto")
    result = model.inference(prompt="segment the red object", image_path="photo.jpg")

thinklab\thinklab\models\multimodal\facebook\sam3\builder.py
"""
"""SAM3 builder — constructs and loads Segment Anything Model 3."""
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/sam3"
ARCH = "sam3"
REGISTRY_PATTERN = "sam3"


def _extract_config(config: dict) -> tuple[dict, dict, dict]:
    raw_vis = config.get("vision_config", {})
    backbone = raw_vis.get("backbone_config", {})
    vc = {**backbone}
    if "fpn_hidden_size" in raw_vis:
        vc["fpn_hidden_size"] = raw_vis["fpn_hidden_size"]
    if "scale_factors" in raw_vis:
        vc["scale_factors"] = raw_vis["scale_factors"]

    tc = config.get("text_config", {})

    detr_enc = config.get("detr_encoder_config", {})
    detr_dec = config.get("detr_decoder_config", {})
    geo = config.get("geometry_encoder_config", {})
    mask_dec = config.get("mask_decoder_config", {})

    dc = {
        "hidden_size": detr_enc.get("hidden_size", 256),
        "num_attention_heads": detr_enc.get("num_attention_heads", 8),
        "intermediate_size": detr_enc.get("intermediate_size", 2048),
        "encoder_layers": detr_enc.get("num_layers", 6),
        "dropout": detr_enc.get("dropout", 0.0),
        "decoder_layers": detr_dec.get("num_layers", 6),
        "num_queries": detr_dec.get("num_queries", 200),
        "geometry_layers": geo.get("num_layers", 3),
        "num_upsampling_stages": mask_dec.get("num_upsampling_stages", 3),
    }
    return vc, tc, dc


def build_sam3(save_dir, config, dtype, device, max_memory_gb=None, **kw):
    from .model import Sam3Model

    debug = kw.pop("debug", False)
    vc, tc, dc = _extract_config(config)

    vc.setdefault("hidden_size", 1024)
    vc.setdefault("num_attention_heads", 16)
    vc.setdefault("intermediate_size", 4736)
    vc.setdefault("num_hidden_layers", 32)
    vc.setdefault("image_size", 1008)
    vc.setdefault("patch_size", 14)
    vc.setdefault("pretrain_image_size", 336)
    vc.setdefault("window_size", 8)
    vc.setdefault("global_attn_indexes", [7, 15, 23, 31])
    vc.setdefault("scale_factors", [4.0, 2.0, 1.0, 0.5])

    tc.setdefault("vocab_size", 49408)
    tc.setdefault("hidden_size", 1024)
    tc.setdefault("num_attention_heads", 16)
    tc.setdefault("num_hidden_layers", 24)
    tc.setdefault("intermediate_size", 4096)
    tc.setdefault("max_position_embeddings", 32)
    tc.setdefault("projection_dim", 512)

    dc.setdefault("hidden_size", 256)
    dc.setdefault("num_attention_heads", 8)
    dc.setdefault("intermediate_size", 2048)
    dc.setdefault("encoder_layers", 6)
    dc.setdefault("decoder_layers", 6)
    dc.setdefault("geometry_layers", 3)
    dc.setdefault("num_queries", 200)
    dc.setdefault("num_upsampling_stages", 3)

    model = Sam3Model(vision_cfg=vc, text_cfg=tc, decoder_cfg=dc, dtype=dtype)
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
            logger.info("Offloading SAM3 layers (%.0f MB free, %.0f needed)", free, needed)
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