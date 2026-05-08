"""
SAM3 Side-by-Side Diagnostic: Run YOUR model with HF inputs/weights.
Run this AFTER diagnose_on_kaggle.py on the same Kaggle notebook.
"""
import torch, re
from transformers import Sam3Processor

# Use HF processor to get IDENTICAL inputs
processor = Sam3Processor.from_pretrained("facebook/sam3")
import requests
from PIL import Image
url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(images=image, text="nose", return_tensors="pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
pixel_values = inputs["pixel_values"].to(device)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# ── Build YOUR model with correct config ──
# Import your model (adjust path as needed for Kaggle)
from thinklab.models.multimodal.facebook.sam3.model import Sam3Model

# Load HF config to build your model correctly
import json
from huggingface_hub import hf_hub_download
config_path = hf_hub_download("facebook/sam3", "config.json")
with open(config_path) as f:
    config = json.load(f)

from thinklab.models.multimodal.facebook.sam3.builder import _extract_config
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

my_model = Sam3Model(vision_cfg=vc, text_cfg=tc, decoder_cfg=dc, dtype=torch.float32)

# Load HF weights
from safetensors.torch import load_file
import os
weight_dir = hf_hub_download("facebook/sam3", "model.safetensors")
# Or if multiple shards:
from pathlib import Path
wdir = Path(weight_dir).parent
safetensor_files = sorted(wdir.glob("*.safetensors"))
state_dict = {}
for f in safetensor_files:
    state_dict.update(load_file(str(f)))
# Strip detector_model. prefix
cleaned = {}
for k, v in state_dict.items():
    new_key = re.sub(r'^detector_model\.', '', k)
    if not new_key.startswith("tracker_model.") and not new_key.startswith("tracker_neck."):
        cleaned[new_key] = v

missing, unexpected = my_model.load_state_dict(cleaned, strict=False)
print(f"Weight loading: {len(cleaned)} total, missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    print(f"  MISSING: {missing[:10]}")
if unexpected:
    print(f"  UNEXPECTED: {unexpected[:10]}")

my_model = my_model.to(device).eval()

# ══════════════════════════════════════════════════════════════
# Compare at each stage
# ══════════════════════════════════════════════════════════════
from thinklab.models.multimodal.facebook.sam3.decoder import inverse_sigmoid, box_cxcywh_to_xyxy

with torch.no_grad():
    # STAGE 1: Text
    print("\n" + "="*60)
    print("YOUR STAGE 1: Text Encoder")
    print("="*60)
    text_out = my_model.text_encoder(input_ids, attention_mask)
    my_lhs = text_out["last_hidden_state"]
    my_proj = my_model.text_projection(my_lhs)
    print(f"last_hidden_state: {my_lhs.shape}")
    print(f"  min={my_lhs.min():.4f} max={my_lhs.max():.4f} mean={my_lhs.mean():.6f}")
    print(f"  [0,0,:5] = {my_lhs[0,0,:5].tolist()}")
    print(f"  [0,1,:5] = {my_lhs[0,1,:5].tolist()}")
    print(f"  HF ref:   [-0.2105, -0.1924, 0.3384, 0.2954, 0.0889]")
    print(f"text_proj: {my_proj.shape}")
    print(f"  [0,0,:5] = {my_proj[0,0,:5].tolist()}")
    print(f"  HF ref:   [0.0689, -0.0397, -0.0311, -0.0243, -0.0878]")

    # STAGE 2: Vision
    print("\n" + "="*60)
    print("YOUR STAGE 2: Vision Encoder")
    print("="*60)
    vis_out = my_model.vision_encoder(pixel_values)
    print(f"last_hidden_state: {vis_out['last_hidden_state'].shape}")
    print(f"  min={vis_out['last_hidden_state'].min():.4f} max={vis_out['last_hidden_state'].max():.4f}")
    print(f"  mean={vis_out['last_hidden_state'].mean():.6f} (HF ref: -0.011117)")
    for i, (fh, fp) in enumerate(zip(vis_out["fpn_hidden_states"], vis_out["fpn_position_encoding"])):
        print(f"  FPN level {i}: feat={fh.shape} feat_mean={fh.mean():.6f}")

    # STAGE 3: DETR Encoder
    print("\n" + "="*60)
    print("YOUR STAGE 3: DETR Encoder")
    print("="*60)
    fpn_hidden = vis_out["fpn_hidden_states"][:-1]
    fpn_pos = vis_out["fpn_position_encoding"][:-1]
    text_mask = attention_mask.bool()
    enc_out = my_model.detr_encoder(
        vision_features=[fpn_hidden[-1]],
        text_features=my_proj,
        vision_pos_embeds=[fpn_pos[-1]],
        text_mask=text_mask)
    print(f"encoder output: {enc_out['last_hidden_state'].shape}")
    print(f"  mean={enc_out['last_hidden_state'].mean():.6f} (HF ref: -0.076233)")
    print(f"  spatial_shapes: {enc_out['spatial_shapes']}")

    # STAGE 4: DETR Decoder
    print("\n" + "="*60)
    print("YOUR STAGE 4: DETR Decoder")
    print("="*60)
    dec_out = my_model.detr_decoder(
        vision_features=enc_out["last_hidden_state"],
        text_features=enc_out["text_features"],
        vision_pos_encoding=enc_out["pos_embeds_flattened"],
        text_mask=text_mask,
        spatial_shapes=enc_out["spatial_shapes"])
    print(f"intermediate_hs: {dec_out['intermediate_hidden_states'].shape}")
    print(f"  last layer mean={dec_out['intermediate_hidden_states'][-1].mean():.6f} (HF ref: -0.001942)")
    print(f"reference_boxes sample: {dec_out['reference_boxes'][-1, 0, :3]}")
    print(f"presence_logits last: {dec_out['presence_logits'][-1]} (HF ref: [[3.9548]])")

    # STAGE 5: Full forward
    print("\n" + "="*60)
    print("YOUR STAGE 5: Full Forward")
    print("="*60)
    out = my_model(pixel_values, input_ids, attention_mask)
    scores = out["pred_logits"].sigmoid()
    if out["presence_logits"] is not None:
        scores = scores * out["presence_logits"].sigmoid()
    print(f"pred_logits: min={out['pred_logits'].min():.4f} max={out['pred_logits'].max():.4f}")
    print(f"presence_logits: {out['presence_logits']} (HF ref: [[3.9548]])")
    print(f"final scores: min={scores.min():.4f} max={scores.max():.4f} (HF ref max: 0.7081)")
    print(f"scores > 0.3: {(scores > 0.3).sum().item()} (HF ref: 2)")
    print(f"scores > 0.5: {(scores > 0.5).sum().item()} (HF ref: 1)")
