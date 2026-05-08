"""
SAM3 Weight Verification: Compare weight values between HF and YOUR model.
Run on Kaggle after loading both models.
"""
import torch, re
from transformers import Sam3Model as HFSam3Model

# Load HF model
hf_model = HFSam3Model.from_pretrained("facebook/sam3").eval()

# Load YOUR model (same as diagnose_your_model.py)
import json
from huggingface_hub import hf_hub_download
from pathlib import Path
from safetensors.torch import load_file

# Load weights
config_path = hf_hub_download("facebook/sam3", "config.json")
with open(config_path) as f:
    config = json.load(f)

from thinklab.models.multimodal.facebook.sam3.builder import _extract_config
from thinklab.models.multimodal.facebook.sam3.model import Sam3Model
vc, tc, dc = _extract_config(config)
vc.setdefault("hidden_size", 1024); vc.setdefault("num_attention_heads", 16)
vc.setdefault("intermediate_size", 4736); vc.setdefault("num_hidden_layers", 32)
vc.setdefault("image_size", 1008); vc.setdefault("patch_size", 14)
vc.setdefault("pretrain_image_size", 336); vc.setdefault("window_size", 8)
vc.setdefault("global_attn_indexes", [7, 15, 23, 31])
vc.setdefault("scale_factors", [4.0, 2.0, 1.0, 0.5])
tc.setdefault("vocab_size", 49408); tc.setdefault("hidden_size", 1024)
tc.setdefault("num_attention_heads", 16); tc.setdefault("num_hidden_layers", 24)
tc.setdefault("intermediate_size", 4096); tc.setdefault("max_position_embeddings", 32)
tc.setdefault("projection_dim", 512)
dc.setdefault("hidden_size", 256); dc.setdefault("num_attention_heads", 8)
dc.setdefault("intermediate_size", 2048); dc.setdefault("encoder_layers", 6)
dc.setdefault("decoder_layers", 6); dc.setdefault("geometry_layers", 3)
dc.setdefault("num_queries", 200); dc.setdefault("num_upsampling_stages", 3)

my_model = Sam3Model(vision_cfg=vc, text_cfg=tc, decoder_cfg=dc, dtype=torch.float32)

# Load safetensors
wdir = Path(hf_hub_download("facebook/sam3", "model.safetensors")).parent
safetensor_files = sorted(wdir.glob("*.safetensors"))
state_dict = {}
for f in safetensor_files:
    state_dict.update(load_file(str(f)))
cleaned = {}
for k, v in state_dict.items():
    new_key = re.sub(r'^detector_model\.', '', k)
    if not new_key.startswith("tracker_model.") and not new_key.startswith("tracker_neck."):
        cleaned[new_key] = v

missing, unexpected = my_model.load_state_dict(cleaned, strict=False)
print(f"YOUR model: missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    print("MISSING keys:")
    for k in sorted(missing):
        print(f"  {k}")
if unexpected:
    print(f"UNEXPECTED keys (first 20):")
    for k in sorted(unexpected)[:20]:
        print(f"  {k}")

# ── Compare weight values ──
print("\n" + "="*60)
print("WEIGHT VALUE COMPARISON")
print("="*60)

hf_sd = hf_model.state_dict()
my_sd = my_model.state_dict()

# Check key critical weights
critical_keys = [
    "vision_encoder.backbone.embeddings.position_embeddings",
    "vision_encoder.backbone.layer_norm.weight",
    "vision_encoder.backbone.layers.0.attention.q_proj.weight",
    "vision_encoder.backbone.layers.0.layer_norm1.weight",
    "vision_encoder.neck.fpn_layers.0.proj1.weight",
    "text_encoder.text_model.embeddings.token_embedding.weight",
    "text_projection.weight",
    "detr_encoder.layers.0.self_attn.q_proj.weight",
    "detr_decoder.query_embed.weight",
    "detr_decoder.reference_points.weight",
    "detr_decoder.presence_token.weight",
    "detr_decoder.presence_head.layer1.weight",
    "detr_decoder.box_head.layer1.weight",
    "dot_product_scoring.text_proj.weight",
    "dot_product_scoring.query_proj.weight",
    "mask_decoder.instance_projection.weight",
]

max_diff_keys = []
for key in critical_keys:
    hf_key = key
    my_key = key
    if hf_key in hf_sd and my_key in my_sd:
        hf_w = hf_sd[hf_key].float().cpu()
        my_w = my_sd[my_key].float().cpu()
        if hf_w.shape == my_w.shape:
            diff = (hf_w - my_w).abs().max().item()
            match = "✅" if diff < 1e-6 else "❌"
            print(f"  {match} {key}: max_diff={diff:.2e}, shape={hf_w.shape}")
            if diff > 1e-6:
                max_diff_keys.append((key, diff))
        else:
            print(f"  ⚠️  {key}: SHAPE MISMATCH hf={hf_w.shape} yours={my_w.shape}")
    elif hf_key in hf_sd and my_key not in my_sd:
        print(f"  ❌ {key}: EXISTS in HF, MISSING in yours")
    elif hf_key not in hf_sd and my_key in my_sd:
        print(f"  ❌ {key}: MISSING in HF, exists in yours")
    else:
        print(f"  ❌ {key}: missing in BOTH")

# Also check ALL weights for any mismatches
print(f"\n{'='*60}")
print("FULL WEIGHT SCAN")
print("="*60)
total_checked = 0
total_match = 0
total_mismatch = 0
total_shape_mismatch = 0
for key in sorted(hf_sd.keys()):
    if key in my_sd:
        total_checked += 1
        hf_w = hf_sd[key].float().cpu()
        my_w = my_sd[key].float().cpu()
        if hf_w.shape != my_w.shape:
            total_shape_mismatch += 1
            print(f"  SHAPE: {key} hf={hf_w.shape} yours={my_w.shape}")
        elif (hf_w - my_w).abs().max().item() > 1e-6:
            total_mismatch += 1
            diff = (hf_w - my_w).abs().max().item()
            print(f"  DIFF:  {key} max_diff={diff:.2e}")
        else:
            total_match += 1

print(f"\nTotal: {total_checked} checked, {total_match} exact match, "
      f"{total_mismatch} value mismatch, {total_shape_mismatch} shape mismatch")
print(f"HF keys not in yours: {len(set(hf_sd.keys()) - set(my_sd.keys()))}")
print(f"Your keys not in HF: {len(set(my_sd.keys()) - set(hf_sd.keys()))}")
