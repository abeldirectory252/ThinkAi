"""
SAM3 Diagnostic: Run on Kaggle to find where YOUR model diverges from HF.
Compares intermediate outputs at each stage.
"""
import torch
import requests
from PIL import Image

# ── Load HF model ──
from transformers import Sam3Processor, Sam3Model as HFSam3Model
processor = Sam3Processor.from_pretrained("facebook/sam3")
hf_model = HFSam3Model.from_pretrained("facebook/sam3").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
hf_model = hf_model.to(device)

# ── Load image ──
url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = processor(images=image, text="nose", return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

pixel_values = inputs["pixel_values"]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print(f"pixel_values: {pixel_values.shape}, dtype={pixel_values.dtype}")
print(f"input_ids: {input_ids.shape} = {input_ids[0].tolist()}")
print(f"attention_mask: {attention_mask[0].tolist()}")

# ══════════════════════════════════════════════════════════════
# STAGE 1: Text Encoder
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 1: Text Encoder (CLIP)")
print("="*60)

with torch.no_grad():
    hf_text = hf_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hf_lhs = hf_text.last_hidden_state  # [B, 32, 1024]
    hf_proj = hf_text.pooler_output      # [B, 32, 256] (text_projection applied)

print(f"HF last_hidden_state: {hf_lhs.shape}")
print(f"  min={hf_lhs.min():.4f} max={hf_lhs.max():.4f} mean={hf_lhs.mean():.6f}")
print(f"  [0,0,:5] = {hf_lhs[0,0,:5].tolist()}")
print(f"  [0,1,:5] = {hf_lhs[0,1,:5].tolist()}")
print(f"HF pooler_output (text_proj): {hf_proj.shape}")
print(f"  min={hf_proj.min():.4f} max={hf_proj.max():.4f} mean={hf_proj.mean():.6f}")
print(f"  [0,0,:5] = {hf_proj[0,0,:5].tolist()}")

# ══════════════════════════════════════════════════════════════
# STAGE 2: Vision Encoder
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 2: Vision Encoder")
print("="*60)

with torch.no_grad():
    hf_vis = hf_model.vision_encoder(pixel_values)

print(f"HF last_hidden_state: {hf_vis.last_hidden_state.shape}")
print(f"  min={hf_vis.last_hidden_state.min():.4f} max={hf_vis.last_hidden_state.max():.4f}")
for i, (fh, fp) in enumerate(zip(hf_vis.fpn_hidden_states, hf_vis.fpn_position_encoding)):
    print(f"  FPN level {i}: feat={fh.shape} pos={fp.shape} feat_mean={fh.mean():.6f}")

# ══════════════════════════════════════════════════════════════
# STAGE 3: DETR Encoder
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3: DETR Encoder")
print("="*60)

fpn_hidden = hf_vis.fpn_hidden_states[:-1]
fpn_pos = hf_vis.fpn_position_encoding[:-1]
text_mask = attention_mask.bool()

with torch.no_grad():
    hf_enc = hf_model.detr_encoder(
        vision_features=[fpn_hidden[-1]],
        text_features=hf_proj,
        vision_pos_embeds=[fpn_pos[-1]],
        text_mask=text_mask,
    )

print(f"HF encoder output: {hf_enc.last_hidden_state.shape}")
print(f"  min={hf_enc.last_hidden_state.min():.4f} max={hf_enc.last_hidden_state.max():.4f}")
print(f"  mean={hf_enc.last_hidden_state.mean():.6f}")
print(f"  spatial_shapes: {hf_enc.spatial_shapes}")

# ══════════════════════════════════════════════════════════════
# STAGE 4: DETR Decoder
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 4: DETR Decoder")
print("="*60)

with torch.no_grad():
    hf_dec = hf_model.detr_decoder(
        vision_features=hf_enc.last_hidden_state,
        text_features=hf_enc.text_features,
        vision_pos_encoding=hf_enc.pos_embeds_flattened,
        text_mask=text_mask,
        spatial_shapes=hf_enc.spatial_shapes,
    )

print(f"HF decoder intermediate_hs: {hf_dec.intermediate_hidden_states.shape}")
print(f"  last layer mean={hf_dec.intermediate_hidden_states[-1].mean():.6f}")
print(f"HF reference_boxes: {hf_dec.reference_boxes.shape}")
print(f"  last layer sample: {hf_dec.reference_boxes[-1, 0, :3]}")
print(f"HF presence_logits: {hf_dec.presence_logits.shape}")
print(f"  last layer: {hf_dec.presence_logits[-1]}")

# ══════════════════════════════════════════════════════════════
# STAGE 5: Scoring
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 5: Scoring + Final Output")
print("="*60)

with torch.no_grad():
    hf_out = hf_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

scores = hf_out.pred_logits.sigmoid()
if hf_out.presence_logits is not None:
    scores = scores * hf_out.presence_logits.sigmoid()

print(f"HF pred_logits: min={hf_out.pred_logits.min():.4f} max={hf_out.pred_logits.max():.4f}")
print(f"HF presence_logits: {hf_out.presence_logits}")
print(f"HF final scores: min={scores.min():.4f} max={scores.max():.4f}")
print(f"HF scores > 0.3: {(scores > 0.3).sum().item()}")
print(f"HF scores > 0.5: {(scores > 0.5).sum().item()}")
print(f"HF pred_masks: {hf_out.pred_masks.shape}")
print(f"HF pred_boxes sample: {hf_out.pred_boxes[0, :3]}")

# ══════════════════════════════════════════════════════════════
# STAGE 6: Print key reference values for YOUR model to match
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("REFERENCE VALUES — Your model should produce these:")
print("="*60)
print(f"1. CLIP last_hidden_state[0,0,:5] = {hf_lhs[0,0,:5].tolist()}")
print(f"2. text_proj[0,0,:5] = {hf_proj[0,0,:5].tolist()}")
print(f"3. Vision last_hs mean = {hf_vis.last_hidden_state.mean():.6f}")
print(f"4. Encoder output mean = {hf_enc.last_hidden_state.mean():.6f}")
print(f"5. Decoder last hs mean = {hf_dec.intermediate_hidden_states[-1].mean():.6f}")
print(f"6. Final max score = {scores.max():.4f}")
print(f"7. Detections > 0.3 = {(scores > 0.3).sum().item()}")
