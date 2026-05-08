"""
SAM3 Input Pipeline Comparison: HF vs ThinkLab
Run this on Kaggle to compare exactly what inputs HF passes to the model
vs what your implementation passes.
"""
import torch
import requests
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# PART 1: HF Pipeline — Trace exact inputs to model.forward()
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: HuggingFace Pipeline Input Tracing")
print("=" * 70)

from transformers import Sam3Processor, Sam3Model

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
print(f"Original image size: {image.size} (W x H)")

# Process inputs through HF processor
inputs = processor(images=image, text="nose", return_tensors="pt")

print("\n--- HF Processor Output Keys ---")
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: type={type(v)}")

# Inspect tokenizer details
print("\n--- HF Tokenizer Details ---")
hf_tokenizer = processor.tokenizer
print(f"  Tokenizer class: {type(hf_tokenizer).__name__}")
print(f"  Vocab size: {hf_tokenizer.vocab_size}")
print(f"  BOS token: '{hf_tokenizer.bos_token}' id={hf_tokenizer.bos_token_id}")
print(f"  EOS token: '{hf_tokenizer.eos_token}' id={hf_tokenizer.eos_token_id}")
print(f"  PAD token: '{hf_tokenizer.pad_token}' id={hf_tokenizer.pad_token_id}")

# Show the actual token IDs
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
print(f"\n--- Token IDs for 'nose' ---")
print(f"  input_ids shape: {input_ids.shape}")
print(f"  input_ids: {input_ids[0].tolist()}")
print(f"  attention_mask: {attention_mask[0].tolist()}")
print(f"  Non-pad tokens: {attention_mask[0].sum().item()}")

# Decode back
decoded = hf_tokenizer.decode(input_ids[0], skip_special_tokens=False)
print(f"  Decoded: '{decoded}'")

# Show pixel_values stats
pv = inputs["pixel_values"]
print(f"\n--- HF pixel_values stats ---")
print(f"  shape: {pv.shape}")
print(f"  dtype: {pv.dtype}")
print(f"  min={pv.min():.4f}, max={pv.max():.4f}")
print(f"  mean={pv.mean():.4f}, std={pv.std():.4f}")
ch_means = [f"{pv[0, c].mean().item():.4f}" for c in range(3)]
print(f"  channel means: {ch_means}")

# ═══════════════════════════════════════════════════════════════
# PART 2: Hook into model.forward() to capture internal tensors
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: HF Model Internal Tensor Flow")
print("=" * 70)

inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()}

# Remove original_sizes from model inputs (it's for post-processing only)
model_inputs = {k: v for k, v in inputs_on_device.items()
                if k in ('pixel_values', 'input_ids', 'attention_mask')}

print(f"\n--- Inputs passed to model.forward() ---")
for k, v in model_inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

with torch.no_grad():
    # Step-by-step trace through HF model
    # 1) Text encoding
    text_out = model.get_text_features(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        return_dict=True
    )
    text_features = text_out.pooler_output  # This is text_projection(last_hidden_state)
    print(f"\n--- After get_text_features ---")
    print(f"  last_hidden_state: {text_out.last_hidden_state.shape}")
    print(f"  pooler_output (=text_projection output): {text_features.shape}")
    print(f"  pooler_output min={text_features.min():.4f}, max={text_features.max():.4f}")
    print(f"  pooler_output[:, :5] sample: {text_features[0, 0, :5].tolist()}")

    # 2) Vision encoding
    vis_out = model.vision_encoder(model_inputs["pixel_values"])
    print(f"\n--- After vision_encoder ---")
    print(f"  last_hidden_state: {vis_out.last_hidden_state.shape}")
    for i, (fh, fp) in enumerate(zip(vis_out.fpn_hidden_states, vis_out.fpn_position_encoding)):
        print(f"  FPN level {i}: features={fh.shape}, pos={fp.shape}")

    # 3) Which FPN levels go to encoder vs mask decoder
    fpn_hidden = vis_out.fpn_hidden_states[:-1]  # all but last
    fpn_pos = vis_out.fpn_position_encoding[:-1]
    print(f"\n--- FPN levels used ---")
    print(f"  DETR encoder uses: fpn_hidden[-1] = {fpn_hidden[-1].shape}")
    print(f"  Mask decoder uses: fpn_hidden[:-1] = {[f.shape for f in fpn_hidden[:-1]]}")

    # 4) text_mask
    text_mask = model_inputs["attention_mask"].bool()
    print(f"\n--- text_mask ---")
    print(f"  shape: {text_mask.shape}")
    print(f"  values: {text_mask[0].tolist()}")

    # 5) Combined prompt = text_features only (no geometry)
    combined_prompt = text_features
    combined_mask = text_mask
    print(f"\n--- Combined prompt (text only, no geometry) ---")
    print(f"  combined_prompt: {combined_prompt.shape}")
    print(f"  combined_mask: {combined_mask.shape}")

    # 6) DETR encoder
    enc_out = model.detr_encoder(
        vision_features=[fpn_hidden[-1]],
        text_features=combined_prompt,
        vision_pos_embeds=[fpn_pos[-1]],
        text_mask=combined_mask,
    )
    print(f"\n--- After DETR encoder ---")
    print(f"  last_hidden_state: {enc_out.last_hidden_state.shape}")
    print(f"  text_features: {enc_out.text_features.shape}")
    print(f"  spatial_shapes: {enc_out.spatial_shapes}")

    # 7) DETR decoder
    dec_out = model.detr_decoder(
        vision_features=enc_out.last_hidden_state,
        text_features=enc_out.text_features,
        vision_pos_encoding=enc_out.pos_embeds_flattened,
        text_mask=combined_mask,
        spatial_shapes=enc_out.spatial_shapes,
    )
    print(f"\n--- After DETR decoder ---")
    print(f"  intermediate_hidden_states: {dec_out.intermediate_hidden_states.shape}")
    print(f"  reference_boxes: {dec_out.reference_boxes.shape}")
    print(f"  presence_logits: {dec_out.presence_logits.shape}")

    # 8) Full model output
    outputs = model(**model_inputs)
    print(f"\n--- Full model output ---")
    print(f"  pred_masks: {outputs.pred_masks.shape}")
    print(f"  pred_boxes: {outputs.pred_boxes.shape}")
    print(f"  pred_logits: {outputs.pred_logits.shape}")
    print(f"  presence_logits: {outputs.presence_logits.shape}")

    # 9) Post-process
    scores = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()
    print(f"\n--- Score distribution ---")
    print(f"  max score: {scores.max():.4f}")
    print(f"  scores > 0.3: {(scores > 0.3).sum().item()}")
    print(f"  scores > 0.5: {(scores > 0.5).sum().item()}")

    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=[tuple(inputs["original_sizes"][0].tolist())]
    )[0]
    print(f"  Found {len(results['masks'])} objects with threshold=0.5")


# ═══════════════════════════════════════════════════════════════
# PART 3: KEY DIFFERENCE — How get_text_features works
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: CRITICAL — get_text_features vs your text pipeline")
print("=" * 70)

with torch.no_grad():
    # HF get_text_features does this:
    #   1. text_encoder(input_ids, attention_mask) -> last_hidden_state, pooler_output
    #   2. text_projection(last_hidden_state) -> THIS becomes pooler_output
    # So pooler_output = text_projection(last_hidden_state)  [B, seq_len, 256]
    
    # Step by step:
    clip_out = model.text_encoder(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        return_dict=True,
    )
    print(f"CLIP text_encoder outputs:")
    print(f"  last_hidden_state: {clip_out.last_hidden_state.shape}")
    print(f"  text_embeds (CLIP projection): {clip_out.text_embeds.shape if hasattr(clip_out, 'text_embeds') and clip_out.text_embeds is not None else 'N/A'}")
    
    # HF then does: text_projection(last_hidden_state)
    projected = model.text_projection(clip_out.last_hidden_state)
    print(f"  After model.text_projection: {projected.shape}")
    print(f"  This is what goes to DETR encoder as 'text_features'")
    
    # YOUR implementation does:
    # text_out = self.text_encoder(input_ids, attention_mask)
    # text_hidden = text_out["last_hidden_state"]  
    # text_proj = self.text_projection(text_hidden)  # [B, seq, 256]
    # This part is CORRECT — same as HF
    
    print(f"\n  YOUR impl: text_projection(last_hidden_state) -> same ✅")


# ═══════════════════════════════════════════════════════════════
# PART 4: Compare tokenization specifically
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: Tokenization Comparison")
print("=" * 70)

test_words = ["nose", "Ear", "Eye", "cat eye", "cat ear", "cat nose"]

for word in test_words:
    hf_tokens = processor.tokenizer(
        word, return_tensors="pt", padding="max_length", max_length=32
    )
    ids = hf_tokens["input_ids"][0].tolist()
    mask = hf_tokens["attention_mask"][0].tolist()
    non_pad = sum(mask)
    print(f"\n  '{word}':")
    print(f"    input_ids[:{non_pad}]: {ids[:non_pad]}")
    print(f"    attention_mask sum: {non_pad}")
    print(f"    full ids: {ids}")
    print(f"    full mask: {mask}")


# ═══════════════════════════════════════════════════════════════
# PART 5: Verify FPN level selection (CRITICAL)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 5: FPN Level Selection Analysis")
print("=" * 70)

with torch.no_grad():
    print(f"\nHF uses fpn_hidden_states[:-1] for processing:")
    print(f"  Total FPN levels: {len(vis_out.fpn_hidden_states)}")
    for i, fh in enumerate(vis_out.fpn_hidden_states):
        print(f"  Level {i}: {fh.shape}")
    
    print(f"\n  fpn_hidden_states[:-1] = levels 0..{len(vis_out.fpn_hidden_states)-2}")
    print(f"  DETR encoder: vision_features=[fpn_hidden_states[:-1][-1]] = level {len(vis_out.fpn_hidden_states)-2}")
    print(f"  Mask decoder backbone: fpn_hidden_states[:-1][:-1] = levels 0..{len(vis_out.fpn_hidden_states)-3}")
    
    # YOUR implementation uses:
    # fpn_hidden = vis_out["fpn_hidden_states"]
    # detr_encoder: vision_features=[fpn_hidden[-2]]
    # mask_decoder: backbone_features=list(fpn_hidden[:-1])
    print(f"\n  YOUR impl uses:")
    print(f"    DETR encoder: fpn_hidden[-2] — check if this matches level {len(vis_out.fpn_hidden_states)-2}")
    print(f"    Mask decoder: fpn_hidden[:-1] — check if this matches levels 0..{len(vis_out.fpn_hidden_states)-2}")

    # HF mask_decoder receives: list(fpn_hidden_states[:-1]) as backbone_features
    # Then inside _embed_pixels: backbone_features[-1] gets replaced
    # YOUR mask_decoder receives: list(fpn_hidden[:-1]) 
    # If fpn_hidden = vis_out.fpn_hidden_states (all 4 levels),
    # then fpn_hidden[:-1] = levels 0,1,2 
    # But HF passes fpn_hidden_states[:-1] = levels 0,1,2 to forward,
    # then list(fpn_hidden_states) inside forward = levels 0,1,2
    # So backbone_features = levels 0,1,2 — SAME


print("\n" + "=" * 70)
print("DONE — Compare these outputs with your implementation")
print("=" * 70)
