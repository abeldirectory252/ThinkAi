"""Quick test: build model, dump keys, compare against checkpoint safetensors."""
import sys
import torch
sys.path.insert(0, r"e:\think\thinklab")

from thinklab.models.multimodal.facebook.sam3.model import Sam3Model

# Build with default configs
model = Sam3Model(dtype=torch.float32)
model_keys = sorted(model.state_dict().keys())

print(f"✅ Model built: {len(model_keys)} parameters")
print(f"   ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Print key prefixes to verify structure
prefixes = set()
for k in model_keys:
    parts = k.split(".")
    prefixes.add(".".join(parts[:2]))

print(f"\n📋 Top-level key prefixes ({len(prefixes)}):")
for p in sorted(prefixes):
    count = sum(1 for k in model_keys if k.startswith(p + ".") or k == p)
    print(f"  {p}: {count} keys")

# Check for checkpoint if available
import os
ckpt_dir = os.environ.get("SAM3_WEIGHTS", r"e:\think\models\facebook\sam3")
safetensor_files = list(filter(lambda f: f.endswith(".safetensors"),
                                os.listdir(ckpt_dir))) if os.path.isdir(ckpt_dir) else []

if safetensor_files:
    from thinklab.weights.huggingface import HuggingFaceDownloader
    import re
    dl = HuggingFaceDownloader(repo_id="", save_dir=ckpt_dir)
    ckpt = dl.load_state_dict()

    # Strip detector_model. prefix + tracker keys
    cleaned = {}
    for k, v in ckpt.items():
        new_key = re.sub(r'^detector_model\.', '', k)
        if not new_key.startswith("tracker_model.") and not new_key.startswith("tracker_neck."):
            cleaned[new_key] = v

    ckpt_keys = set(cleaned.keys())
    own_keys = set(model_keys)

    matched = ckpt_keys & own_keys
    missing = own_keys - ckpt_keys
    unexpected = ckpt_keys - own_keys

    print(f"\n📦 Checkpoint: {len(ckpt_keys)} | Model: {len(own_keys)} | Matched: {len(matched)}")

    if unexpected:
        print(f"\n  ? UNEXPECTED in checkpoint ({len(unexpected)}):")
        for k in sorted(unexpected)[:30]:
            print(f"    ? {k}  shape={cleaned[k].shape}")

    if missing:
        print(f"\n  ✗ MISSING from checkpoint ({len(missing)}):")
        for k in sorted(missing)[:30]:
            print(f"    ✗ {k}")

    # Shape mismatches
    mismatched = []
    for k in matched:
        if cleaned[k].shape != model.state_dict()[k].shape:
            mismatched.append((k, cleaned[k].shape, model.state_dict()[k].shape))
    if mismatched:
        print(f"\n  ⚠ SHAPE MISMATCHES ({len(mismatched)}):")
        for k, cs, ms in mismatched[:20]:
            print(f"    {k}: ckpt={cs} model={ms}")
    else:
        print(f"\n  ✅ All {len(matched)} matched keys have correct shapes!")

    if not unexpected and not missing and not mismatched:
        print("\n🎉 PERFECT MATCH — all keys align!")
else:
    print(f"\n⚠ No checkpoint found at {ckpt_dir}")
    print("  Set SAM3_WEIGHTS env var or download weights first")
    print("\n  Model keys sample (first 30):")
    for k in model_keys[:30]:
        print(f"    {k}")
