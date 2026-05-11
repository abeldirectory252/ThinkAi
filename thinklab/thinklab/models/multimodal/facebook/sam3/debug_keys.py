"""
SAM3 Weight Key Diagnostic — run on Kaggle to find missing/mismatched keys.
Paste this into a Kaggle cell AFTER loading both models.
"""
import torch
from safetensors.torch import load_file

def diagnose_weight_keys(model, checkpoint_path, prefix_strip="detector_model."):
    """Compare model state dict keys against checkpoint keys."""
    # Load checkpoint
    ckpt = load_file(checkpoint_path)
    
    # Strip prefix from checkpoint keys
    ckpt_keys = {}
    for k, v in ckpt.items():
        clean = k.replace(prefix_strip, '', 1) if k.startswith(prefix_strip) else k
        # Skip tracker keys
        if clean.startswith("tracker_model.") or clean.startswith("tracker_neck."):
            continue
        ckpt_keys[clean] = v
    
    model_keys = model.state_dict()
    
    matched = []
    shape_mismatch = []
    missing_from_ckpt = []  # In model but not checkpoint
    unexpected_in_ckpt = []  # In checkpoint but not model
    
    for k in sorted(model_keys.keys()):
        if k in ckpt_keys:
            if ckpt_keys[k].shape == model_keys[k].shape:
                matched.append(k)
            else:
                shape_mismatch.append((k, model_keys[k].shape, ckpt_keys[k].shape))
        else:
            missing_from_ckpt.append(k)
    
    for k in sorted(ckpt_keys.keys()):
        if k not in model_keys:
            unexpected_in_ckpt.append(k)
    
    print(f"\n{'='*70}")
    print(f"WEIGHT KEY DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Checkpoint keys (after strip): {len(ckpt_keys)}")
    print(f"Model keys:                    {len(model_keys)}")
    print(f"Matched:                       {len(matched)}")
    print(f"Shape mismatches:              {len(shape_mismatch)}")
    print(f"Missing from checkpoint:       {len(missing_from_ckpt)}")
    print(f"Unexpected in checkpoint:      {len(unexpected_in_ckpt)}")
    
    if missing_from_ckpt:
        print(f"\n{'─'*70}")
        print(f"MISSING FROM CHECKPOINT (model has, checkpoint doesn't):")
        print(f"{'─'*70}")
        for k in missing_from_ckpt:
            print(f"  ✗ {k}  shape={list(model_keys[k].shape)}")
    
    if unexpected_in_ckpt:
        print(f"\n{'─'*70}")
        print(f"UNEXPECTED IN CHECKPOINT (checkpoint has, model doesn't):")
        print(f"{'─'*70}")
        for k in unexpected_in_ckpt:
            print(f"  ? {k}  shape={list(ckpt_keys[k].shape)}")
    
    if shape_mismatch:
        print(f"\n{'─'*70}")
        print(f"SHAPE MISMATCHES:")
        print(f"{'─'*70}")
        for k, ms, cs in shape_mismatch:
            print(f"  ⚠ {k}  model={list(ms)} ckpt={list(cs)}")
    
    # Try to suggest mappings
    if missing_from_ckpt and unexpected_in_ckpt:
        print(f"\n{'─'*70}")
        print(f"SUGGESTED KEY MAPPINGS (checkpoint → model):")
        print(f"{'─'*70}")
        # Group by prefix
        for ckpt_key in unexpected_in_ckpt:
            # Find model keys with similar suffix or shape
            ckpt_shape = ckpt_keys[ckpt_key].shape
            candidates = [
                mk for mk in missing_from_ckpt 
                if model_keys[mk].shape == ckpt_shape
            ]
            if candidates:
                # Try to find best match by common suffix
                ckpt_suffix = ckpt_key.split('.')[-2:]
                best = None
                for c in candidates:
                    c_suffix = c.split('.')[-2:]
                    if c_suffix == ckpt_suffix:
                        best = c
                        break
                if best is None and len(candidates) <= 3:
                    best = candidates[0]
                if best:
                    print(f"  {ckpt_key} → {best}  shape={list(ckpt_shape)}")
    
    print(f"\n{'='*70}")
    return {
        'matched': matched,
        'missing': missing_from_ckpt,
        'unexpected': unexpected_in_ckpt,
        'shape_mismatch': shape_mismatch,
    }


if __name__ == "__main__":
    # Usage: 
    # from thinklab.models.multimodal.facebook.sam3.debug_keys import diagnose_weight_keys
    # result = diagnose_weight_keys(your_model, "/path/to/model.safetensors")
    pass
