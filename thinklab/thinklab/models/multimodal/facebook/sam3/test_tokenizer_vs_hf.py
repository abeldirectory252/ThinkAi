"""
SAM3 Tokenizer Verification: Compare ThinkLab BPE output vs HF CLIPTokenizer.
Run on Kaggle where both the HF model and your tokenizer files are available.

Expected HF token IDs (from comparison run):
  'nose'     → [49406, 8231, 49407, pad...]    mask: [1,1,1,0...]
  'Ear'      → [49406, 8373, 49407, pad...]    mask: [1,1,1,0...]
  'Eye'      → [49406, 3272, 49407, pad...]    mask: [1,1,1,0...]
  'cat eye'  → [49406, 2368, 3272, 49407, pad...] mask: [1,1,1,1,0...]
  'cat ear'  → [49406, 2368, 8373, 49407, pad...] mask: [1,1,1,1,0...]
  'cat nose' → [49406, 2368, 8231, 49407, pad...] mask: [1,1,1,1,0...]
"""
import sys
import os

# === Option A: If running on Kaggle with HF available ===
try:
    from transformers import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    hf_tokenizer = processor.tokenizer
    HF_AVAILABLE = True
    print("✅ HF CLIPTokenizer loaded")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HF not available, using hardcoded reference IDs")

# === Option B: Load your tokenizer ===
# Adjust path as needed for your Kaggle setup
try:
    # If running from the project root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer import Sam3Tokenizer

    # Point to where your vocab.json + merges.txt are
    # On Kaggle this might be in the model weights directory
    TOKENIZER_PATH = "/kaggle/working/sam3_weights"  # ADJUST THIS
    your_tokenizer = Sam3Tokenizer(TOKENIZER_PATH)
    YOUR_AVAILABLE = True
    print("✅ Your Sam3Tokenizer loaded")
except Exception as e:
    YOUR_AVAILABLE = False
    print(f"⚠️  Your tokenizer not loaded: {e}")

# === Reference data from HF run ===
HF_REFERENCE = {
    "nose":     {"ids": [49406, 8231, 49407], "mask_len": 3},
    "Ear":      {"ids": [49406, 8373, 49407], "mask_len": 3},
    "Eye":      {"ids": [49406, 3272, 49407], "mask_len": 3},
    "cat eye":  {"ids": [49406, 2368, 3272, 49407], "mask_len": 4},
    "cat ear":  {"ids": [49406, 2368, 8373, 49407], "mask_len": 4},
    "cat nose": {"ids": [49406, 2368, 8231, 49407], "mask_len": 4},
}

print("\n" + "=" * 70)
print("TOKENIZER COMPARISON")
print("=" * 70)

test_words = ["nose", "Ear", "Eye", "cat eye", "cat ear", "cat nose"]
all_match = True

for word in test_words:
    print(f"\n--- '{word}' ---")

    # HF reference
    ref = HF_REFERENCE[word]
    print(f"  HF reference IDs: {ref['ids']} (valid tokens: {ref['mask_len']})")

    # HF live (if available)
    if HF_AVAILABLE:
        hf_out = hf_tokenizer(word, return_tensors="pt", padding="max_length", max_length=32)
        hf_ids = hf_out["input_ids"][0].tolist()
        hf_mask = hf_out["attention_mask"][0].tolist()
        hf_valid = sum(hf_mask)
        print(f"  HF live IDs:      {hf_ids[:hf_valid]} (valid tokens: {hf_valid})")

    # Your tokenizer
    if YOUR_AVAILABLE:
        your_ids = your_tokenizer.encode(word)
        your_mask = your_tokenizer.get_attention_mask(your_ids)
        your_valid = sum(your_mask)
        your_active = your_ids[:your_valid]
        print(f"  YOUR IDs:         {your_active} (valid tokens: {your_valid})")
        print(f"  YOUR full IDs:    {your_ids}")
        print(f"  YOUR full mask:   {your_mask}")

        # Compare
        match = your_active == ref["ids"]
        status = "✅ MATCH" if match else "❌ MISMATCH"
        print(f"  Status: {status}")
        if not match:
            all_match = False
            print(f"    Expected: {ref['ids']}")
            print(f"    Got:      {your_active}")
    else:
        print(f"  YOUR IDs: (tokenizer not loaded)")

print("\n" + "=" * 70)
if YOUR_AVAILABLE:
    if all_match:
        print("✅ ALL TOKENIZATIONS MATCH HF!")
    else:
        print("❌ SOME TOKENIZATIONS DON'T MATCH — BPE merges may differ")
        print("   Check that vocab.json and merges.txt are from the SAM3 CLIP model")
print("=" * 70)
