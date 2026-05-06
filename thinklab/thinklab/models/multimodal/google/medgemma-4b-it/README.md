# MedGemma 4B (`google/medgemma-4b-it`)

| Property | Value |
|---|---|
| **Parameters** | ~4B |
| **Architecture** | SigLIP ViT + Gemma 3 Decoder |
| **Image Size** | 896 × 896 |
| **Decoder Layers** | 34, QK-norm, sliding window |
| **Vocab Size** | 262144 |

## Files

| File | Purpose |
|---|---|
| `model.py` | MedGemma multimodal model |
| `decoder.py` | Gemma 3 causal LM decoder |
| `vision_encoder.py` | SigLIP ViT |
| `projector.py` | RMSNorm + AvgPool projector |
| `tokenizer.py` | Gemma 3 chat template tokenizer |
| `image_processor.py` | 896px SigLIP preprocessor |
| `layers.py` | RMSNorm, KVCache, RoPE |
| `builder.py` | Builder + auto-registration |
