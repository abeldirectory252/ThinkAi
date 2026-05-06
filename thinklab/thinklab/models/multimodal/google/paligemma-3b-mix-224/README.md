
```markdown
# 🌟 PaliGemma 3B (`google/paligemma-3b-mix-224`)
> A compact, high-performance vision-language model combining SigLIP vision encoding with the Gemma 1 decoder architecture.

![Parameters](https://img.shields.io/badge/Parameters-~3B-4A90E2)
![Image Size](https://img.shields.io/badge/Image%20Size-224×224-F5A623)
![Decoder Layers](https://img.shields.io/badge/Decoder%20Layers-18-50C878)
![License](https://img.shields.io/badge/License-Apache%202.0-2ECC71)

## 📊 Model Specifications
| Property          | Value                          |
|-------------------|--------------------------------|
| **Parameters**    | ~3B                            |
| **Architecture**  | SigLIP ViT + Gemma 1 Decoder   |
| **Image Size**    | 224 × 224                      |
| **Decoder Layers**| 18 (no QK-norm)                |
| **Vocab Size**    | 257,216                        |

## 🔍 Token Sequence Breakdown
**Total Tokens:** `65,541`

```text
Token 0     | ID: 257152 | String: '<image>'
      ... [ Skipped 65534 identical '<image>' tokens ] ...
Token 65535 | ID: 257152 | String: '<image>'
Token 65536 | ID:      2 | String: '<bos>'
Token 65537 | ID:  50721 | String: 'Describe'
Token 65538 | ID:    736 | String: '▁this'
Token 65539 | ID:   2416 | String: '▁image'
Token 65540 | ID:    108 | String: '\n'
```

## 🌳 Architecture & Information Flow
<details open>
<summary>📂 Click to expand full model tree</summary>

```text
PaliGemmaModel: PaliGemmaForConditionalGeneration (2,923,466,480 params)
     ↳ Flow: In: None ➔ Out: {loss: [], logits: [1, 517, 257216], image_hidd...
  ├─ model: PaliGemmaModel (2,923,466,480 params)
  ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 517, 2048], image_hidde...
    ├─ vision_tower: SiglipVisionModel (412,442,352 params)
    ├─      ↳ Flow: In: [1, 3, 224, 224] ➔ Out: {last_hidden_state: [1, 256, 1152]}
    └─ vision_model: SiglipVisionTransformer (412,442,352 params)
    └─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 256, 1152]}
        ├─ embeddings: SiglipVisionEmbeddings (973,440 params)
        ├─      ↳ Flow: In: [1, 3, 224, 224] ➔ Out: [1, 256, 1152]
        ├─ encoder: SiglipEncoder (411,466,608 params)
        ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 256, 1152]}
      └─ post_layernorm: LayerNorm (2,304 params) | W: [1152]
      └─      ↳ Flow: In: [1, 256, 1152] ➔ Out: [1, 256, 1152]
    ├─ multi_modal_projector: PaliGemmaMultiModalProjector (2,361,344 params)
    ├─      ↳ Flow: In: [1, 256, 1152] ➔ Out: [1, 256, 2048]
    └─ linear: Linear (2,361,344 params) | W: [2048, 1152]
    └─      ↳ Flow: In: [1, 256, 1152] ➔ Out: [1, 256, 2048]
  └─ language_model: GemmaModel (2,508,662,784 params)
  └─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 517, 2048]}
      ├─ embed_tokens: Embedding (526,778,368 params) | W: [257216, 2048]
      ├─      ↳ Flow: In: [1, 517] ➔ Out: [1, 517, 2048]
      ├─ layers: ModuleList (1,981,882,368 params)
        ├─ 0: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 1: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 2: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 3: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 4: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 5: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 6: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 7: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 8: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 9: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 10: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 11: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 12: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 13: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 14: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 15: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
        ├─ 16: GemmaDecoderLayer (110,104,576 params)
        ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
      └─ 17: GemmaDecoderLayer (110,104,576 params)
      └─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
      ├─ norm: GemmaRMSNorm (2,048 params) | W: [2048]
      ├─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 2048]
    └─ rotary_emb: GemmaRotaryEmbedding (0 params)
    └─      ↳ Flow: In: [1, 517, 2048] ➔ Out: ([1, 517, 256], [1, 517, 256])
└─ lm_head: Linear (526,778,368 params) | W: [257216, 2048]
└─      ↳ Flow: In: [1, 517, 2048] ➔ Out: [1, 517, 257216]
```
</details>

## 🛠️ Repository Usage
### 📦 Installation
```bash
pip install transformers torch accelerate
```

### 💻 Inference Example
```python
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

model_name = "google/paligemma-3b-mix-224"
processor = PaliGemmaProcessor.from_pretrained(model_name)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)

# Load & preprocess
image = Image.open("sample.jpg").convert("RGB")
prompt = "Describe this image"
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
    
print(processor.decode(output[0], skip_special_tokens=True))
```

## 📜 Citation & License
- **Original Model**: [Google PaliGemma](https://ai.google.dev/gemma)
- **License**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Citation**:
  ```bibtex
  @misc{google2024paligemma,
    title={PaliGemma: A Versatile Vision-Language Model},
    author={Google DeepMind},
    year={2024},
    url={https://ai.google.dev/gemma}
  }
  ```

