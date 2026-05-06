# PaliGemma 3B (`google/paligemma-3b-mix-224`)

| Property | Value |
|---|---|
| **Parameters** | ~3B |
| **Architecture** | SigLIP ViT + Gemma 1 Decoder |
| **Image Size** | 224 × 224 |
| **Decoder Layers** | 18, no QK-norm |
| **Vocab Size** | 257216 |


======================================================================
🔍 TOKEN BY TOKEN BREAKDOWN (Total tokens: 65541)
======================================================================
Token    0 | ID:  257152 | String: '<image>'
      ... [ Skipped 65534 identical '<image>' tokens ] ...
Token 65535 | ID:  257152 | String: '<image>'
Token 65536 | ID:       2 | String: '<bos>'
Token 65537 | ID:   50721 | String: 'Describe'
Token 65538 | ID:     736 | String: '▁this'
Token 65539 | ID:    2416 | String: '▁image'
Token 65540 | ID:     108 | String: '\n'
======================================================================

==================================================================================================
🌳 COMPLETE MODEL TREE (WITH VECTOR SIZES & INFORMATION FLOW)
=================================================================================================
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