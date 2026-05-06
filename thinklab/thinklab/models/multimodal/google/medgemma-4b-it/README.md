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


## Tokenization
==================================================
🔍 TOKEN BY TOKEN BREAKDOWN (First 30 tokens):
==================================================
Token  0 | ID: 2       | String: '<bos>'
Token  1 | ID: 105     | String: '<start_of_turn>'
Token  2 | ID: 2364    | String: 'user'
Token  3 | ID: 107     | String: '\n'
Token  4 | ID: 3048    | String: 'You'
Token  5 | ID: 659     | String: ' are'
Token  6 | ID: 614     | String: ' an'
Token  7 | ID: 7710    | String: ' expert'
Token  8 | ID: 4574    | String: ' radi'
Token  9 | ID: 16097   | String: 'ologist'
Token 10 | ID: 236761  | String: '.'
Token 11 | ID: 108     | String: '\n\n'
Token 12 | ID: 82858   | String: 'Describe'
Token 13 | ID: 672     | String: ' this'
Token 14 | ID: 1684    | String: ' X'
Token 15 | ID: 236772  | String: '-'
Token 16 | ID: 1254    | String: 'ray'
Token 17 | ID: 108     | String: '\n\n'
Token 18 | ID: 255999  | String: '<start_of_image>'
Token 19 | ID: 262144  | String: '<image_soft_token>'
Token 20 | ID: 262144  | String: '<image_soft_token>'
Token 21 | ID: 262144  | String: '<image_soft_token>'
Token 22 | ID: 262144  | String: '<image_soft_token>'
Token 23 | ID: 262144  | String: '<image_soft_token>'
Token 24 | ID: 262144  | String: '<image_soft_token>'
Token 25 | ID: 262144  | String: '<image_soft_token>'
Token 26 | ID: 262144  | String: '<image_soft_token>'
Token 27 | ID: 262144  | String: '<image_soft_token>'
Token 28 | ID: 262144  | String: '<image_soft_token>'
Token 29 | ID: 262144  | String: '<image_soft_token>'

==================================================
🔍 TOKEN BY TOKEN BREAKDOWN (Last 10 tokens):
==================================================
Token -10 | ID: 262144  | String: '<image_soft_token>'
Token -9  | ID: 262144  | String: '<image_soft_token>'
Token -8  | ID: 262144  | String: '<image_soft_token>'
Token -7  | ID: 256000  | String: '<end_of_image>'
Token -6  | ID: 108     | String: '\n\n'
Token -5  | ID: 106     | String: '<end_of_turn>'
Token -4  | ID: 107     | String: '\n'
Token -3  | ID: 105     | String: '<start_of_turn>'
Token -2  | ID: 4368    | String: 'model'
Token -1  | ID: 107     | String: '\n'
Token  0 | ID: 2       | String: '<bos>'
Token  1 | ID: 105     | String: '<start_of_turn>'
Token  2 | ID: 2364    | String: 'user'
Token  3 | ID: 107     | String: '\n'
Token  4 | ID: 3048    | String: 'You'
Token  5 | ID: 659     | String: ' are'
Token  6 | ID: 614     | String: ' an'
Token  7 | ID: 7710    | String: ' expert'
Token  8 | ID: 4574    | String: ' radi'
Token  9 | ID: 16097   | String: 'ologist'
Token 10 | ID: 236761  | String: '.'
Token 11 | ID: 108     | String: '\n\n'
Token 12 | ID: 82858   | String: 'Describe'
Token 13 | ID: 672     | String: ' this'
Token 14 | ID: 1684    | String: ' X'
Token 15 | ID: 236772  | String: '-'
Token 16 | ID: 1254    | String: 'ray'
Token 17 | ID: 108     | String: '\n\n'
Token 18 | ID: 255999  | String: '<start_of_image>'
Token 19 | ID: 262144  | String: '<image_soft_token>'
Token 20 | ID: 262144  | String: '<image_soft_token>'
Token 21 | ID: 262144  | String: '<image_soft_token>'
Token 22 | ID: 262144  | String: '<image_soft_token>'
Token 23 | ID: 262144  | String: '<image_soft_token>'
Token 24 | ID: 262144  | String: '<image_soft_token>'
Token 25 | ID: 262144  | String: '<image_soft_token>'
Token 26 | ID: 262144  | String: '<image_soft_token>'
Token 27 | ID: 262144  | String: '<image_soft_token>'
Token 28 | ID: 262144  | String: '<image_soft_token>'
Token 29 | ID: 262144  | String: '<image_soft_token>'

==================================================
🔍 TOKEN BY TOKEN BREAKDOWN (Last 10 tokens):
==================================================
Token -10 | ID: 262144  | String: '<image_soft_token>'
Token -9  | ID: 262144  | String: '<image_soft_token>'
Token -8  | ID: 262144  | String: '<image_soft_token>'
Token -7  | ID: 256000  | String: '<end_of_image>'
Token -6  | ID: 108     | String: '\n\n'
Token -5  | ID: 106     | String: '<end_of_turn>'
Token -4  | ID: 107     | String: '\n'
Token -3  | ID: 105     | String: '<start_of_turn>'
Token -2  | ID: 4368    | String: 'model'
Token -1  | ID: 107     | String: '\n'



======================================================================
📐 MODEL TREE (first 2 levels)
======================================================================
  model: Gemma3Model
    └─ vision_tower: SiglipVisionModel
        └─ vision_model: SiglipVisionTransformer
    └─ multi_modal_projector: Gemma3MultiModalProjector
        └─ mm_soft_emb_norm: Gemma3RMSNorm
        └─ avg_pool: AvgPool2d
    └─ language_model: Gemma3TextModel
        └─ embed_tokens: Gemma3TextScaledWordEmbedding
        └─ layers: ModuleList
        └─ norm: Gemma3RMSNorm
        └─ rotary_emb: Gemma3RotaryEmbedding
  lm_head: Linear

======================================================================
⚙️  CONFIG VALUES
======================================================================
  model_type:          gemma3
  image_token_index:   262144
  boi_token_index:     255999
  eoi_token_index:     256000
  mm_tokens_per_image: 256

  [Text Config]
    vocab_size:              262208
    hidden_size:             2560
    num_hidden_layers:       34
    num_attention_heads:     8
    num_key_value_heads:     4
    head_dim:                256
    intermediate_size:       10240
    rms_norm_eps:            1e-06
    query_pre_attn_scalar:   256
    attn_logit_softcapping:  None
    final_logit_softcapping: None
    sliding_window:          1024
    layer_types:             ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention']
    hidden_activation:       gelu_pytorch_tanh

  [Vision Config]
    hidden_size:         1152
    image_size:          896
    patch_size:          14
    num_hidden_layers:   27
    num_attention_heads: 16
    intermediate_size:   4304

======================================================================
📝 FINDING EMBED_TOKENS
======================================================================
  Found: model.language_model.embed_tokens
    Class:          Gemma3TextScaledWordEmbedding
    num_embeddings: 262208
    embedding_dim:  2560
    embed_scale:    50.5
    scalar_value:   50.59644256269407

======================================================================
🔗 MULTI-MODAL PROJECTOR
======================================================================
  Class: Gemma3MultiModalProjector
  param: mm_input_projection_weight shape=[1152, 2560] dtype=torch.bfloat16
  param: mm_soft_emb_norm.weight shape=[1152] dtype=torch.bfloat16
  module: mm_soft_emb_norm: Gemma3RMSNorm
  module: avg_pool: AvgPool2d
  AvgPool2d kernel: 4
  patches_per_image: 64
  tokens_per_side: 16

======================================================================
🏗️  ATTENTION LAYERS
======================================================================
  Found decoder layers at: model.language_model (34 layers)

  Layer 0: Gemma3DecoderLayer
    attn class:     Gemma3Attention
    num_heads:      N/A
    num_kv_heads:   N/A
    head_dim:       256
    scaling:        0.0625
    is_sliding:     True
    sliding_window: 1024
    softcapping:    None
    has q_norm:     True
    q_proj shape:   [2048, 2560]
    k_proj shape:   [1024, 2560]

  Layer 1: Gemma3DecoderLayer
    attn class:     Gemma3Attention
    num_heads:      N/A
    num_kv_heads:   N/A
    head_dim:       256
    scaling:        0.0625
    is_sliding:     True
    sliding_window: 1024
    softcapping:    None
    has q_norm:     True
    q_proj shape:   [2048, 2560]
    k_proj shape:   [1024, 2560]

  Layer 2: Gemma3DecoderLayer
    attn class:     Gemma3Attention
    num_heads:      N/A
    num_kv_heads:   N/A
    head_dim:       256
    scaling:        0.0625
    is_sliding:     True
    sliding_window: 1024
    softcapping:    None
    has q_norm:     True
    q_proj shape:   [2048, 2560]
    k_proj shape:   [1024, 2560]

  [All Layer Types]
     0: sliding=True, window=1024
     1: sliding=True, window=1024
     2: sliding=True, window=1024
     3: sliding=True, window=1024
     4: sliding=True, window=1024
     5: sliding=False, window=None
     6: sliding=True, window=1024
     7: sliding=True, window=1024
     8: sliding=True, window=1024
     9: sliding=True, window=1024
    10: sliding=True, window=1024
    11: sliding=False, window=None
    12: sliding=True, window=1024
    13: sliding=True, window=1024
    14: sliding=True, window=1024
    15: sliding=True, window=1024
    16: sliding=True, window=1024
    17: sliding=False, window=None
    18: sliding=True, window=1024
    19: sliding=True, window=1024
    20: sliding=True, window=1024
    21: sliding=True, window=1024
    22: sliding=True, window=1024
    23: sliding=False, window=None
    24: sliding=True, window=1024
    25: sliding=True, window=1024
    26: sliding=True, window=1024
    27: sliding=True, window=1024
    28: sliding=True, window=1024
    29: sliding=False, window=None
    30: sliding=True, window=1024
    31: sliding=True, window=1024
    32: sliding=True, window=1024
    33: sliding=True, window=1024

======================================================================
🔬 FORWARD PASS TRACE
======================================================================
  input_ids shape:    torch.Size([1, 282])
  pixel_values shape: torch.Size([1, 3, 896, 896])
  pixel_values dtype: torch.float32
  pixel_values range: [-1.0000, 1.0000]
  token_type_ids unique: [0, 1]
  image positions: 19 to 274 (256 tokens)

  image_token_id=262144, count_in_input=256
  vocab_size=262208
  ⚠️  OOB check: 262144 >= 262208 = False

======================================================================
🧬 VISION + EMBED MERGE
======================================================================
  Replaced 256 OOV tokens with 0
  text_embeds: shape=[1, 282, 2560], range=[-24.500, 28.750], NaN=False
  vision out:  shape=[1, 4096, 1152]
  projected:   shape=[1, 256, 2560], range=[-4.188, 8.625], NaN=False
  merged:      NaN=False, range=[-24.500, 28.750]

🔍 TOKEN BREAKDOWN (Total tokens: 14)
Token    0 | ID:       2 | String: '<bos>'
Token    1 | ID:     105 | String: '<start_of_turn>'
Token    2 | ID:    2364 | String: 'user'
Token    3 | ID:     107 | String: '\n'
Token    4 | ID:  255999 | String: '<start_of_image>'
Token    5 | ID:   82858 | String: 'Describe'
Token    6 | ID:     672 | String: '▁this'
Token    7 | ID:    5526 | String: '▁medical'
Token    8 | ID:    2471 | String: '▁image'
Token    9 | ID:     106 | String: '<end_of_turn>'
Token   10 | ID:     107 | String: '\n'
Token   11 | ID:     105 | String: '<start_of_turn>'
Token   12 | ID:    4368 | String: 'model'
Token   13 | ID:     107 | String: '\n'



  MedGemmaModel: Gemma3ForConditionalGeneration (4,300,079,472 params)
     ↳ Flow: In: None ➔ Out: {logits: [1, 14, 262208]}
  ├─ model: Gemma3Model (4,300,079,472 params)
  ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 14, 2560]}
    ├─ vision_tower: SiglipVisionModel (416,866,032 params)
    └─ vision_model: SiglipVisionTransformer (416,866,032 params)
        ├─ embeddings: SiglipVisionEmbeddings (5,397,120 params)
        ├─ encoder: SiglipEncoder (411,466,608 params)
      └─ post_layernorm: LayerNorm (2,304 params) | W: [1152]
    ├─ multi_modal_projector: Gemma3MultiModalProjector (2,950,272 params)
      ├─ mm_soft_emb_norm: Gemma3RMSNorm (1,152 params) | W: [1152]
    └─ avg_pool: AvgPool2d (0 params)
  └─ language_model: Gemma3TextModel (3,880,263,168 params)
  └─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 14, 2560]}
      ├─ embed_tokens: Gemma3TextScaledWordEmbedding (671,252,480 params) | W: [262208, 2560]
      ├─      ↳ Flow: In: [1, 14] ➔ Out: [1, 14, 2560]
      ├─ layers: ModuleList (3,209,008,128 params)
        ├─ 0: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 1: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 2: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 3: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 4: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 5: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 6: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 7: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 8: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 9: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 10: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 11: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 12: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 13: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 14: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 15: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 16: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 17: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 18: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 19: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 20: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 21: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 22: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 23: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 24: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 25: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 26: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 27: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 28: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 29: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 30: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 31: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
        ├─ 32: Gemma3DecoderLayer (94,382,592 params)
        ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
      └─ 33: Gemma3DecoderLayer (94,382,592 params)
      └─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
      ├─ norm: Gemma3RMSNorm (2,560 params) | W: [2560]
      ├─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 2560]
    └─ rotary_emb: Gemma3RotaryEmbedding (0 params)
    └─      ↳ Flow: In: ([1, 14, 2560], [1, 14]) ➔ Out: ([1, 14, 256], [1, 14, 256])
└─ lm_head: Linear (671,252,480 params) | W: [262208, 2560]
└─      ↳ Flow: In: [1, 14, 2560] ➔ Out: [1, 14, 262208]