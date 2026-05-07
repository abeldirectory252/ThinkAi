# 🏥 SAM3 (`facebook/sam3`)
> A high-resolution multimodal Segment Anything Model (SAM) with a ViT backbone, CLIP text encoder, and DETR-style mask decoder.


![Parameters](https://img.shields.io/badge/Parameters-~4B-4A90E2)
![Image Size](https://img.shields.io/badge/Image%20Size-896×896-F5A623)
![Architecture](https://img.shields.io/badge/Decoder-Gemma%203%20(34%20Layers%20+%20Sliding%20Window)-50C878)
![Vocab](https://img.shields.io/badge/Vocab-262,208-9B59B6)

## 📊 Model Specifications
Property	Value
Parameters	~840.3M (840,376,374)
Architecture	Sam3ViT + CLIP Text + DETR Decoder
Image Size	1008 × 1008
ViT Layers	32
Decoder Layers	6 (Detr Decoder)
Vocab Size	49,408


## 📁 Repository Structure
File	Purpose
model.py	Main SAM3 model architecture
decoder.py	DETR Mask Decoder and Geometry Encoder
vision_encoder.py	ViT Backbone and FPN Neck
projector.py	Text Projection Layers
image_processor.py	1008px image preprocessor
layers.py	SAM3 Attention, MLPs, Norms
builder.py	Builder + auto-registration



## 🔍 Tokenization Sequence
**Example Prompt Structure:** `14 tokens`
```text
🔍 INPUT TENSORS BREAKDOWN:
  - pixel_values        : shape [1, 3, 1008, 1008], dtype torch.float32
  - original_sizes      : shape [1, 2], dtype torch.int64
  - input_ids           : shape [1, 32], dtype torch.int64
  - attention_mask      : shape [1, 32], dtype torch.int64

🔍 TEXT TOKEN BREAKDOWN (Total tokens: 32)
Token    0 | ID:   49406 | String: '<|startoftext|>'
Token    1 | ID:   15615 | String: 'segment</w>'
Token    2 | ID:     518 | String: 'the</w>'
Token    3 | ID:     736 | String: 'red</w>'
Token    4 | ID:   14115 | String: 'object</w>'
Token    5 | ID:   49407 | String: '<|endoftext|>'
      ... [ Skipped 25 identical '<|endoftext|>' tokens ] ...
Token   31 | ID:   49407 | String: '<|endoftext|>'




### ⚙️ Configuration Values
| Key | Value |
|---|---|
| `model_type` | `` |
| `image_token_index` | `` |
| `boi_token_index` | `` |
| `eoi_token_index` | `` |
| `mm_tokens_per_image` | `` |

<details>
<summary>📊 Text & Vision Configs</summary>

**Text Configuration**
| Parameter | Value |
|---|---|
| `vocab_size` |  |
| `hidden_size` |  |
| `num_hidden_layers` |  |
| `num_attention_heads` | |
| `num_key_value_heads` | |
| `head_dim` |  |
| `intermediate_size` |  |
| `rms_norm_eps` |  |
| `query_pre_attn_scalar` |  |
| `sliding_window` |  |
| `hidden_activation` |  |
| `attn_logit_softcapping` | |
| `final_logit_softcapping` |  |

**Vision Configuration**
| Parameter | Value |
|---|---|
| `hidden_size` | |
| `image_size` |  |
| `patch_size` |  |
| `num_hidden_layers` | |
| `num_attention_heads` |  |
| `intermediate_size` |  |

**Layer Types Array**
```python

```
</details>
Embedding Tokens
Location: model.text_encoder.text_model.embeddings.token_embedding
Class: Embedding
num_embeddings: 49,408
embedding_dim: 1,024
🔗 Multi-Modal Projector (Vision Neck)
Class: Sam3VisionNeck
Parameters: 7,802,112 params
Modules:
position_encoding: Sam3SinePositionEmbedding
fpn_layers: 4 × Sam3FPNLayer
Flow Configuration:
In: [1, 1024, 72, 72]
Out: Multi-scale features ([1, 256, 288, 288], [1, 256, 144, 144], [1, 256, 72, 72], [1, 256, 36, 36])
🏗️ Attention Layers
Encoder Layers Found: model.detr_encoder (6 layers) Decoder Layers Found: model.detr_decoder (6 layers)

Layer 0-5 Details (Representative DETR Decoder Layer)

Property	Value
attn class	Sam3Attention
hidden_size	256
self_attn	Yes
text_cross_attn	Yes
vision_cross_attn	Yes
🔬 Forward Pass Trace
text
input_ids shape:    torch.Size([1, 32])
pixel_values shape: torch.Size([1, 3, 1008, 1008])
pixel_values dtype: torch.float32
vocab_size=49408
🧬 Vision + Embed Merge
Unlike autoregressive LLMs where vision tokens are concatenated sequentially, SAM3 utilizes a cross-attention fusion strategy:

text_embeds: Projected to shape [1, 512] from [1, 32, 1024]
vision_features: High-res multi-scale features via FPN Neck [1, 256, 288, 288]
Modality Merge: Text and vision features are deeply fused within the Sam3DetrDecoderLayer using independent text_cross_attn and vision_cross_attn blocks.

**Layer 0-2 Details (Representative)**
| Property | Value |
|---|---|
| `attn class` | `` |
| `num_heads` / `num_kv_heads` | `N/A` |
| `head_dim` | `` |
| `scaling` | `` |
| `sliding_window` | `` |
| `is_sliding` | `` |
| `softcapping` | `` |
| `has q_norm` | `` |
| `q_proj shape` | `` |
| `k_proj shape` | `` |

<details>


### 🔬 Forward Pass Trace
```text

```

### 🧬 Vision + Embed Merge

```


Sam3Model: Sam3Model (840,376,374 params)
     ↳ Flow: In: None ➔ Out: {pred_masks: [1, 200, 288, 288], pred_boxes: [1, 200, 4],...
  ├─ vision_encoder: Sam3VisionModel (454,038,784 params)
  ├─      ↳ Flow: In: [1, 3, 1008, 1008] ➔ Out: {last_hidden_state: [1, 5184, 1024]}
    ├─ backbone: Sam3ViTModel (446,236,672 params)
    ├─      ↳ Flow: In: [1, 3, 1008, 1008] ➔ Out: {last_hidden_state: [1, 5184, 1024]}
      ├─ embeddings: Sam3ViTEmbeddings (1,191,936 params)
      ├─      ↳ Flow: In: [1, 3, 1008, 1008] ➔ Out: [1, 5184, 1024]
        ├─ patch_embeddings: Sam3ViTPatchEmbeddings (602,112 params)
        ├─      ↳ Flow: In: [1, 3, 1008, 1008] ➔ Out: [1, 5184, 1024]
      └─ dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 5184, 1024] ➔ Out: [1, 5184, 1024]
      ├─ layer_norm: LayerNorm (2,048 params) | W: [1024]
      ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
    └─ layers: ModuleList (445,042,688 params)
        ├─ 0: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 1: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 2: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 3: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 4: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 5: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 6: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 7: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 8: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 9: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 10: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 11: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 12: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 13: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 14: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 15: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 16: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 17: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 18: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 19: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 20: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 21: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 22: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 23: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 24: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 25: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 26: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 27: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 28: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 29: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
        ├─ 30: Sam3ViTLayer (13,907,584 params)
        ├─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
      └─ 31: Sam3ViTLayer (13,907,584 params)
      └─      ↳ Flow: In: [1, 72, 72, 1024] ➔ Out: [1, 72, 72, 1024]
  └─ neck: Sam3VisionNeck (7,802,112 params)
  └─      ↳ Flow: In: [1, 1024, 72, 72] ➔ Out: (([1, 256, 288, 288], [1, 256, 144, 144], [1, 256, 72, 72...
      ├─ position_encoding: Sam3SinePositionEmbedding (0 params)
      ├─      ↳ Flow: In: (None, device, dtype) ➔ Out: [1, 256, 36, 36]
    └─ fpn_layers: ModuleList (7,802,112 params)
        ├─ 0: Sam3FPNLayer (3,278,080 params)
        ├─      ↳ Flow: In: [1, 1024, 72, 72] ➔ Out: [1, 256, 288, 288]
        ├─ 1: Sam3FPNLayer (2,819,072 params)
        ├─      ↳ Flow: In: [1, 1024, 72, 72] ➔ Out: [1, 256, 144, 144]
        ├─ 2: Sam3FPNLayer (852,480 params)
        ├─      ↳ Flow: In: [1, 1024, 72, 72] ➔ Out: [1, 256, 72, 72]
      └─ 3: Sam3FPNLayer (852,480 params)
      └─      ↳ Flow: In: [1, 1024, 72, 72] ➔ Out: [1, 256, 36, 36]
  ├─ text_encoder: CLIPTextModelWithProjection (353,462,272 params)
  ├─      ↳ Flow: In: None ➔ Out: {text_embeds: [1, 512], last_hidden_state: [1, 32, 1024]}
    ├─ text_model: CLIPTextTransformer (352,937,984 params)
    ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 32, 1024], pooler_output: [1, 1024]}
      ├─ embeddings: CLIPTextEmbeddings (50,626,560 params)
      ├─      ↳ Flow: In: None ➔ Out: [1, 32, 1024]
        ├─ token_embedding: Embedding (50,593,792 params) | W: [49408, 1024]
        ├─      ↳ Flow: In: [1, 32] ➔ Out: [1, 32, 1024]
      └─ position_embedding: Embedding (32,768 params) | W: [32, 1024]
      └─      ↳ Flow: In: [1, 32] ➔ Out: [1, 32, 1024]
      ├─ encoder: CLIPEncoder (302,309,376 params)
      ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 32, 1024]}
      └─ layers: ModuleList (302,309,376 params)
    └─ final_layer_norm: LayerNorm (2,048 params) | W: [1024]
    └─      ↳ Flow: In: [1, 32, 1024] ➔ Out: [1, 32, 1024]
  └─ text_projection: Linear (524,288 params) | W: [512, 1024]
  └─      ↳ Flow: In: [1, 1024] ➔ Out: [1, 512]
  ├─ text_projection: Linear (262,400 params) | W: [256, 1024]
  ├─      ↳ Flow: In: [1, 32, 1024] ➔ Out: [1, 32, 256]
  ├─ geometry_encoder: Sam3GeometryEncoder (8,083,456 params)
    ├─ position_encoding: Sam3SinePositionEmbedding (0 params)
    ├─ label_embed: Embedding (512 params) | W: [2, 256]
    ├─ cls_embed: Embedding (256 params) | W: [1, 256]
    ├─ boxes_direct_project: Linear (1,280 params) | W: [256, 4]
    ├─ boxes_pool_project: Conv2d (3,211,520 params) | W: [256, 256, 7, 7]
    ├─ boxes_pos_enc_project: Linear (66,304 params) | W: [256, 258]
    ├─ vision_layer_norm: LayerNorm (512 params) | W: [256]
    ├─ final_proj: Linear (65,792 params) | W: [256, 256]
    ├─ prompt_layer_norm: LayerNorm (512 params) | W: [256]
    ├─ layers: ModuleList (4,736,256 params)
      ├─ 0: Sam3GeometryEncoderLayer (1,578,752 params)
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─ dropout: Dropout (0 params)
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─ mlp: Sam3MLP (1,050,880 params)
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      ├─ 1: Sam3GeometryEncoderLayer (1,578,752 params)
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─ dropout: Dropout (0 params)
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─ mlp: Sam3MLP (1,050,880 params)
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
    └─ 2: Sam3GeometryEncoderLayer (1,578,752 params)
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─ dropout: Dropout (0 params)
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─ mlp: Sam3MLP (1,050,880 params)
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
  └─ output_layer_norm: LayerNorm (512 params) | W: [256]
  ├─ detr_encoder: Sam3DetrEncoder (9,472,512 params)
  ├─      ↳ Flow: In: None ➔ Out: {last_hidden_state: [1, 5184, 256], pos_embeds_flattened:...
  └─ layers: ModuleList (9,472,512 params)
      ├─ 0: Sam3DetrEncoderLayer (1,578,752 params)
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      ├─ 1: Sam3DetrEncoderLayer (1,578,752 params)
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      ├─ 2: Sam3DetrEncoderLayer (1,578,752 params)
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      ├─ 3: Sam3DetrEncoderLayer (1,578,752 params)
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      ├─ 4: Sam3DetrEncoderLayer (1,578,752 params)
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
    └─ 5: Sam3DetrEncoderLayer (1,578,752 params)
    └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ layer_norm1: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
        ├─ layer_norm2: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      └─ layer_norm3: LayerNorm (512 params) | W: [256]
      └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
  ├─ detr_decoder: Sam3DetrDecoder (11,575,093 params)
  ├─      ↳ Flow: In: None ➔ Out: {intermediate_hidden_states: [6, 1, 200, 256], reference_...
    ├─ layers: ModuleList (11,054,592 params)
      ├─ 0: Sam3DetrDecoderLayer (1,842,432 params)
      ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      ├─ 1: Sam3DetrDecoderLayer (1,842,432 params)
      ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      ├─ 2: Sam3DetrDecoderLayer (1,842,432 params)
      ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      ├─ 3: Sam3DetrDecoderLayer (1,842,432 params)
      ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      ├─ 4: Sam3DetrDecoderLayer (1,842,432 params)
      ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
    └─ 5: Sam3DetrDecoderLayer (1,842,432 params)
    └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ self_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ self_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ text_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ text_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn: Sam3Attention (263,168 params)
        ├─      ↳ Flow: In: None ➔ Out: ([1, 201, 256])
        ├─ vision_cross_attn_dropout: Dropout (0 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ vision_cross_attn_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp: Sam3MLP (1,050,880 params)
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
        ├─ mlp_layer_norm: LayerNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
      └─ mlp_dropout: Dropout (0 params)
      └─      ↳ Flow: In: [1, 201, 256] ➔ Out: [1, 201, 256]
    ├─ output_layer_norm: LayerNorm (512 params) | W: [256]
    ├─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
    ├─ box_head: Sam3DecoderMLP (132,612 params)
    ├─      ↳ Flow: In: [6, 1, 200, 256] ➔ Out: [6, 1, 200, 4]
      ├─ layer1: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [6, 1, 200, 256] ➔ Out: [6, 1, 200, 256]
      ├─ layer2: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [6, 1, 200, 256] ➔ Out: [6, 1, 200, 256]
    └─ layer3: Linear (1,028 params) | W: [4, 256]
    └─      ↳ Flow: In: [6, 1, 200, 256] ➔ Out: [6, 1, 200, 4]
    ├─ query_embed: Embedding (51,200 params) | W: [200, 256]
    ├─ reference_points: Embedding (800 params) | W: [200, 4]
    ├─ presence_token: Embedding (256 params) | W: [1, 256]
    ├─ presence_head: Sam3DecoderMLP (131,841 params)
    ├─      ↳ Flow: In: [1, 1, 256] ➔ Out: [1, 1, 1]
      ├─ layer1: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [1, 1, 256] ➔ Out: [1, 1, 256]
      ├─ layer2: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [1, 1, 256] ➔ Out: [1, 1, 256]
    └─ layer3: Linear (257 params) | W: [1, 256]
    └─      ↳ Flow: In: [1, 1, 256] ➔ Out: [1, 1, 1]
    ├─ presence_layer_norm: LayerNorm (512 params) | W: [256]
    ├─      ↳ Flow: In: [1, 1, 256] ➔ Out: [1, 1, 256]
    ├─ ref_point_head: Sam3DecoderMLP (197,120 params)
    ├─      ↳ Flow: In: [1, 200, 512] ➔ Out: [1, 200, 256]
      ├─ layer1: Linear (131,328 params) | W: [256, 512]
      ├─      ↳ Flow: In: [1, 200, 512] ➔ Out: [1, 200, 256]
    └─ layer2: Linear (65,792 params) | W: [256, 256]
    └─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
    ├─ box_rpb_embed_x: Sam3DecoderMLP (2,824 params)
    ├─      ↳ Flow: In: [1, 200, 72, 2] ➔ Out: [1, 200, 72, 8]
      ├─ layer1: Linear (768 params) | W: [256, 2]
      ├─      ↳ Flow: In: [1, 200, 72, 2] ➔ Out: [1, 200, 72, 256]
    └─ layer2: Linear (2,056 params) | W: [8, 256]
    └─      ↳ Flow: In: [1, 200, 72, 256] ➔ Out: [1, 200, 72, 8]
    ├─ box_rpb_embed_y: Sam3DecoderMLP (2,824 params)
    ├─      ↳ Flow: In: [1, 200, 72, 2] ➔ Out: [1, 200, 72, 8]
      ├─ layer1: Linear (768 params) | W: [256, 2]
      ├─      ↳ Flow: In: [1, 200, 72, 2] ➔ Out: [1, 200, 72, 256]
    └─ layer2: Linear (2,056 params) | W: [8, 256]
    └─      ↳ Flow: In: [1, 200, 72, 256] ➔ Out: [1, 200, 72, 8]
  └─ position_encoding: Sam3SinePositionEmbedding (0 params)
  ├─ mask_decoder: Sam3MaskDecoder (2,298,881 params)
  ├─      ↳ Flow: In: None ➔ Out: {pred_masks: [1, 200, 288, 288], semantic_seg: [1, 1, 288...
    ├─ pixel_decoder: Sam3PixelDecoder (1,771,776 params)
    ├─      ↳ Flow: In: ([1, 256, 288, 288], [1, 256, 144, 144], [1, 256, 72, 72]) ➔ Out: [1, 256, 288, 288]
      ├─ conv_layers: ModuleList (1,770,240 params)
        ├─ 0: Conv2d (590,080 params) | W: [256, 256, 3, 3]
        ├─      ↳ Flow: In: [1, 256, 144, 144] ➔ Out: [1, 256, 144, 144]
        ├─ 1: Conv2d (590,080 params) | W: [256, 256, 3, 3]
        ├─      ↳ Flow: In: [1, 256, 288, 288] ➔ Out: [1, 256, 288, 288]
      └─ 2: Conv2d (590,080 params) | W: [256, 256, 3, 3]
    └─ norms: ModuleList (1,536 params)
        ├─ 0: GroupNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 256, 144, 144] ➔ Out: [1, 256, 144, 144]
        ├─ 1: GroupNorm (512 params) | W: [256]
        ├─      ↳ Flow: In: [1, 256, 288, 288] ➔ Out: [1, 256, 288, 288]
      └─ 2: GroupNorm (512 params) | W: [256]
    ├─ mask_embedder: Sam3MaskEmbedder (197,376 params)
    ├─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
      ├─ layers: ModuleList (197,376 params)
        ├─ 0: Linear (65,792 params) | W: [256, 256]
        ├─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
        ├─ 1: Linear (65,792 params) | W: [256, 256]
        ├─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
      └─ 2: Linear (65,792 params) | W: [256, 256]
      └─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
    └─ activation: ReLU (0 params)
    └─      ↳ Flow: In: [1, 200, 256] ➔ Out: [1, 200, 256]
    ├─ instance_projection: Conv2d (65,792 params) | W: [256, 256, 1, 1]
    ├─      ↳ Flow: In: [1, 256, 288, 288] ➔ Out: [1, 256, 288, 288]
    ├─ semantic_projection: Conv2d (257 params) | W: [1, 256, 1, 1]
    ├─      ↳ Flow: In: [1, 256, 288, 288] ➔ Out: [1, 1, 288, 288]
    ├─ prompt_cross_attn: Sam3Attention (263,168 params)
    ├─      ↳ Flow: In: None ➔ Out: ([1, 5184, 256])
      ├─ q_proj: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
      ├─ k_proj: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 256]
      ├─ v_proj: Linear (65,792 params) | W: [256, 256]
      ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 256]
    └─ o_proj: Linear (65,792 params) | W: [256, 256]
    └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
    ├─ prompt_cross_attn_norm: LayerNorm (512 params) | W: [256]
    ├─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
  └─ prompt_cross_attn_dropout: Dropout (0 params)
  └─      ↳ Flow: In: [1, 5184, 256] ➔ Out: [1, 5184, 256]
└─ dot_product_scoring: Sam3DotProductScoring (1,182,976 params)
└─      ↳ Flow: In: None ➔ Out: [6, 1, 200, 1]
    ├─ text_mlp: Sam3DecoderMLP (1,050,880 params)
    ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 256]
      ├─ layer1: Linear (526,336 params) | W: [2048, 256]
      ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 2048]
    └─ layer2: Linear (524,544 params) | W: [256, 2048]
    └─      ↳ Flow: In: [1, 32, 2048] ➔ Out: [1, 32, 256]
    ├─ text_mlp_dropout: Dropout (0 params)
    ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 256]
    ├─ text_mlp_out_norm: LayerNorm (512 params) | W: [256]
    ├─      ↳ Flow: In: [1, 32, 256] ➔ Out: [1, 32, 256]
    ├─ text_proj: Linear (65,792 params) | W: [256, 256]
    ├─      ↳ Flow: In: [1, 256] ➔ Out: [1, 256]
  └─ query_proj: Linear (65,792 params) | W: [256, 256]
  └─      ↳ Flow: In: [6, 1, 200, 256] ➔ Out: [6, 1, 200, 256]