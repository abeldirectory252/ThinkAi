"""
facebook/sam3 — Segment Anything Model 3

Architecture: Sam3ViT (32L) → FPN Neck → CLIP Text → DETR Encoder/Decoder → Mask Decoder
Output: Segmentation masks + bounding boxes (NOT text generation)
"""
MODEL_ID = "facebook/sam3"
MODEL_TYPE = "sam3"
MODALITY = "multimodal"
