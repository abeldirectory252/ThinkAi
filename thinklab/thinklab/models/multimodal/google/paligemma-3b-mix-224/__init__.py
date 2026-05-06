"""
google/paligemma-3b-mix-224 — PaliGemma 3B
Architecture: SigLIP ViT → Linear Projector → Gemma 1 Decoder
"""
MODEL_ID = "google/paligemma-3b-mix-224"
MODEL_TYPE = "gemma1"
MODALITY = "multimodal"

from .model import PaliGemma
from .builder import build_paligemma
