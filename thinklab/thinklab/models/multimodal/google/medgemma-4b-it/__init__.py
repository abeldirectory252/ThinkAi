"""
google/medgemma-4b-it — MedGemma 4B Instruct-Tuned

HuggingFace: https://huggingface.co/google/medgemma-4b-it
Architecture: SigLIP ViT → AvgPool Projector → Gemma 3 Decoder (QK-norm)
"""

MODEL_ID = "google/medgemma-4b-it"
MODEL_TYPE = "gemma3"
MODALITY = "multimodal"

from .model import MedGemma
from .builder import build_medgemma
