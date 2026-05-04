"""ThinkLab — Pure PyTorch AI Research Framework."""
__version__ = "0.1.0"

from .registry import load_llm, list_models, register_model
from .inference import InferenceEngine
from .trainer import Trainer

# Auto-register built-in models
import thinklab.model_builders  # noqa: F401
