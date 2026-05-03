"""Abstract base class for all ThinkLab models."""
import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, abc.ABC):
    """Abstract base for all ThinkLab models with memory-aware utilities."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.model_dtype = dtype
        self._device_map: Dict[str, torch.device] = {}

    # ── Memory helpers ──────────────────────────────────────────────
    @staticmethod
    def get_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_mem / 1024**2
        return 0.0

    @staticmethod
    def get_free_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return free / 1024**2
        return 0.0

    def estimate_param_memory_mb(self) -> float:
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        return total_bytes / 1024**2

    def smart_device(self, prefer: str = "cuda") -> torch.device:
        """Pick best device given available memory."""
        if prefer == "cuda" and torch.cuda.is_available():
            needed = self.estimate_param_memory_mb()
            free = self.get_free_gpu_memory_mb()
            if free > needed * 1.3:          # 30 % headroom
                return torch.device("cuda")
            logger.warning(
                "Not enough GPU memory (%.0f MB free, %.0f MB needed). "
                "Falling back to CPU.", free, needed,
            )
        return torch.device("cpu")

    # ── Layer-wise offloading ───────────────────────────────────────
    def offload_layers_to_cpu(
        self, layers: nn.ModuleList, keep_on_gpu: int = 2
    ) -> None:
        """Keep only *keep_on_gpu* layers on GPU, rest on CPU."""
        for i, layer in enumerate(layers):
            if i < len(layers) - keep_on_gpu:
                layer.to("cpu")
                self._device_map[f"layer.{i}"] = torch.device("cpu")
            else:
                if torch.cuda.is_available():
                    layer.to("cuda")
                self._device_map[f"layer.{i}"] = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

    def layer_forward_with_offload(
        self, layer: nn.Module, *args, target_device: str = "cuda", **kw
    ):
        """Move layer to GPU, run forward, move back to CPU."""
        original = next(layer.parameters()).device
        dev = torch.device(target_device if torch.cuda.is_available() else "cpu")
        layer.to(dev)
        args = tuple(a.to(dev) if isinstance(a, torch.Tensor) else a for a in args)
        kw = {
            k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in kw.items()
        }
        out = layer(*args, **kw)
        layer.to(original)
        return out

    # ── Abstract interface ──────────────────────────────────────────
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def load_weights(self, path: Path) -> None:
        ...
