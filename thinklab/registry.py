"""
Model registry + factory.
Each model is independent — no shared base, no conflicts.

Usage:
    import thinklab
    model = thinklab.load_llm("google/medgemma-4b-it", token="hf_xxx")
    model = thinklab.load_llm("google/paligemma-3b-pt-224-128", token="hf_xxx")
"""
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger("thinklab.registry")

# ── Registry storage ────────────────────────────────────────────────
_MODEL_REGISTRY: Dict[str, dict] = {}


def register_model(
    pattern: str,
    builder: Callable,
    arch: str,
    defaults: Optional[dict] = None,
):
    """Register a model builder.
    pattern: substring matched against repo_id (e.g. 'medgemma', 'paligemma')
    builder: callable(save_dir, config, **kwargs) -> model instance
    """
    _MODEL_REGISTRY[pattern] = {
        "builder": builder,
        "arch": arch,
        "defaults": defaults or {},
    }


def list_models() -> list:
    """List all registered model patterns."""
    return [
        {"pattern": k, "arch": v["arch"]}
        for k, v in _MODEL_REGISTRY.items()
    ]


def _match_registry(model_name: str) -> Optional[dict]:
    for pattern, entry in _MODEL_REGISTRY.items():
        if pattern in model_name.lower():
            return entry
    return None


# ── Main factory ────────────────────────────────────────────────────
def load_llm(
    model_name: str,
    save_dir: Optional[str] = None,
    token: Optional[str] = None,
    tokenizer: Optional[str] = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    IsAgent: bool = False,
    isExplainerEnabled: bool = True,
    explainer_methods: list = None,
    max_memory_gb: Optional[float] = None,
    **kwargs,
) -> "ThinkLabModel":
    """
    Load any supported model by repo ID.

    Args:
        model_name: HuggingFace repo ID (e.g. "google/medgemma-4b-it")
        save_dir: where to cache weights (default: ./weights/<model>)
        token: HuggingFace access token
        tokenizer: custom tokenizer path (auto-detected if empty)
        dtype: "bfloat16", "float16", or "float32"
        device: "auto", "cpu", or "cuda"
        IsAgent: enable agentic mode (future)
        isExplainerEnabled: attach Grad-CAM + LIME explainers
        explainer_methods: list of explainers, default ["grad_cam", "lime"]
        max_memory_gb: GPU memory budget (None = use all available)

    Returns:
        ThinkLabModel wrapper with .inference(), .explain(), .train()
    """
    from .weights import HuggingFaceDownloader

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pt_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Resolve save directory
    if save_dir is None:
        slug = model_name.replace("/", "_")
        save_dir = f"./weights/{slug}"
    save_path = Path(save_dir)

    # Download weights
    logger.info("Downloading %s → %s", model_name, save_path)
    dl = HuggingFaceDownloader(model_name, save_path, token=token)
    dl.download_model()

    # Read config
    import json
    config = {}
    cfg_file = save_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            config = json.load(f)

    # Find matching builder
    entry = _match_registry(model_name)
    if entry is None:
        raise ValueError(
            f"No registered builder for '{model_name}'.\n"
            f"Registered: {[e['pattern'] for e in _MODEL_REGISTRY.values()]}\n"
            f"Use thinklab.register_model() to add support."
        )

    logger.info("Matched arch: %s", entry["arch"])

    # Build the raw model
    builder_kwargs = {**entry["defaults"], **kwargs}
    raw_model = entry["builder"](<
        save_dir=save_path,
        config=config,
        dtype=pt_dtype,
        device=device,
        max_memory_gb=max_memory_gb,
        **builder_kwargs,
    >)

    # Resolve tokenizer
    tok_path = tokenizer if tokenizer else str(save_path / "tokenizer.model")

    # Wrap in ThinkLabModel
    return ThinkLabModel(
        model=raw_model,
        model_name=model_name,
        arch=entry["arch"],
        tokenizer_path=tok_path,
        config=config,
        dtype=pt_dtype,
        device=device,
        IsAgent=IsAgent,
        isExplainerEnabled=isExplainerEnabled,
        explainer_methods=explainer_methods or ["grad_cam", "lime"],
    )


# ── Unified wrapper ─────────────────────────────────────────────────
class ThinkLabModel:
    """
    Unified model wrapper. Holds the raw model + tokenizer + explainers.
    Provides .inference(), .explain(), .train() as separate concerns.
    """

    def __init__(self, model, model_name, arch, tokenizer_path,
                 config, dtype, device, IsAgent, isExplainerEnabled,
                 explainer_methods):
        self.model = model
        self.model_name = model_name
        self.arch = arch
        self.config = config
        self.dtype = dtype
        self.device_str = device
        self.IsAgent = IsAgent
        self.isExplainerEnabled = isExplainerEnabled
        self.explainer_methods = explainer_methods

        # Load tokenizer
        from .models.multimodal.tokenizer import GemmaTokenizer
        self.tokenizer = GemmaTokenizer(tokenizer_path)

        # Image processor (detect image_size from config)
        from .models.multimodal.image_processor import ImageProcessor
        vis_cfg = config.get("vision_config", {})
        img_size = vis_cfg.get("image_size", 224)
        self.image_processor = ImageProcessor(image_size=img_size)
        self.image_size = img_size
        self.patch_size = vis_cfg.get("patch_size", 14)

        # Explainers (lazy init)
        self._grad_cam = None
        self._lime = None

        logger.info(
            "✓ ThinkLabModel ready | %s | arch=%s | explainer=%s",
            model_name, arch, isExplainerEnabled,
        )

    # ── Inference ───────────────────────────────────────────────────
    def inference(self, image, prompt: str, max_tokens: int = 128,
                  temperature: float = 0.7, top_k: int = 40,
                  top_p: float = 0.95) -> dict:
        """
        Run inference on image + prompt.

        Args:
            image: PIL Image, numpy array, or file path
            prompt: text prompt

        Returns:
            dict with keys: text, generated_ids, vision_features, attentions
        """
        from .inference import InferenceEngine
        engine = InferenceEngine(self)
        return engine.run(image, prompt, max_tokens, temperature, top_k, top_p)

    # ── Explain ─────────────────────────────────────────────────────
    def explain(self, image, prompt: str, max_tokens: int = 64,
                methods: list = None, lime_samples: int = 128) -> dict:
        """
        Run inference + explainability (Grad-CAM + LIME + correlation).

        Returns:
            dict with: text, grad_cam_heatmaps, lime_result, correlation
        """
        if not self.isExplainerEnabled:
            raise RuntimeError("Explainer is disabled. Set isExplainerEnabled=True")

        from .inference import InferenceEngine
        engine = InferenceEngine(self)
        return engine.run_with_explanation(
            image, prompt, max_tokens,
            methods=methods or self.explainer_methods,
            lime_samples=lime_samples,
        )

    # ── Trainer (returns a Trainer instance) ────────────────────────
    def trainer(self, **kwargs):
        """Get a Trainer instance for fine-tuning."""
        from .trainer import Trainer
        return Trainer(self, **kwargs)

    # ── Info ────────────────────────────────────────────────────────
    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        dev = next(self.model.parameters()).device
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "params": f"{n_params / 1e9:.2f}B",
            "device": str(dev),
            "dtype": str(self.dtype),
            "image_size": self.image_size,
            "explainer": self.isExplainerEnabled,
            "agent": self.IsAgent,
        }

    def __repr__(self):
        i = self.info()
        return f"ThinkLabModel({i['model_name']}, {i['params']}, {i['device']})"

        # ── Agent mode ──────────────────────────────────────────────────
    def agent(
        self,
        sandbox_url: str = "http://localhost:8000",
        sandbox_api_key: str = "",
        mcp_endpoints: list = None,
        max_steps: int = 20,
    ) -> "ThinkLabAgent":
        """Get an agent instance connected to sandbox + MCP."""
        if not self.IsAgent:
            raise RuntimeError("Agent mode disabled. Set IsAgent=True in load_llm()")
        from .agent import ThinkLabAgent
        from .agent.mcp_client import connect_mcp_tools

        ag = ThinkLabAgent(
            self,
            sandbox_url=sandbox_url,
            sandbox_api_key=sandbox_api_key,
            max_steps=max_steps,
        )
        if mcp_endpoints:
            connect_mcp_tools(ag, mcp_endpoints)
        return ag

