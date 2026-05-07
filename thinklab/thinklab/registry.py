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
    # Output
    isJsonAsOutput: bool = True,
    isWebSocEnable: bool = False,
    # Agent
    IsAgent: bool = False,
    # Inference defaults
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    max_tokens: int = 1024,
    repetition_penalty: float = 1.1,
    # Safety
    safety_filter: str = "strict",
    clinical_guidelines: Optional[str] = None,
    differential_enabled: bool = False,
    # Processing
    precision: str = "fp16",
    compile_model: bool = False,
    use_flash_attention: bool = False,
    # Network
    timeout_seconds: int = 30,
    retry_attempts: int = 3,
    # Cache
    enable_cache: bool = False,
    cache_ttl_hours: int = 24,
    # Logging
    logging_level: str = "INFO",
    metrics_enabled: bool = True,
    trace_id: Optional[str] = None,
    # Medical specialization
    radiology_specialization: bool = False,
    anatomy_prior: str = "general",
    confidence_threshold: float = 0.5,
    # Memory
    max_memory_gb: Optional[float] = None,
    **kwargs,
) -> "ThinkLabModel":
    """
    Load any supported model by repo ID.

    Explainability is NOT configured here — pass it at .inference() time.

    Returns:
        ThinkLabModel wrapper with .inference(), .trainer(), .agent()
    """
    from .weights import HuggingFaceDownloader
    from .schema import InferenceConfig

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pt_dtype = dtype_map.get(dtype, torch.bfloat16)

    if save_dir is None:
        slug = model_name.replace("/", "_")
        save_dir = f"./weights/{slug}"
    save_path = Path(save_dir)

    logger.info("Downloading %s → %s", model_name, save_path)
    dl = HuggingFaceDownloader(model_name, save_path, token=token)
    dl.download_model()

    import json
    config = {}
    cfg_file = save_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            config = json.load(f)

    entry = _match_registry(model_name)
    if entry is None:
        raise ValueError(
            f"No registered builder for '{model_name}'.\n"
            f"Registered: {[e['pattern'] for e in _MODEL_REGISTRY.values()]}\n"
            f"Use thinklab.register_model() to add support."
        )

    logger.info("Matched arch: %s", entry["arch"])

    builder_kwargs = {**entry["defaults"], **kwargs}
    builder_kwargs["debug"] = (logging_level == "DEBUG")
    # Strip internal metadata keys (prefixed with _) before calling builder
    builder_kwargs = {k: v for k, v in builder_kwargs.items() if not k.startswith("_")}
    raw_model = entry["builder"](
        save_dir=save_path,
        config=config,
        dtype=pt_dtype,
        device=device,
        max_memory_gb=max_memory_gb,
        **builder_kwargs,
    )

    tok_path = tokenizer if tokenizer else str(save_path / "tokenizer.model")

    inf_cfg = InferenceConfig(
        temperature=temperature, top_p=top_p, top_k=top_k,
        max_tokens=max_tokens, repetition_penalty=repetition_penalty,
        safety_filter=safety_filter, clinical_guidelines=clinical_guidelines,
        differential_enabled=differential_enabled, precision=precision,
        compile_model=compile_model, use_flash_attention=use_flash_attention,
        timeout_seconds=timeout_seconds, retry_attempts=retry_attempts,
        enable_cache=enable_cache, cache_ttl_hours=cache_ttl_hours,
        logging_level=logging_level, metrics_enabled=metrics_enabled,
        trace_id=trace_id, radiology_specialization=radiology_specialization,
        anatomy_prior=anatomy_prior, confidence_threshold=confidence_threshold,
    )

    return ThinkLabModel(
        model=raw_model,
        model_name=model_name,
        arch=entry["arch"],
        tokenizer_path=tok_path,
        config=config,
        dtype=pt_dtype,
        device=device,
        IsAgent=IsAgent,
        isJsonAsOutput=isJsonAsOutput,
        isWebSocEnable=isWebSocEnable,
        inference_config=inf_cfg,
    )


# ── Unified wrapper ─────────────────────────────────────────────────
class ThinkLabModel:
    """
    Unified model wrapper.

    Explainability is a RUNTIME option on .inference(), not a model property:
        result = model.inference("img.jpg", "prompt",
            explainability={"enabled": True, "mode": "grad_cam"}
        )
        result.explain.grad_cam_overlays  # actual images
    """

    def __init__(self, model, model_name, arch, tokenizer_path,
                 config, dtype, device, IsAgent,
                 isJsonAsOutput, isWebSocEnable, inference_config):
        self.model = model
        self.model_name = model_name
        self.arch = arch
        self.config = config
        self.dtype = dtype
        self.device_str = device
        self.IsAgent = IsAgent
        self.isJsonAsOutput = isJsonAsOutput
        self.isWebSocEnable = isWebSocEnable
        self.inference_config = inference_config

        # Vision config
        vis_cfg = config.get("vision_config", {})
        self.image_size = vis_cfg.get("image_size", 224)
        self.patch_size = vis_cfg.get("patch_size", 14)

        # Load model-specific tokenizer and image processor
        # Each model package (e.g. google/medgemma-4b-it/) has its own
        self.tokenizer = self._load_tokenizer(model_name, arch, tokenizer_path)
        self.image_processor = self._load_image_processor(model_name, arch, self.image_size)

        logger.info(
            "✓ ThinkLabModel ready | %s | arch=%s | json=%s",
            model_name, arch, isJsonAsOutput,
        )

    # ── Model-specific component loaders ────────────────────────────
    @staticmethod
    def _load_tokenizer(model_name: str, arch: str, tokenizer_path: str):
        """Load tokenizer from the model's own package, fallback to legacy."""
        import importlib.util, sys, types
        from pathlib import Path

        parts = model_name.split("/")
        if len(parts) == 2:
            models_root = Path(__file__).parent / "models" / "multimodal"
            tok_file = models_root / parts[0] / parts[1] / "tokenizer.py"

            if tok_file.exists():
                safe_name = f"thinklab.models.multimodal.{parts[0]}.{parts[1].replace('-', '_')}.tokenizer"
                # Ensure parent package is registered (builder may have done this already)
                pkg_name = ".".join(safe_name.split(".")[:-1])
                if pkg_name not in sys.modules:
                    pkg = types.ModuleType(pkg_name)
                    pkg.__path__ = [str(tok_file.parent)]
                    pkg.__package__ = pkg_name
                    sys.modules[pkg_name] = pkg
                try:
                    spec = importlib.util.spec_from_file_location(safe_name, str(tok_file))
                    mod = importlib.util.module_from_spec(spec)
                    mod.__package__ = pkg_name
                    sys.modules[safe_name] = mod
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and "Tokenizer" in attr_name:
                            logger.info("Using model-specific tokenizer: %s", attr_name)
                            return attr(tokenizer_path)
                except Exception as e:
                    logger.warning("Failed to load model tokenizer: %s", e)

        # Fallback to legacy shared tokenizer
        logger.info("Using legacy shared tokenizer")
        from .models.multimodal.tokenizer import GemmaTokenizer
        return GemmaTokenizer(tokenizer_path)

    @staticmethod
    def _load_image_processor(model_name: str, arch: str, image_size: int):
        """Load image processor from the model's own package, fallback to legacy."""
        import importlib.util, sys, types
        from pathlib import Path

        parts = model_name.split("/")
        if len(parts) == 2:
            models_root = Path(__file__).parent / "models" / "multimodal"
            proc_file = models_root / parts[0] / parts[1] / "image_processor.py"

            if proc_file.exists():
                safe_name = f"thinklab.models.multimodal.{parts[0]}.{parts[1].replace('-', '_')}.image_processor"
                pkg_name = ".".join(safe_name.split(".")[:-1])
                if pkg_name not in sys.modules:
                    pkg = types.ModuleType(pkg_name)
                    pkg.__path__ = [str(proc_file.parent)]
                    pkg.__package__ = pkg_name
                    sys.modules[pkg_name] = pkg
                try:
                    spec = importlib.util.spec_from_file_location(safe_name, str(proc_file))
                    mod = importlib.util.module_from_spec(spec)
                    mod.__package__ = pkg_name
                    sys.modules[safe_name] = mod
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and "Processor" in attr_name:
                            logger.info("Using model-specific image processor: %s", attr_name)
                            return attr(image_size=image_size)
                except Exception as e:
                    logger.warning("Failed to load model processor: %s", e)

        # Fallback to legacy shared processor
        logger.info("Using legacy shared image processor")
        from .models.multimodal.image_processor import ImageProcessor
        return ImageProcessor(image_size=image_size)

    # ── Inference (with optional explainability) ────────────────────
    def inference(self, image=None, prompt: str = "",
                  max_tokens: int = 200, temperature: float = 0.7,
                  top_k: int = 50, top_p: float = 0.9,
                  do_sample: bool = True,
                  repetition_penalty: float = 1.1,
                  payload: dict = None, image_path: str = None,
                  explainability: dict = None,
                  messages: list = None,
                  system_prompt: str = None,
                  **kwargs):
        """
        Run inference. Supports both simple and HF-style messages API.

        Simple API:
            model.inference(prompt="Describe this X-ray", image_path="xray.jpg")

        Messages API (HuggingFace-style):
            model.inference(messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this X-ray"},
                    {"type": "image", "image": "xray.jpg"}
                ]}
            ])

        Args:
            image: PIL Image, numpy array, or file path
            prompt: text prompt
            messages: HF-style messages list (overrides prompt/image if provided)
            system_prompt: system instruction (overridden by messages if provided)
            payload: production payload (clinical context, etc.)
            image_path: alias for image
            explainability: {
                "enabled": True,
                "mode": "grad_cam" | "lime" | "both",
            }

        Returns:
            InferenceResult with:
                .model_output    → generated text
                .explain         → ExplainResult (heatmap images, overlays)
                .to_json()       → full structured JSON
        """
        # ── Parse messages if provided ───────────────────────────────
        if messages:
            # Validate: only models with chat templates support messages API
            has_template = getattr(self.tokenizer, 'HAS_CHAT_TEMPLATE', False)
            if not has_template:
                raise ValueError(
                    f"Cannot use messages= API because '{self.model_name}' does not have a chat template.\n"
                    f"Use the simple API instead:\n"
                    f"  model.inference(prompt='...', image_path='...')"
                )
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", [])

                if role == "system":
                    # Extract system prompt
                    if isinstance(content, str):
                        system_prompt = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                system_prompt = item["text"]

                elif role == "user":
                    if isinstance(content, str):
                        prompt = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    prompt = item["text"]
                                elif item.get("type") == "image":
                                    img_val = item.get("image")
                                    if img_val:
                                        image = img_val

        from .inference import InferenceEngine
        engine = InferenceEngine(self)
        actual_image = image_path or image
        return engine.run(
            actual_image, prompt, max_tokens,
            temperature, top_k, top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            payload=payload,
            explainability=explainability,
            system_prompt=system_prompt,
            **kwargs,
        )

    # ── Trainer (returns a Trainer instance) ────────────────────────
    def trainer(self, **kwargs):
        """Get a Trainer instance for fine-tuning."""
        from .trainer import Trainer
        return Trainer(self, **kwargs)

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
            "agent": self.IsAgent,
        }

    def __repr__(self):
        i = self.info()
        return f"ThinkLabModel({i['model_name']}, {i['params']}, {i['device']})"
