"""
Auto-register all supported model builders.

Discovery system:
  1. Scans models/multimodal/ and models/llm/ for model packages
  2. Each model package has a builder.py with REGISTRY_PATTERN, ARCH, and a build function
  3. Builders are loaded via importlib.util (supports hyphenated directory names)
  4. Each builder auto-registers with the ThinkLab registry

Directory structure mirrors HuggingFace repo paths:
  models/multimodal/google/medgemma-4b-it/builder.py
  models/multimodal/google/paligemma-3b-mix-224/builder.py
"""
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("thinklab.builders")


def _discover_and_register_builders():
    """Scan model directories for builder.py files and register them."""
    from .registry import register_model

    models_root = Path(__file__).parent / "models"

    for modality_dir in models_root.iterdir():
        if not modality_dir.is_dir() or modality_dir.name.startswith("_"):
            continue
        for builder_path in modality_dir.rglob("builder.py"):
            _load_builder(builder_path, register_model)


def _load_builder(builder_path: Path, register_fn):
    """Load a builder.py module from a file path (supports hyphenated dirs)."""
    parts = builder_path.relative_to(Path(__file__).parent).with_suffix("").parts
    safe_parts = [p.replace("-", "_") for p in parts]
    module_name = "thinklab." + ".".join(safe_parts)

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(builder_path))
        if spec is None or spec.loader is None:
            logger.warning("Cannot load builder: %s", builder_path)
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        pattern = getattr(module, "REGISTRY_PATTERN", None)
        arch = getattr(module, "ARCH", None)
        model_id = getattr(module, "MODEL_ID", None)

        if pattern is None:
            logger.debug("Skipping %s (no REGISTRY_PATTERN)", builder_path)
            return

        # Find the build function
        build_fn = None
        for name in dir(module):
            if name.startswith("build_") and callable(getattr(module, name)):
                build_fn = getattr(module, name)
                break

        if build_fn is None:
            logger.warning("No build function found in %s", builder_path)
            return

        defaults = {
            "_model_id": model_id,
            "_builder_path": str(builder_path.parent),
        }

        register_fn(pattern, build_fn, arch=arch, defaults=defaults)
        logger.info("✓ Registered: %s → %s (arch=%s)", pattern, model_id, arch)

    except Exception as e:
        logger.error("Failed to load builder %s: %s", builder_path, e)


# ── Run discovery on import ────────────────────────────────────────
_discover_and_register_builders()
