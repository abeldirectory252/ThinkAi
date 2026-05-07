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
import types
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
    """Load a builder.py module from a file path (supports hyphenated dirs).

    Key challenge: directories like 'medgemma-4b-it' have hyphens, so Python
    can't import them normally. We use importlib.util but must also register
    the full parent package chain in sys.modules so relative imports
    (e.g. 'from .model import MedGemma') work inside the builder.
    """
    thinklab_root = Path(__file__).parent

    # Build the module name: thinklab.models.multimodal.google.medgemma_4b_it.builder
    parts = builder_path.relative_to(thinklab_root).with_suffix("").parts
    safe_parts = [p.replace("-", "_") for p in parts]
    module_name = "thinklab." + ".".join(safe_parts)

    # The package that builder.py belongs to (e.g. thinklab.models...medgemma_4b_it)
    package_name = ".".join(module_name.split(".")[:-1])

    try:
        # ── Register parent package chain ──────────────────────────
        # Walk from thinklab → thinklab.models → ... → thinklab...medgemma_4b_it
        # so that relative imports inside builder.py can resolve.
        dir_parts = builder_path.parent.relative_to(thinklab_root).parts
        for i in range(len(dir_parts)):
            pkg_parts = ["thinklab"] + [p.replace("-", "_") for p in dir_parts[:i+1]]
            pkg_name = ".".join(pkg_parts)
            if pkg_name not in sys.modules:
                pkg_dir = thinklab_root / Path(*dir_parts[:i+1])
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = [str(pkg_dir)]
                pkg.__package__ = pkg_name
                pkg.__file__ = str(pkg_dir / "__init__.py")
                sys.modules[pkg_name] = pkg

        # ── Load the builder module ────────────────────────────────
        spec = importlib.util.spec_from_file_location(
            module_name, str(builder_path),
            submodule_search_locations=[str(builder_path.parent)]
        )
        if spec is None or spec.loader is None:
            logger.warning("Cannot load builder: %s", builder_path)
            return

        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
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
