"""ThinkLab — Pure PyTorch AI Research Framework."""
import warnings
from datetime import datetime, timezone

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)
__release_date__ = "2026-05-04"
__deprecation_date__ = "2028-05-04"  # 2 years from release


def _check_deprecation():
    """Warn if this version is past its support window."""
    try:
        now = datetime.now(timezone.utc)
        dep = datetime.strptime(__deprecation_date__, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if now >= dep:
            warnings.warn(
                f"ThinkLab v{__version__} (released {__release_date__}) is no longer "
                f"supported as of {__deprecation_date__}. Please upgrade to the latest "
                f"version: pip install --upgrade thinklab",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            days_left = (dep - now).days
            if days_left <= 90:
                warnings.warn(
                    f"ThinkLab v{__version__} will be deprecated in {days_left} days "
                    f"({__deprecation_date__}). Consider upgrading.",
                    FutureWarning,
                    stacklevel=2,
                )
    except Exception:
        pass


_check_deprecation()

# ── Public API ──────────────────────────────────────────────────────
from .registry import load_llm, list_models, register_model
from .schema import InferenceResult, ExplainResult, ExplainConfig

# Auto-register built-in models
import thinklab.model_builders  # noqa: F401

__all__ = [
    "load_llm",
    "list_models",
    "register_model",
    "InferenceResult",
    "ExplainResult",
    "ExplainConfig",
    "__version__",
]
