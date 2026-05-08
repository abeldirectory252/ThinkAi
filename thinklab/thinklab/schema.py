"""Production-grade data structures for ThinkLab inference."""
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Explainability config (passed at .inference() time) ─────────────
@dataclass
class ExplainConfig:
    """
    Explainability settings — passed to .inference(), NOT load_llm().

    Usage:
        explainability={"enabled": True, "mode": "grad_cam"}
        explainability={"enabled": True, "mode": "lime", "lime_samples": 200}
        explainability={"enabled": True, "mode": "both", "per_class": True}
    """
    enabled: bool = False
    mode: str = "both"             # "grad_cam", "lime", or "both"
    lime_samples: int = 128
    per_class: bool = True         # class-specific heatmaps
    overlay_alpha: float = 0.5     # heatmap overlay transparency
    colormap: str = "jet"          # matplotlib colormap name

    @classmethod
    def from_dict(cls, d: dict) -> "ExplainConfig":
        if d is None:
            return cls(enabled=False)
        return cls(
            enabled=d.get("enabled", d.get("isExplainerEnabled", False)),
            mode=d.get("mode", "both"),
            lime_samples=d.get("lime_samples", 128),
            per_class=d.get("per_class", True),
            overlay_alpha=d.get("overlay_alpha", 0.5),
            colormap=d.get("colormap", "jet"),
        )


# ── Explainability result (actual images + data) ───────────────────
@dataclass
class ExplainResult:
    """
    Contains actual heatmap images as numpy arrays.

    Attributes:
        grad_cam: per-token/per-class Grad-CAM heatmaps
        lime: LIME explanation with region importance
        correlation: cross-modal overlap score
    """
    # Grad-CAM outputs
    grad_cam_heatmaps: Optional[List[np.ndarray]] = None      # raw H×W heatmaps
    grad_cam_overlays: Optional[List[np.ndarray]] = None       # overlaid on original image
    grad_cam_labels: Optional[List[str]] = None                # token/class label per map

    # LIME outputs
    lime_mask: Optional[np.ndarray] = None                     # full importance mask
    lime_positive_overlay: Optional[np.ndarray] = None         # regions supporting prediction
    lime_negative_overlay: Optional[np.ndarray] = None         # regions against prediction
    lime_segments: Optional[np.ndarray] = None                 # superpixel segment IDs
    lime_weights: Optional[Dict[int, float]] = None            # segment_id → weight

    # Correlation
    mean_overlap: float = 0.0
    per_token_correlation: Optional[List[dict]] = None

    # Metadata
    mode: str = "both"
    total_heatmaps: int = 0

    def to_dict(self) -> dict:
        """Serialize (images become shape descriptions, not raw data)."""
        d = {}
        d["mode"] = self.mode
        d["mean_overlap"] = self.mean_overlap
        d["total_heatmaps"] = self.total_heatmaps
        if self.grad_cam_heatmaps:
            d["grad_cam"] = {
                "count": len(self.grad_cam_heatmaps),
                "labels": self.grad_cam_labels or [],
                "heatmap_shape": list(self.grad_cam_heatmaps[0].shape),
            }
        if self.lime_mask is not None:
            d["lime"] = {
                "mask_shape": list(self.lime_mask.shape),
                "n_segments": int(self.lime_segments.max()) + 1 if self.lime_segments is not None else 0,
                "top_positive_segments": sorted(
                    [(k, v) for k, v in (self.lime_weights or {}).items() if v > 0],
                    key=lambda x: -x[1],
                )[:5],
            }
        if self.per_token_correlation:
            d["correlation"] = {
                "mean_overlap": self.mean_overlap,
                "per_token": self.per_token_correlation[:5],
            }
        return d


# ── Image config ───────────────────────────────────────────────────
@dataclass
class ImageConfig:
    normalize: bool = True
    resize_to: Tuple[int, int] = (512, 512)
    convert_to_grayscale: bool = False
    contrast_enhance: bool = False
    denoise: bool = False
    window_center: Optional[int] = None
    window_width: Optional[int] = None
    min_dpi: int = 72
    auto_orient: bool = True


# ── Clinical context ──────────────────────────────────────────────
@dataclass
class ClinicalContext:
    """Optional clinical context — only used by medical models."""
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    bmi: Optional[float] = None
    symptoms: Optional[List[dict]] = None
    vital_signs: Optional[dict] = None
    lab_results: Optional[dict] = None
    medical_history: Optional[dict] = None
    risk_factors: Optional[List[str]] = None
    reason_for_exam: Optional[str] = None
    clinical_question: Optional[str] = None
    comparison_studies: Optional[List[dict]] = None


# ── Inference config (model-level defaults from load_llm) ─────────
@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 1024
    repetition_penalty: float = 1.1
    safety_filter: str = "strict"
    clinical_guidelines: Optional[str] = None
    differential_enabled: bool = False
    precision: str = "fp16"
    compile_model: bool = False
    use_flash_attention: bool = False
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_cache: bool = False
    cache_ttl_hours: int = 24
    logging_level: str = "INFO"
    metrics_enabled: bool = True
    trace_id: Optional[str] = None
    radiology_specialization: bool = False
    anatomy_prior: str = "general"
    confidence_threshold: float = 0.5


# ── Inference payload ─────────────────────────────────────────────
@dataclass
class InferencePayload:
    prompt: str = ""
    image_path: Optional[str] = None
    image_config: Optional[ImageConfig] = None
    clinical_context: Optional[ClinicalContext] = None
    inference_config: Optional[dict] = None
    output_format: Optional[dict] = None
    metadata: Optional[dict] = None
    callbacks: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> "InferencePayload":
        img_cfg = None
        if "image_config" in d:
            prep = d["image_config"].get("preprocessing", {})
            qual = d["image_config"].get("image_quality", {})
            wl = prep.get("window_level", {})
            img_cfg = ImageConfig(
                normalize=prep.get("normalize", True),
                resize_to=tuple(prep.get("resize_to", (512, 512))),
                convert_to_grayscale=prep.get("convert_to_grayscale", False),
                contrast_enhance=prep.get("contrast_enhance", False),
                denoise=prep.get("denoise", False),
                window_center=wl.get("center"),
                window_width=wl.get("width"),
                min_dpi=qual.get("min_dpi", 72),
                auto_orient=qual.get("auto_orient", True),
            )
        cc = None
        if "clinical_context" in d:
            c = d["clinical_context"]
            demo = c.get("patient_demographics", {})
            cc = ClinicalContext(
                patient_age=demo.get("age_years"),
                patient_sex=demo.get("sex"),
                bmi=demo.get("bmi"),
                symptoms=c.get("symptoms"),
                vital_signs=c.get("vital_signs"),
                lab_results=c.get("lab_results"),
                medical_history=c.get("medical_history"),
                risk_factors=c.get("risk_factors"),
                reason_for_exam=c.get("reason_for_exam"),
                clinical_question=c.get("clinical_question"),
                comparison_studies=c.get("comparison_studies"),
            )
        return cls(
            prompt=d.get("prompt", ""),
            image_path=d.get("image_path"),
            image_config=img_cfg,
            clinical_context=cc,
            inference_config=d.get("inference_config"),
            output_format=d.get("output_format"),
            metadata=d.get("metadata"),
            callbacks=d.get("callbacks"),
        )


# ── Inference result ──────────────────────────────────────────────
@dataclass
class InferenceResult:
    """Structured production output."""
    request_id: str = ""
    model: str = ""
    version: str = ""
    inference_mode: str = "standard"
    timestamp: str = ""
    processing_latency_ms: float = 0.0
    token_speed: str = ""
    tokens_generated: int = 0
    hipaa_compliant: bool = True
    audit_log_ready: bool = True

    # Core output
    model_output: str = ""
    confidence_score: float = 0.0
    critical_finding_detected: bool = False

    # Structured sections
    findings: Optional[List[dict]] = None
    impression: Optional[str] = None
    recommendations: Optional[List[str]] = None
    differential_diagnosis: Optional[List[dict]] = None

    # Explainability — contains actual images
    explain: Optional[ExplainResult] = None

    # Clinical
    clinical_context_used: bool = False
    anatomy_regions_analyzed: Optional[List[str]] = None

    # Metadata
    disclaimer: str = (
        "AI-generated analysis. Must be reviewed by a qualified "
        "healthcare professional before clinical use."
    )
    review_required: bool = True
    trace_id: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        # Segmentation-specific fields (set dynamically by inference engine)
        self.raw_outputs: Optional[dict] = None       # pred_masks, pred_boxes, pred_logits, etc.
        self.inputs: Optional[dict] = None             # pixel_values, original_sizes
        self._image_processor = None                    # Sam3ImageProcessor reference

    def post_process_instance_segmentation(
        self,
        outputs=None,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: Optional[List[tuple]] = None,
    ) -> list:
        """Post-process segmentation outputs into instance masks, boxes, and scores.

        Args:
            outputs: Raw model outputs dict. If None, uses self.raw_outputs.
            threshold: Score threshold to keep instances.
            mask_threshold: Threshold for binarizing predicted masks.
            target_sizes: List of (height, width) tuples for resizing.

        Returns:
            list[dict]: Each dict has 'scores', 'boxes', 'masks' tensors.
        """
        if self._image_processor is None:
            raise RuntimeError(
                "post_process_instance_segmentation is only available for "
                "segmentation models (e.g. facebook/sam3). "
                "This result was produced by a text-generation model."
            )
        if outputs is None:
            outputs = self.raw_outputs
        if outputs is None:
            raise ValueError("No outputs to post-process.")
        return self._image_processor.post_process_instance_segmentation(
            outputs, threshold, mask_threshold, target_sizes
        )

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if v is not None and k != "explain":
                d[k] = v
        if self.explain:
            d["explain"] = self.explain.to_dict()
        return d

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    @staticmethod
    def generate_request_id(model_name: str) -> str:
        slug = model_name.split("/")[-1].replace("-", "")
        ts = time.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        return f"{slug}_req_{ts}_{uid}"
