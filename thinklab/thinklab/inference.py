"""
Production inference engine.

Explainability is controlled at inference time, NOT at model load time:
    result = model.inference(
        "xray.jpg", "analyze",
        explainability={"enabled": True, "mode": "grad_cam"}
    )
    result.explain.grad_cam_heatmaps   # list of numpy H×W arrays
    result.explain.grad_cam_overlays   # heatmaps overlaid on original image
"""
import logging
import time
from typing import Optional
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

from .schema import (
    InferenceConfig, InferencePayload, InferenceResult,
    ClinicalContext, ImageConfig, ExplainConfig, ExplainResult,
)

logger = logging.getLogger("thinklab.inference")

CLINICAL_MODELS = {"medgemma", "med-palm", "biomedclip", "radlm"}


class InferenceEngine:
    """Unified inference with optional explainability."""

    def __init__(self, thinklab_model):
        self.tm = thinklab_model
        self.model = thinklab_model.model
        self.tokenizer = thinklab_model.tokenizer
        self.processor = thinklab_model.image_processor
        self.config: InferenceConfig = getattr(
            thinklab_model, "inference_config", InferenceConfig()
        )

    def _is_clinical_model(self) -> bool:
        name = self.tm.model_name.lower()
        return any(c in name for c in CLINICAL_MODELS)

    # ── Image preprocessing ─────────────────────────────────────────
    def _preprocess_image(self, image, img_cfg: Optional[ImageConfig] = None):
        if isinstance(image, str):
            import os
            if not os.path.exists(image):
                raise FileNotFoundError(
                    f"\n{'='*60}\n"
                    f"  ❌ IMAGE PATH ERROR\n"
                    f"{'='*60}\n"
                    f"  The image path you provided does not exist:\n\n"
                    f"    → \"{image}\"\n\n"
                    f"  Please verify:\n"
                    f"    1. The file path is correct and accessible\n"
                    f"    2. The file extension is valid (.jpg, .png, .dcm)\n"
                    f"    3. On Kaggle, ensure the dataset is attached\n"
                    f"       under /kaggle/input/<dataset-name>/\n"
                    f"{'='*60}"
                )
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif image is None:
            raise ValueError(
                f"\n{'='*60}\n"
                f"  ❌ NO IMAGE PROVIDED\n"
                f"{'='*60}\n"
                f"  You must provide an image via image_path= or image=.\n"
                f"  Supported: file path (str), numpy array, or PIL Image.\n"
                f"{'='*60}"
            )

        if img_cfg:
            if img_cfg.convert_to_grayscale:
                image = image.convert("L").convert("RGB")
            if img_cfg.contrast_enhance:
                from PIL import ImageEnhance
                image = ImageEnhance.Contrast(image).enhance(1.5)

        image_np = np.array(image.resize((self.tm.image_size, self.tm.image_size)))
        pv = self.processor(image, dtype=self.tm.dtype)
        dev = next(self.model.parameters()).device
        return pv.to(dev), image_np

    # ── Clinical prompt builder ─────────────────────────────────────
    def _build_clinical_prompt(self, prompt, cc, inf_cfg=None):
        if not cc or not self._is_clinical_model():
            return prompt

        parts = []
        demo = []
        if cc.patient_age: demo.append(f"Age: {cc.patient_age}")
        if cc.patient_sex: demo.append(f"Sex: {cc.patient_sex}")
        if cc.bmi: demo.append(f"BMI: {cc.bmi}")
        if demo:
            parts.append("Patient: " + ", ".join(demo))

        if cc.symptoms:
            syms = ", ".join(
                f"{s['symptom']} ({s.get('severity','')}, {s.get('duration_days','')}d)"
                for s in cc.symptoms
            )
            parts.append(f"Symptoms: {syms}")

        if cc.vital_signs:
            vs = cc.vital_signs
            parts.append(
                f"Vitals: T={vs.get('temperature_celsius')}°C, "
                f"HR={vs.get('heart_rate_bpm')}, RR={vs.get('respiratory_rate')}, "
                f"SpO2={vs.get('oxygen_saturation')}%, BP={vs.get('blood_pressure')}"
            )

        if cc.lab_results:
            labs = ", ".join(f"{k}={v}" for k, v in cc.lab_results.items())
            parts.append(f"Labs: {labs}")

        if cc.medical_history:
            mh = cc.medical_history
            if mh.get("prior_conditions"):
                parts.append(f"Hx: {', '.join(mh['prior_conditions'])}")
            if mh.get("medications"):
                parts.append(f"Meds: {', '.join(mh['medications'])}")
            if mh.get("allergies"):
                parts.append(f"Allergies: {', '.join(mh['allergies'])}")

        if cc.reason_for_exam:
            parts.append(f"Reason: {cc.reason_for_exam}")
        if cc.clinical_question:
            parts.append(f"Question: {cc.clinical_question}")

        if inf_cfg:
            if inf_cfg.get("pathologies_to_check"):
                parts.append(f"Check for: {', '.join(inf_cfg['pathologies_to_check'])}")
            if inf_cfg.get("generate_differential"):
                parts.append(f"Provide {inf_cfg.get('num_differentials', 3)} differentials.")

        context = "\n".join(parts)
        return f"Clinical Context:\n{context}\n\nTask: {prompt}"

    # ── Heatmap overlay helper ──────────────────────────────────────
    @staticmethod
    def _overlay_heatmap(image_np, heatmap, alpha=0.5, colormap="jet"):
        """Overlay a H×W heatmap on an RGB image → RGB numpy array."""
        import matplotlib.cm as cm

        h, w = image_np.shape[:2]
        # Resize heatmap to image size
        from PIL import Image as PILImage
        hm_resized = np.array(
            PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
        ).astype(np.float32) / 255.0

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = (cmap(hm_resized)[:, :, :3] * 255).astype(np.uint8)

        # Blend
        blended = (alpha * colored + (1 - alpha) * image_np).astype(np.uint8)
        return blended

    # ── Grad-CAM execution ──────────────────────────────────────────
    def _run_grad_cam(self, pv, input_ids, generated_ids, image_np, ecfg):
        """Run Grad-CAM, return per-token heatmaps + overlays."""
        from .models.ModelExplain.grad_cam import GradCAM

        gc = GradCAM(self.model, target_layer_idx=-1)
        gc_maps = gc.compute_per_token(
            pv, input_ids, generated_ids,
            image_size=self.tm.image_size,
            patch_size=self.tm.patch_size,
        )
        gc.remove_hooks()

        heatmaps = []
        overlays = []
        labels = []

        for entry in gc_maps:
            hm = entry["heatmap"]             # H×W numpy float
            tok = entry.get("token", "")
            heatmaps.append(hm)
            overlays.append(
                self._overlay_heatmap(image_np, hm, ecfg.overlay_alpha, ecfg.colormap)
            )
            labels.append(tok)

        return heatmaps, overlays, labels

    # ── LIME execution ──────────────────────────────────────────────
    def _run_lime(self, image_np, prompt, max_tokens, ecfg):
        """Run LIME, return masks + overlays."""
        from .models.ModelExplain.lime_explainer import LIMEExplainer

        lime = LIMEExplainer(
            self.model, self.tokenizer, self.processor,
            num_samples=ecfg.lime_samples,
        )
        lime_res = lime.explain(image_np, prompt, max_tokens)

        segments = lime_res.get("segments")         # H×W int
        weights = lime_res.get("feature_weights", {})
        mask = lime_res.get("importance_mask")       # H×W float

        if mask is None and segments is not None and weights:
            mask = np.zeros_like(segments, dtype=np.float32)
            for seg_id, w in weights.items():
                mask[segments == seg_id] = w

        # Positive / negative overlays
        pos_overlay = None
        neg_overlay = None
        if mask is not None:
            pos_mask = np.clip(mask, 0, None)
            neg_mask = np.clip(-mask, 0, None)
            if pos_mask.max() > 0:
                pos_overlay = self._overlay_heatmap(
                    image_np, pos_mask / pos_mask.max(), ecfg.overlay_alpha, "Greens"
                )
            if neg_mask.max() > 0:
                neg_overlay = self._overlay_heatmap(
                    image_np, neg_mask / neg_mask.max(), ecfg.overlay_alpha, "Reds"
                )

        return mask, segments, weights, pos_overlay, neg_overlay

    # ═════════════════════════════════════════════════════════════════
    #  MAIN RUN — single entry point for inference + optional explain
    # ═════════════════════════════════════════════════════════════════
    def run(self, image, prompt, max_tokens=128,
            temperature=0.7, top_k=40, top_p=0.95,
            payload=None, explainability=None, **kwargs):
        """
        Unified inference. Explainability is controlled HERE:

            engine.run(img, prompt, explainability={"enabled": True, "mode": "grad_cam"})

        Returns:
            InferenceResult with .model_output and .explain (contains images)
        """
        t0 = time.perf_counter()

        # Parse payload
        pl = InferencePayload.from_dict(payload) if payload else InferencePayload()
        actual_prompt = pl.prompt or prompt
        actual_image = pl.image_path or image

        # Parse explainability config
        ecfg = ExplainConfig.from_dict(explainability)

        # Preprocess
        pv, image_np = self._preprocess_image(actual_image, pl.image_config)

        # Build prompt with clinical context
        full_prompt = self._build_clinical_prompt(
            actual_prompt, pl.clinical_context, pl.inference_config
        )

        dev = next(self.model.parameters()).device
        ids = self.tokenizer.build_paligemma_input(full_prompt)
        input_ids = torch.tensor([ids], device=dev)

        # ── Generate ────────────────────────────────────────────────
        use_temp = temperature
        need_attn = ecfg is not None and ecfg.enabled
        with torch.no_grad():
            gen_out = self.model.generate(
                pv, input_ids,
                max_new_tokens=max_tokens,
                temperature=use_temp,
                top_k=top_k, top_p=top_p,
                repetition_penalty=getattr(self.config, 'repetition_penalty', 1.1),
                output_attentions=need_attn,
            )

        text = self.tokenizer.decode(gen_out["generated_ids"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_tokens = len(gen_out["generated_ids"])
        tok_per_sec = n_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        # ── Explainability (if enabled) ─────────────────────────────
        explain_result = None
        if ecfg.enabled:
            explain_result = ExplainResult(mode=ecfg.mode)
            run_gc = ecfg.mode in ("grad_cam", "both")
            run_lime = ecfg.mode in ("lime", "both")

            gc_maps_raw = None

            if run_gc:
                heatmaps, overlays, labels = self._run_grad_cam(
                    pv, input_ids, gen_out["generated_ids"], image_np, ecfg
                )
                explain_result.grad_cam_heatmaps = heatmaps
                explain_result.grad_cam_overlays = overlays
                explain_result.grad_cam_labels = labels
                explain_result.total_heatmaps += len(heatmaps)

            if run_lime:
                mask, segs, wts, pos_ov, neg_ov = self._run_lime(
                    image_np, actual_prompt, max_tokens, ecfg
                )
                explain_result.lime_mask = mask
                explain_result.lime_segments = segs
                explain_result.lime_weights = wts
                explain_result.lime_positive_overlay = pos_ov
                explain_result.lime_negative_overlay = neg_ov

            if run_gc and run_lime:
                from .models.ModelExplain.correlator import TextVisionCorrelator
                corr = TextVisionCorrelator(self.tokenizer)
                # Rebuild gc_maps for correlator
                gc_maps_for_corr = [
                    {"heatmap": h, "token": l}
                    for h, l in zip(
                        explain_result.grad_cam_heatmaps,
                        explain_result.grad_cam_labels
                    )
                ]
                lime_for_corr = {
                    "segments": segs,
                    "feature_weights": wts,
                    "importance_mask": mask,
                }
                correlation = corr.correlate(
                    gen_out["generated_ids"], gc_maps_for_corr,
                    lime_for_corr, image_np,
                )
                explain_result.mean_overlap = correlation.get("mean_overlap", 0)
                explain_result.per_token_correlation = correlation.get("per_token", [])

        # ── Build result ────────────────────────────────────────────
        result = InferenceResult(
            request_id=InferenceResult.generate_request_id(self.tm.model_name),
            model=self.tm.model_name,
            version=self.tm.arch,
            inference_mode="explain" if ecfg.enabled else "standard",
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_latency_ms=round(elapsed_ms, 1),
            token_speed=f"{tok_per_sec:.0f}tok/sec",
            tokens_generated=n_tokens,
            model_output=text,
            explain=explain_result,
            clinical_context_used=(
                pl.clinical_context is not None and self._is_clinical_model()
            ),
            trace_id=self.config.trace_id or kwargs.get("trace_id"),
            metadata=pl.metadata,
        )

        # Callbacks
        if pl.callbacks:
            cb = pl.callbacks
            if "on_complete" in cb and callable(cb["on_complete"]):
                try:
                    cb["on_complete"](result)
                except Exception:
                    pass

        return result
