"""Correlate model output text with Grad-CAM and LIME visual explanations."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional


class TextVisionCorrelator:
    """
    Correlates generated text tokens with visual explanations:
      - Per-token Grad-CAM heatmaps
      - LIME superpixel importances
      - Combined correlation score
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def correlate(
        self,
        generated_ids: List[int],
        grad_cam_heatmaps: Dict[int, np.ndarray],
        lime_result: dict,
        image_np: np.ndarray,
    ) -> dict:
        """
        Build correlation between each token and image regions.

        Returns dict with:
            tokens: list of decoded token strings
            per_token: list of dicts with grad_cam, lime overlap, score
            aggregate_grad_cam: averaged heatmap over all tokens
            lime_heatmap: LIME pixel-level heatmap
            correlation_matrix: (n_tokens, n_superpixels) correlation
        """
        tokens = []
        for tid in generated_ids:
            tokens.append(self.tokenizer.decode([tid]))

        segments = lime_result["segments"]
        importances = lime_result["importances"]
        n_sp = lime_result["n_features"]

        # LIME heatmap
        lime_hm = np.zeros(segments.shape, dtype=np.float32)
        for sid in range(n_sp):
            lime_hm[segments == sid] = importances[sid]
        lime_hm_norm = self._normalize(lime_hm)

        # Per-token analysis
        per_token = []
        correlation_matrix = np.zeros((len(generated_ids), n_sp))

        for i, tok in enumerate(tokens):
            gc_hm = grad_cam_heatmaps.get(i)
            if gc_hm is None:
                gc_hm = np.zeros_like(lime_hm_norm)

            # Resize grad-cam to match segments shape if needed
            if gc_hm.shape != segments.shape:
                from PIL import Image
                gc_img = Image.fromarray((gc_hm * 255).astype(np.uint8))
                gc_img = gc_img.resize(
                    (segments.shape[1], segments.shape[0]),
                    Image.BILINEAR,
                )
                gc_hm = np.array(gc_img).astype(np.float32) / 255.0

            # Per-superpixel correlation: mean grad-cam in each superpixel
            for sid in range(n_sp):
                mask = segments == sid
                if mask.sum() > 0:
                    correlation_matrix[i, sid] = gc_hm[mask].mean()

            # Overlap score between grad-cam and LIME
            overlap = np.corrcoef(gc_hm.flatten(), lime_hm_norm.flatten())[0, 1]
            if np.isnan(overlap):
                overlap = 0.0

            per_token.append({
                "token": tok,
                "token_id": generated_ids[i],
                "grad_cam": gc_hm,
                "overlap_with_lime": float(overlap),
                "top_superpixels": np.argsort(
                    correlation_matrix[i]
                )[-5:][::-1].tolist(),
            })

        # Aggregate grad-cam (mean over all tokens)
        all_gc = [pt["grad_cam"] for pt in per_token]
        agg_gc = np.mean(all_gc, axis=0) if all_gc else np.zeros_like(lime_hm)
        agg_gc = self._normalize(agg_gc)

        return {
            "tokens": tokens,
            "per_token": per_token,
            "aggregate_grad_cam": agg_gc,
            "lime_heatmap": lime_hm_norm,
            "correlation_matrix": correlation_matrix,
            "mean_overlap": float(np.mean(
                [pt["overlap_with_lime"] for pt in per_token]
            )),
        }

    def visualize(
        self,
        image_np: np.ndarray,
        correlation: dict,
        save_path: Optional[str] = None,
        top_k_tokens: int = 6,
    ):
        """Visualize correlation between text and vision explanations."""
        tokens = correlation["tokens"]
        per_token = correlation["per_token"]
        agg_gc = correlation["aggregate_grad_cam"]
        lime_hm = correlation["lime_heatmap"]

        n_show = min(top_k_tokens, len(tokens))
        fig = plt.figure(figsize=(4 * (n_show + 3), 4))
        gs = gridspec.GridSpec(1, n_show + 3, figure=fig)

        # Original image
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(image_np)
        ax0.set_title("Original", fontsize=10)
        ax0.axis("off")

        # Aggregate Grad-CAM
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(image_np)
        ax1.imshow(agg_gc, cmap="jet", alpha=0.5)
        ax1.set_title(f"Agg Grad-CAM\noverlap={correlation['mean_overlap']:.2f}",
                      fontsize=9)
        ax1.axis("off")

        # LIME
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(image_np)
        ax2.imshow(lime_hm, cmap="hot", alpha=0.5)
        ax2.set_title("LIME", fontsize=10)
        ax2.axis("off")

        # Per-token Grad-CAMs
        for j in range(n_show):
            ax = fig.add_subplot(gs[0, 3 + j])
            ax.imshow(image_np)
            ax.imshow(per_token[j]["grad_cam"], cmap="jet", alpha=0.5)
            tok_str = per_token[j]["token"][:12]
            overlap = per_token[j]["overlap_with_lime"]
            ax.set_title(f'"{tok_str}"\n↔LIME:{overlap:.2f}', fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization → {save_path}")
        plt.show()

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.zeros_like(arr)
