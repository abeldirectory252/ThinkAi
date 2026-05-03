"""LIME explainer for PaliGemma multimodal model."""
import numpy as np
import torch
from typing import Callable, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from skimage.segmentation import quickshift


class LIMEExplainer:
    """
    Local Interpretable Model-Agnostic Explanations for images.
    Perturbs superpixels and fits a local linear model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        num_samples: int = 256,
        num_superpixels: int = 50,
        kernel_width: float = 0.25,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_samples = num_samples
        self.num_superpixels = num_superpixels
        self.kernel_width = kernel_width

    def _segment_image(self, image_np: np.ndarray) -> np.ndarray:
        """Segment image into superpixels using quickshift."""
        segments = quickshift(
            image_np, kernel_size=4, max_dist=200, ratio=0.2
        )
        # Remap to contiguous IDs
        unique = np.unique(segments)
        remap = {old: new for new, old in enumerate(unique)}
        return np.vectorize(remap.get)(segments)

    def _perturb(
        self, image_np: np.ndarray, segments: np.ndarray, n_features: int
    ) -> tuple:
        """Generate perturbed images by masking random superpixels."""
        perturbations = np.random.binomial(1, 0.5,
                                           size=(self.num_samples, n_features))
        # First sample = original (all on)
        perturbations[0] = 1

        perturbed_images = []
        for mask_vec in perturbations:
            img = image_np.copy()
            for seg_id in range(n_features):
                if mask_vec[seg_id] == 0:
                    img[segments == seg_id] = 128  # gray out
            perturbed_images.append(img)

        return perturbations, perturbed_images

    @torch.no_grad()
    def _get_model_output(
        self,
        image_np: np.ndarray,
        prompt: str,
        max_tokens: int = 32,
    ) -> str:
        """Run model inference on a numpy image."""
        pv = self.image_processor(image_np, dtype=self.model.model_dtype)
        dev = next(self.model.parameters()).device
        pv = pv.to(dev)

        ids = self.tokenizer.build_paligemma_input(prompt)
        input_ids = torch.tensor([ids], device=dev)

        out = self.model.generate(
            pv, input_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,  # greedy for consistency
        )
        return self.tokenizer.decode(out["generated_ids"])

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity between two texts."""
        w1 = set(text1.lower().split())
        w2 = set(text2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / max(len(w1 | w2), 1)

    def explain(
        self,
        image_np: np.ndarray,
        prompt: str,
        max_tokens: int = 32,
    ) -> dict:
        """
        Run LIME explanation.

        Args:
            image_np: (H, W, 3) uint8 numpy array
            prompt: text prompt

        Returns:
            dict with keys:
                segments: (H, W) superpixel map
                importances: (n_features,) importance scores
                original_text: model output on original image
                model: fitted Ridge model
        """
        segments = self._segment_image(image_np)
        n_features = segments.max() + 1

        # Get original output
        original_text = self._get_model_output(image_np, prompt, max_tokens)

        # Generate perturbations
        perturbation_masks, perturbed_images = self._perturb(
            image_np, segments, n_features
        )

        # Get outputs for each perturbation
        similarities = np.zeros(self.num_samples)
        for i, pimg in enumerate(perturbed_images):
            text = self._get_model_output(pimg, prompt, max_tokens)
            similarities[i] = self._text_similarity(original_text, text)

        # Compute distances from original (all-ones) for kernel weighting
        distances = cosine_distances(
            perturbation_masks[0:1], perturbation_masks
        ).flatten()
        kernel_weights = np.sqrt(
            np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        )

        # Fit weighted linear model
        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbation_masks, similarities,
                  sample_weight=kernel_weights)

        importances = ridge.coef_

        return {
            "segments": segments,
            "importances": importances,
            "original_text": original_text,
            "model": ridge,
            "n_features": n_features,
        }

    def get_heatmap(
        self, segments: np.ndarray, importances: np.ndarray,
        image_size: int = 224,
    ) -> np.ndarray:
        """Convert superpixel importances to a pixel-level heatmap."""
        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for seg_id in range(len(importances)):
            heatmap[segments == seg_id] = importances[seg_id]

        # Normalize to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            heatmap = (heatmap - hmin) / (hmax - hmin)

        return heatmap
