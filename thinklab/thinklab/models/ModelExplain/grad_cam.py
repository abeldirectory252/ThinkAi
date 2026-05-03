"""Grad-CAM for PaliGemma's vision encoder."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping on SigLIP ViT layers.
    Works by hooking into a target encoder layer, capturing activations
    and gradients, then weighting activations by mean gradient per channel.
    """

    def __init__(self, model, target_layer_idx: int = -1):
        """
        Args:
            model: PaliGemma instance
            target_layer_idx: which ViT encoder layer to hook (-1 = last)
        """
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []

        # Resolve target layer
        layers = model.vision_tower.vision_model.encoder.layers
        idx = target_layer_idx % len(layers)
        target = layers[idx]

        # Register hooks
        self._hooks.append(
            target.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        # out is (hidden_state, attn_weights) tuple
        if isinstance(out, tuple):
            self.activations = out[0].detach()
        else:
            self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        if isinstance(grad_out, tuple):
            self.gradients = grad_out[0].detach()
        else:
            self.gradients = grad_out.detach()

    def compute(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        target_token_idx: int = -1,
        image_size: int = 224,
        patch_size: int = 14,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a specific output token.

        Args:
            pixel_values: (1, 3, 224, 224)
            input_ids: (1, seq_len)
            target_token_idx: which output token to explain (-1 = last)
            image_size: original image size
            patch_size: ViT patch size

        Returns:
            heatmap: (image_size, image_size) numpy array, values in [0, 1]
        """
        self.model.zero_grad()
        pixel_values = pixel_values.requires_grad_(True)

        # Forward with gradients enabled
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            out = self.model(
                pixel_values, input_ids,
                output_attentions=False,
                output_vision_hidden=False,
            )
            logits = out["logits"]  # (1, seq_len, vocab)

            # Target: logit of the predicted token at target_token_idx
            target_logit = logits[0, target_token_idx, :]
            pred_class = target_logit.argmax()
            score = target_logit[pred_class]

            # Backward
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture data. Check target layer.")

        # Grad-CAM computation
        # activations: (1, num_patches, hidden_dim)
        # gradients:   (1, num_patches, hidden_dim)
        grads = self.gradients[0]    # (num_patches, hidden)
        acts = self.activations[0]   # (num_patches, hidden)

        # Channel weights = global average of gradients
        weights = grads.mean(dim=0)  # (hidden,)

        # Weighted sum of activations
        cam = (acts * weights.unsqueeze(0)).sum(dim=-1)  # (num_patches,)
        cam = F.relu(cam)

        # Reshape to spatial grid
        grid = image_size // patch_size  # 16
        cam = cam.reshape(grid, grid)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to image size
        cam = cam.unsqueeze(0).unsqueeze(0).float()
        cam = F.interpolate(cam, size=(image_size, image_size),
                            mode="bilinear", align_corners=False)
        heatmap = cam.squeeze().cpu().numpy()

        if not was_training:
            self.model.eval()

        return heatmap

    def compute_per_token(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        generated_ids: list,
        image_size: int = 224,
        patch_size: int = 14,
    ) -> dict:
        """
        Compute Grad-CAM for each generated token.

        Returns:
            dict mapping token_index -> heatmap (H, W) numpy array
        """
        num_prefix = input_ids.shape[1]
        heatmaps = {}

        for i, tok_id in enumerate(generated_ids):
            token_pos = num_prefix + i - 1  # position in logits
            try:
                hm = self.compute(
                    pixel_values, input_ids,
                    target_token_idx=min(token_pos, num_prefix - 1),
                    image_size=image_size,
                    patch_size=patch_size,
                )
                heatmaps[i] = hm
            except Exception:
                heatmaps[i] = np.zeros((image_size, image_size))

        return heatmaps

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
