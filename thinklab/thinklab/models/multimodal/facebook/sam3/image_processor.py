"""
SAM3 Image Processor — 1008×1008 normalization + post-processing.
Uses ImageNet mean/std (not SigLIP).
"""
from typing import Union, Tuple, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _scale_boxes(boxes: torch.Tensor, target_sizes: list) -> torch.Tensor:
    """Scale boxes from [0,1]-normalized xyxy to absolute pixel coordinates.

    Args:
        boxes: [B, Q, 4] in xyxy format, values in [0, 1].
        target_sizes: list of (height, width) tuples.

    Returns:
        Scaled boxes: [B, Q, 4] in absolute pixel coords.
    """
    if target_sizes is None:
        return boxes
    scaled = boxes.clone()
    for i, (h, w) in enumerate(target_sizes):
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        scaled[i] = boxes[i] * scale
    return scaled


class Sam3ImageProcessor:
    """Preprocess images for SAM3 ViT backbone at 1008×1008."""

    def __init__(self, image_size: int = 1008,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: Union[str, Path, Image.Image, np.ndarray],
                 dtype: torch.dtype = torch.float32) -> dict:
        """Preprocess image and return dict with pixel_values + original_sizes.

        Returns:
            dict with:
                pixel_values: [1, 3, 1008, 1008]
                original_sizes: [1, 2] (H, W of original)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        return {
            "pixel_values": tensor.unsqueeze(0).to(dtype),
            "original_sizes": torch.tensor([[orig_h, orig_w]], dtype=torch.int64),
        }

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: Optional[List[tuple]] = None,
    ):
        """
        Converts the raw output of Sam3Model into instance segmentation predictions
        with bounding boxes and masks.

        Args:
            outputs (dict):
                Raw outputs of the model containing pred_boxes, pred_logits,
                pred_masks, and optionally presence_logits. Can be a dict or
                any object with attribute access.
            threshold (float, optional, defaults to 0.3):
                Score threshold to keep instance predictions.
            mask_threshold (float, optional, defaults to 0.5):
                Threshold for binarizing the predicted masks.
            target_sizes (list[tuple[int, int]], optional):
                List of tuples (height, width) for each image in the batch.
                If unset, predictions will not be resized.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - scores: confidence scores for each instance
                - boxes: bounding boxes in (x1, y1, x2, y2) format
                - masks: binary segmentation masks (num_instances, H, W)
        """
        # Support both dict and attribute-style access
        if isinstance(outputs, dict):
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            pred_masks = outputs["pred_masks"]
            presence_logits = outputs.get("presence_logits")
        else:
            pred_logits = outputs.pred_logits
            pred_boxes = outputs.pred_boxes
            pred_masks = outputs.pred_masks
            presence_logits = getattr(outputs, "presence_logits", None)

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError(
                "Make sure that you pass in as many target sizes as images"
            )

        # Compute scores: combine pred_logits with presence_logits if available
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Scale boxes to target sizes if provided
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(
            zip(batch_scores, batch_boxes, batch_masks)
        ):
            # Filter by score threshold
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]  # (num_keep, height, width)

            # Resize masks to target size if provided
            if target_sizes is not None:
                target_size = target_sizes[idx]
                if len(masks) > 0:
                    masks = F.interpolate(
                        masks.unsqueeze(0),  # (1, num_keep, height, width)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (num_keep, target_height, target_width)

            # Binarize masks
            masks = (masks > mask_threshold).to(torch.long)

            results.append({
                "scores": scores,
                "boxes": boxes,
                "masks": masks,
            })

        return results
