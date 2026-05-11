"""
SAM3 Image Processor — 1008 x 1008 with SAM3 normalization (0.5 mean/std).
image_processor.py
"""

from typing import Union, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# SAM3 uses these, NOT ImageNet!
SAM3_MEAN = (0.5, 0.5, 0.5)
SAM3_STD = (0.5, 0.5, 0.5)
RESCALE_FACTOR = 1.0 / 255.0  # 0.00392156862745098


def _scale_boxes(boxes, target_sizes):
    if target_sizes is None:
        return boxes
    scaled = boxes.clone()
    for i, (h, w) in enumerate(target_sizes):
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        scaled[i] = boxes[i] * scale
    return scaled


class Sam3ImageProcessor:
    def __init__(self, image_size=1008, mean=SAM3_MEAN, std=SAM3_STD):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image, dtype=torch.float32):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        orig_w, orig_h = image.size
        # Resize to 1008x1008
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        # Convert to [0,1] float
        arr = np.array(image).astype(np.float32) * RESCALE_FACTOR  # / 255
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        # SAM3 normalization: (x - 0.5) / 0.5
        tensor = (tensor - self.mean) / self.std
        return {
            "pixel_values": tensor.unsqueeze(0).to(dtype),
            "original_sizes": torch.tensor([[orig_h, orig_w]], dtype=torch.int64),
        }

    def post_process_instance_segmentation(
        self, outputs, threshold=0.3, mask_threshold=0.5,
        target_sizes: Optional[List[tuple]] = None,
    ):
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
            raise ValueError("target_sizes count must match batch size")

        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()
            batch_scores = batch_scores * presence_scores

        batch_masks = pred_masks.sigmoid()
        batch_boxes = pred_boxes
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(
                zip(batch_scores, batch_boxes, batch_masks)):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]
            if target_sizes is not None and len(masks) > 0:
                masks = F.interpolate(
                    masks.unsqueeze(0), size=target_sizes[idx],
                    mode="bilinear", align_corners=False).squeeze(0)
            masks = (masks > mask_threshold).to(torch.long)
            results.append({"scores": scores, "boxes": boxes, "masks": masks})
        return results