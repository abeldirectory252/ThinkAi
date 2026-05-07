"""
SAM3 Image Processor — 1008×1008 normalization.
Uses ImageNet mean/std (not SigLIP).
"""
from typing import Union, Tuple
from pathlib import Path
import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        return {
            "pixel_values": tensor.unsqueeze(0).to(dtype),
            "original_sizes": torch.tensor([[orig_h, orig_w]], dtype=torch.int64),
        }
