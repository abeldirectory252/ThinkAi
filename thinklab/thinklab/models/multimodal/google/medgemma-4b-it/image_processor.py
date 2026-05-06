"""
MedGemma image preprocessor — SigLIP normalisation at 896×896.
"""
from typing import Union
from pathlib import Path
import numpy as np
import torch
from PIL import Image

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD  = (0.5, 0.5, 0.5)


class MedGemmaImageProcessor:
    def __init__(self, image_size: int = 896, mean=SIGLIP_MEAN, std=SIGLIP_STD):
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: Union[str, Path, Image.Image, np.ndarray],
                 dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0).to(dtype)
