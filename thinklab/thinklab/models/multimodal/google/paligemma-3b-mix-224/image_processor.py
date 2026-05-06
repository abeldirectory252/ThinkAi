"""PaliGemma image preprocessor — SigLIP at 224×224."""
from typing import Union
from pathlib import Path
import numpy as np, torch
from PIL import Image

class PaliGemmaImageProcessor:
    def __init__(self, image_size=224, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        self.image_size = image_size
        self.mean, self.std = torch.tensor(mean).view(3,1,1), torch.tensor(std).view(3,1,1)

    def __call__(self, image: Union[str, Path, Image.Image, np.ndarray],
                 dtype=torch.bfloat16):
        if isinstance(image, (str, Path)): image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray): image = Image.fromarray(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        t = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        return ((t - self.mean) / self.std).unsqueeze(0).to(dtype)
