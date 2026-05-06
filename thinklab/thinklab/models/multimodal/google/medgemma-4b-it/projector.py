"""
MedGemma MultiModal Projector: RMSNorm → AvgPool → weight-only linear.
Different from PaliGemma's simple nn.Linear projector.
"""
import torch
import torch.nn as nn
from .layers import RMSNorm


class MultiModalProjector(nn.Module):
    def __init__(self, vis_dim: int = 1152, txt_dim: int = 2560,
                 image_size: int = 896, patch_size: int = 14):
        super().__init__()
        self.mm_soft_emb_norm = RMSNorm(vis_dim)
        self.mm_input_projection_weight = nn.Parameter(torch.empty(vis_dim, txt_dim))
        patches_per_side = image_size // patch_size
        self.kernel_size = 4
        self.patches_per_image = (patches_per_side // self.kernel_size) ** 2
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = self.mm_soft_emb_norm(x)
        side = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, D, side, side)
        x = self.avg_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = x @ self.mm_input_projection_weight
        return x
