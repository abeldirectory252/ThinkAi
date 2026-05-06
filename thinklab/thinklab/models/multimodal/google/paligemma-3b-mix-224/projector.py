"""PaliGemma projector: simple Linear(vis_dim, txt_dim)."""
import torch.nn as nn

class MultiModalProjector(nn.Module):
    def __init__(self, vis_dim=1152, txt_dim=2048):
        super().__init__()
        self.linear = nn.Linear(vis_dim, txt_dim)
    def forward(self, x): return self.linear(x)
