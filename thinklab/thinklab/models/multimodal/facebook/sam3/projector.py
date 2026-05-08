"""
SAM3 Text Projection: projects CLIP text features to decoder dimension.
CLIP text_encoder outputs [B, seq, 1024] → projected to [B, seq, 256] for decoder.
SAM3 Text Projection — kept for compatibility, but actual projection lives in model.py.
thinklab\thinklab\models\multimodal\facebook\sam3\projector.py
"""
        

import torch.nn as nn


class Sam3TextProjection(nn.Module):
    def __init__(self, text_dim=1024, decoder_dim=256):
        super().__init__()
        self.linear = nn.Linear(text_dim, decoder_dim)

    def forward(self, text_features):
        return self.linear(text_features)