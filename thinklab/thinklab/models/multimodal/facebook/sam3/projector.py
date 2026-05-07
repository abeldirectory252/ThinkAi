"""
SAM3 Text Projection: projects CLIP text features to decoder dimension.
CLIP text_encoder outputs [B, seq, 1024] → projected to [B, seq, 256] for decoder.
"""
import torch
import torch.nn as nn


class Sam3TextProjection(nn.Module):
    """Projects CLIP text embeddings to DETR decoder dimension.

    HF weight keys:
      text_projection.weight  → [256, 1024]
    """
    def __init__(self, text_dim: int = 1024, decoder_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(text_dim, decoder_dim)

    def forward(self, text_features):
        """
        Args:
            text_features: [B, seq_len, 1024] from CLIP text encoder
        Returns:
            [B, seq_len, 256] projected features
        """
        return self.linear(text_features)
