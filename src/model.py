from __future__ import annotations

import torch
from torch import nn


class TemporalArtifactDetector(nn.Module):
    """Classify a clip represented as a sequence of frame embeddings."""

    def __init__(self, feature_dim: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(
        self, frame_embeddings: torch.Tensor, valid_frames: torch.Tensor | None = None
    ) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError('expected [batch, time, feature_dim]')
        temporal = self.temporal(frame_embeddings.transpose(1, 2)).transpose(1, 2)
        if valid_frames is None:
            pooled = temporal.mean(dim=1)
        else:
            weights = valid_frames.float().unsqueeze(-1)
            pooled = (temporal * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.classifier(pooled).squeeze(-1)
