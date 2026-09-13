from __future__ import annotations

import torch

from .model import TemporalArtifactDetector


def main() -> None:
    torch.manual_seed(7)
    model = TemporalArtifactDetector(feature_dim=32, hidden=16)
    clips = torch.randn(4, 12, 32)
    valid = torch.ones(4, 12)
    valid[-1, 8:] = 0
    logits = model(clips, valid)
    print('clip embeddings:', tuple(clips.shape), 'logits:', tuple(logits.shape))


if __name__ == '__main__':
    main()
