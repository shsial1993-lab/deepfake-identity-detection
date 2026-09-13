import torch

from src.model import TemporalArtifactDetector


def test_detector_supports_masked_clips() -> None:
    model = TemporalArtifactDetector(feature_dim=8, hidden=4)
    output = model(torch.randn(2, 5, 8), torch.ones(2, 5))
    assert output.shape == (2,)
