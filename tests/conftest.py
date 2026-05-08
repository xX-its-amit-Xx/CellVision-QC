"""Shared pytest fixtures for CellVision-QC tests."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import disk


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_image(rng: np.random.Generator) -> np.ndarray:
    """256×256 float64 image with ~10 bright disk-shaped cells."""
    image = rng.normal(0.05, 0.01, size=(256, 256)).clip(0, 1)
    centers = [
        (40, 40), (40, 120), (40, 200),
        (120, 40), (120, 120), (120, 200),
        (200, 40), (200, 120), (200, 200),
    ]
    for cy, cx in centers:
        rr, cc = disk((cy, cx), 15, shape=image.shape)
        image[rr, cc] = rng.uniform(0.6, 0.9)
    return image.astype(np.float64)


@pytest.fixture
def synthetic_label_image(synthetic_image: np.ndarray) -> np.ndarray:
    """Label image corresponding to synthetic_image."""
    from cellvision_qc.segmentation.threshold import ThresholdSegmenter
    seg = ThresholdSegmenter(min_object_size=50)
    result = seg.segment(synthetic_image)
    return result.label_image
