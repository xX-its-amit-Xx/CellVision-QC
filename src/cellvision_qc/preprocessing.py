"""Image preprocessing pipeline for fluorescence microscopy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import exposure, io, util
from skimage.color import rgb2gray


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    background_radius: int = 50
    gaussian_sigma: Optional[float] = None
    normalize: bool = True
    clip_percentile: tuple[float, float] = field(default_factory=lambda: (1.0, 99.0))


def load_image(path: Path | str) -> np.ndarray:
    """Load a microscopy image as a 2-D float64 array in [0, 1].

    Handles TIFF, PNG, JPEG. Multi-channel images are converted to grayscale.
    """
    path = Path(path)
    img = io.imread(str(path))

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = rgb2gray(img)

    img = img.astype(np.float64)

    if img.max() > 1.0:
        dtype_max = np.iinfo(np.uint16).max if img.max() > 255 else 255
        img = img / dtype_max

    return img


def subtract_background(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """Rolling-ball background subtraction via uniform filter approximation.

    Estimates a smooth background using a large sliding-window mean and
    subtracts it, clipping the result to [0, 1].
    """
    background = uniform_filter(image, size=radius * 2 + 1)
    corrected = image - background
    return np.clip(corrected, 0, None)


def normalize_intensity(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Percentile-based intensity normalization to [0, 1]."""
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    if p_high == p_low:
        return np.zeros_like(image)
    normalized = (image - p_low) / (p_high - p_low)
    return np.clip(normalized, 0.0, 1.0)


def smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing."""
    return gaussian_filter(image, sigma=sigma)


def preprocess(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Run the full preprocessing pipeline on a single image.

    Steps applied in order:
    1. Background subtraction (rolling-ball approximation)
    2. Optional Gaussian smoothing
    3. Percentile-based intensity normalization
    """
    img = subtract_background(image, radius=config.background_radius)

    if config.gaussian_sigma is not None and config.gaussian_sigma > 0:
        img = smooth(img, sigma=config.gaussian_sigma)

    if config.normalize:
        img = normalize_intensity(
            img,
            percentile_low=config.clip_percentile[0],
            percentile_high=config.clip_percentile[1],
        )

    return img


def load_and_preprocess(
    path: Path | str, config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """Convenience wrapper: load an image from disk and preprocess it."""
    if config is None:
        config = PreprocessingConfig()
    img = load_image(path)
    return preprocess(img, config)
