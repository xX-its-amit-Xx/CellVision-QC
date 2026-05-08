"""Tests for cellvision_qc.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from cellvision_qc.preprocessing import (
    PreprocessingConfig,
    normalize_intensity,
    preprocess,
    smooth,
    subtract_background,
)


class TestNormalizeIntensity:
    def test_output_range(self, rng: np.random.Generator) -> None:
        img = rng.uniform(0.1, 0.9, size=(64, 64))
        out = normalize_intensity(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_uniform_image(self) -> None:
        img = np.full((32, 32), 0.5)
        out = normalize_intensity(img)
        assert np.all(out == 0.0)

    def test_preserves_shape(self, synthetic_image: np.ndarray) -> None:
        out = normalize_intensity(synthetic_image)
        assert out.shape == synthetic_image.shape


class TestSubtractBackground:
    def test_output_nonnegative(self, synthetic_image: np.ndarray) -> None:
        out = subtract_background(synthetic_image, radius=20)
        assert out.min() >= 0.0

    def test_reduces_background(self, rng: np.random.Generator) -> None:
        img = np.full((64, 64), 0.3)
        out = subtract_background(img, radius=10)
        assert out.mean() < 0.05

    def test_preserves_shape(self, synthetic_image: np.ndarray) -> None:
        out = subtract_background(synthetic_image)
        assert out.shape == synthetic_image.shape


class TestSmooth:
    def test_output_range(self, synthetic_image: np.ndarray) -> None:
        out = smooth(synthetic_image, sigma=1.5)
        assert out.min() >= 0.0

    def test_reduces_variance(self, rng: np.random.Generator) -> None:
        noisy = rng.normal(0.5, 0.2, size=(128, 128)).clip(0, 1)
        smoothed = smooth(noisy, sigma=3.0)
        assert smoothed.std() < noisy.std()


class TestPreprocess:
    def test_full_pipeline(self, synthetic_image: np.ndarray) -> None:
        config = PreprocessingConfig(background_radius=20, gaussian_sigma=1.0)
        out = preprocess(synthetic_image, config)
        assert out.shape == synthetic_image.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_no_smoothing(self, synthetic_image: np.ndarray) -> None:
        config = PreprocessingConfig(gaussian_sigma=None)
        out = preprocess(synthetic_image, config)
        assert out.dtype == np.float64

    def test_no_normalize(self, synthetic_image: np.ndarray) -> None:
        config = PreprocessingConfig(normalize=False)
        out = preprocess(synthetic_image, config)
        assert out.min() >= 0.0
