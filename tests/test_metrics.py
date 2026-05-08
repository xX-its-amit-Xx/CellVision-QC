"""Tests for cellvision_qc.metrics.qc."""

from __future__ import annotations

import numpy as np
import pytest

from cellvision_qc.metrics.qc import SegmentationQCMetrics, compute_qc_metrics
from cellvision_qc.segmentation.base import SegmentationResult


def _make_result(label_image: np.ndarray, method: str = "test") -> SegmentationResult:
    return SegmentationResult(label_image=label_image, method=method)


class TestComputeQCMetrics:
    def test_empty_segmentation(self) -> None:
        labels = np.zeros((64, 64), dtype=np.int32)
        result = _make_result(labels)
        qc = compute_qc_metrics(result)
        assert qc.n_objects == 0
        assert qc.mean_area == 0.0

    def test_n_objects_correct(self, synthetic_label_image: np.ndarray) -> None:
        n_expected = int(synthetic_label_image.max())
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert qc.n_objects == n_expected

    def test_mean_area_positive(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert qc.mean_area > 0

    def test_coverage_fraction_in_range(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert 0.0 <= qc.coverage_fraction <= 1.0

    def test_cv_area_nonnegative(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert qc.cv_area >= 0.0

    def test_eccentricity_in_range(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert 0.0 <= qc.mean_eccentricity <= 1.0

    def test_solidity_in_range(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert 0.0 <= qc.mean_solidity <= 1.0

    def test_border_objects_nonnegative(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        assert qc.n_border_objects >= 0

    def test_to_dict_keys(self, synthetic_label_image: np.ndarray) -> None:
        result = _make_result(synthetic_label_image)
        qc = compute_qc_metrics(result)
        d = qc.to_dict()
        expected = {
            "method", "image_name", "n_objects", "mean_area", "cv_area",
            "coverage_fraction", "n_border_objects", "mean_eccentricity",
            "std_eccentricity", "mean_solidity", "std_solidity",
        }
        assert expected.issubset(d.keys())

    def test_image_name_propagated(self) -> None:
        labels = np.zeros((32, 32), dtype=np.int32)
        result = _make_result(labels, method="threshold")
        qc = compute_qc_metrics(result, image_name="test_img.tif")
        assert qc.image_name == "test_img.tif"
        assert qc.method == "threshold"
