"""Tests for cellvision_qc.segmentation backends."""

from __future__ import annotations

import numpy as np
import pytest

from cellvision_qc.segmentation.base import SegmentationResult
from cellvision_qc.segmentation.threshold import ThresholdSegmenter
from cellvision_qc.segmentation.watershed import WatershedSegmenter


class TestThresholdSegmenter:
    def test_returns_segmentation_result(self, synthetic_image: np.ndarray) -> None:
        seg = ThresholdSegmenter()
        result = seg.segment(synthetic_image)
        assert isinstance(result, SegmentationResult)

    def test_detects_objects(self, synthetic_image: np.ndarray) -> None:
        seg = ThresholdSegmenter(min_object_size=50)
        result = seg.segment(synthetic_image)
        assert result.n_objects >= 5

    def test_label_image_shape(self, synthetic_image: np.ndarray) -> None:
        seg = ThresholdSegmenter()
        result = seg.segment(synthetic_image)
        assert result.label_image.shape == synthetic_image.shape

    def test_label_image_dtype_integer(self, synthetic_image: np.ndarray) -> None:
        seg = ThresholdSegmenter()
        result = seg.segment(synthetic_image)
        assert np.issubdtype(result.label_image.dtype, np.integer)

    def test_method_name(self, synthetic_image: np.ndarray) -> None:
        seg = ThresholdSegmenter()
        result = seg.segment(synthetic_image)
        assert result.method == "threshold"

    def test_blank_image_returns_zero_objects(self) -> None:
        blank = np.zeros((64, 64), dtype=np.float64)
        seg = ThresholdSegmenter()
        result = seg.segment(blank)
        assert result.n_objects == 0


class TestWatershedSegmenter:
    def test_returns_segmentation_result(self, synthetic_image: np.ndarray) -> None:
        seg = WatershedSegmenter(min_distance=10)
        result = seg.segment(synthetic_image)
        assert isinstance(result, SegmentationResult)

    def test_detects_objects(self, synthetic_image: np.ndarray) -> None:
        seg = WatershedSegmenter(min_distance=8, min_object_size=30)
        result = seg.segment(synthetic_image)
        assert result.n_objects >= 5

    def test_label_image_shape(self, synthetic_image: np.ndarray) -> None:
        seg = WatershedSegmenter()
        result = seg.segment(synthetic_image)
        assert result.label_image.shape == synthetic_image.shape

    def test_method_name(self, synthetic_image: np.ndarray) -> None:
        seg = WatershedSegmenter()
        result = seg.segment(synthetic_image)
        assert result.method == "watershed"


class TestSegmentationResult:
    def test_n_objects_property(self) -> None:
        labels = np.array([[0, 1, 1], [2, 2, 0], [0, 3, 3]])
        result = SegmentationResult(label_image=labels, method="test")
        assert result.n_objects == 3

    def test_shape_property(self) -> None:
        labels = np.zeros((128, 256), dtype=np.int32)
        result = SegmentationResult(label_image=labels, method="test")
        assert result.shape == (128, 256)
