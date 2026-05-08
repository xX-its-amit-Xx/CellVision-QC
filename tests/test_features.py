"""Tests for cellvision_qc.features.extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellvision_qc.features.extraction import aggregate_features, extract_features
from cellvision_qc.segmentation.base import SegmentationResult


def _make_result(label_image: np.ndarray) -> SegmentationResult:
    return SegmentationResult(label_image=label_image, method="threshold")


class TestExtractFeatures:
    def test_returns_dataframe(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(result, synthetic_image)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_object(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(result, synthetic_image)
        assert len(df) == result.n_objects

    def test_expected_columns_present(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(result, synthetic_image)
        for col in ["area", "perimeter", "eccentricity", "solidity", "mean_intensity",
                    "integrated_intensity", "centroid_row", "centroid_col"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_image_name_column(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(result, synthetic_image, image_name="img_001.tif")
        assert (df["image_name"] == "img_001.tif").all()

    def test_label_column_added(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(
            result, synthetic_image, label_column="phenotype", label_value="healthy"
        )
        assert "phenotype" in df.columns
        assert (df["phenotype"] == "healthy").all()

    def test_empty_segmentation_returns_empty_df(
        self, synthetic_image: np.ndarray
    ) -> None:
        blank_labels = np.zeros_like(synthetic_image, dtype=np.int32)
        result = _make_result(blank_labels)
        df = extract_features(result, synthetic_image)
        assert df.empty

    def test_integrated_intensity(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df = extract_features(result, synthetic_image)
        expected = df["mean_intensity"] * df["area"]
        pd.testing.assert_series_equal(
            df["integrated_intensity"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestAggregateFeatures:
    def test_concatenates_correctly(
        self, synthetic_image: np.ndarray, synthetic_label_image: np.ndarray
    ) -> None:
        result = _make_result(synthetic_label_image)
        df1 = extract_features(result, synthetic_image, image_name="a.tif")
        df2 = extract_features(result, synthetic_image, image_name="b.tif")
        agg = aggregate_features([df1, df2])
        assert len(agg) == len(df1) + len(df2)

    def test_empty_list_returns_empty(self) -> None:
        agg = aggregate_features([])
        assert agg.empty
