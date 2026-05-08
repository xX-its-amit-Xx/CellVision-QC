"""Per-object morphological and intensity feature extraction.

Features extracted per segmented object:
- area                : number of pixels
- perimeter           : object perimeter (pixels)
- eccentricity        : 0 = circle, 1 = line segment
- solidity            : area / convex hull area
- mean_intensity      : mean pixel intensity within object
- integrated_intensity: sum of pixel intensities within object
- centroid_row        : row coordinate of centroid
- centroid_col        : column coordinate of centroid
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from cellvision_qc.segmentation.base import SegmentationResult


_REGIONPROP_KEYS = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "centroid",
]

_RENAME_MAP = {
    "centroid-0": "centroid_row",
    "centroid-1": "centroid_col",
}


def extract_features(
    result: SegmentationResult,
    image: np.ndarray,
    image_name: str = "image",
    label_column: Optional[str] = None,
    label_value: Optional[str] = None,
) -> pd.DataFrame:
    """Extract per-object features from a segmentation result.

    Parameters
    ----------
    result:
        Segmentation label image and method metadata.
    image:
        Preprocessed grayscale image (float, [0, 1]) used for intensity
        measurements. Must match the spatial dimensions of ``result.label_image``.
    image_name:
        Identifier for the source image, added as a column.
    label_column:
        Optional column name for a categorical label (e.g., ``"phenotype"``).
    label_value:
        Value to fill for ``label_column`` across all objects in this image.

    Returns
    -------
    pd.DataFrame
        One row per segmented object with the following columns:
        ``image_name``, ``method``, ``object_id``, ``area``, ``perimeter``,
        ``eccentricity``, ``solidity``, ``mean_intensity``,
        ``integrated_intensity``, ``centroid_row``, ``centroid_col``,
        and optionally ``label_column``.
    """
    labels = result.label_image
    if labels.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(
        labels,
        intensity_image=image,
        properties=_REGIONPROP_KEYS,
    )

    df = pd.DataFrame(props).rename(columns=_RENAME_MAP)
    df = df.rename(columns={"label": "object_id"})

    df["integrated_intensity"] = df["mean_intensity"] * df["area"]
    df.insert(0, "method", result.method)
    df.insert(0, "image_name", image_name)

    if label_column is not None and label_value is not None:
        df[label_column] = label_value

    return df.reset_index(drop=True)


def aggregate_features(
    dataframes: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-image feature DataFrames into a single table."""
    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)
