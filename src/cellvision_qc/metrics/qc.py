"""Segmentation quality-control metric computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from skimage.measure import regionprops

from cellvision_qc.segmentation.base import SegmentationResult


@dataclass
class SegmentationQCMetrics:
    """All QC metrics for a single segmentation result.

    Attributes
    ----------
    method:
        Name of the segmentation method.
    image_name:
        Identifier for the source image.
    n_objects:
        Total number of segmented objects.
    mean_area:
        Mean object area in pixels.
    cv_area:
        Coefficient of variation (std/mean) of object areas. High CV may
        indicate under- or over-segmentation artefacts.
    coverage_fraction:
        Fraction of image pixels assigned to any object (mask density).
    n_border_objects:
        Number of objects whose bounding box touches the image border.
        These are likely truncated and may bias downstream statistics.
    mean_eccentricity:
        Mean eccentricity across objects (0 = circle, 1 = line).
    std_eccentricity:
        Standard deviation of eccentricity.
    mean_solidity:
        Mean solidity (area / convex hull area). Low solidity indicates
        irregular or concave morphology.
    std_solidity:
        Standard deviation of solidity.
    """

    method: str
    image_name: str
    n_objects: int
    mean_area: float
    cv_area: float
    coverage_fraction: float
    n_border_objects: int
    mean_eccentricity: float
    std_eccentricity: float
    mean_solidity: float
    std_solidity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _touches_border(region, image_shape: tuple[int, int]) -> bool:
    """Return True if a region's bounding box touches the image edge."""
    min_row, min_col, max_row, max_col = region.bbox
    h, w = image_shape
    return min_row == 0 or min_col == 0 or max_row == h or max_col == w


def compute_qc_metrics(
    result: SegmentationResult,
    image_name: str = "image",
) -> SegmentationQCMetrics:
    """Compute all QC metrics from a :class:`SegmentationResult`.

    Parameters
    ----------
    result:
        Segmentation output containing a label image.
    image_name:
        Identifier string for the source image (used in reports).

    Returns
    -------
    SegmentationQCMetrics
    """
    labels = result.label_image
    image_shape = labels.shape[:2]
    total_pixels = image_shape[0] * image_shape[1]

    regions = regionprops(labels)

    if len(regions) == 0:
        return SegmentationQCMetrics(
            method=result.method,
            image_name=image_name,
            n_objects=0,
            mean_area=0.0,
            cv_area=0.0,
            coverage_fraction=0.0,
            n_border_objects=0,
            mean_eccentricity=0.0,
            std_eccentricity=0.0,
            mean_solidity=0.0,
            std_solidity=0.0,
        )

    areas = np.array([r.area for r in regions], dtype=float)
    eccentricities = np.array([r.eccentricity for r in regions], dtype=float)
    solidities = np.array([r.solidity for r in regions], dtype=float)

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    cv_area = std_area / mean_area if mean_area > 0 else 0.0

    coverage = float(np.sum(labels > 0)) / total_pixels

    n_border = sum(1 for r in regions if _touches_border(r, image_shape))

    return SegmentationQCMetrics(
        method=result.method,
        image_name=image_name,
        n_objects=len(regions),
        mean_area=mean_area,
        cv_area=cv_area,
        coverage_fraction=coverage,
        n_border_objects=n_border,
        mean_eccentricity=float(np.mean(eccentricities)),
        std_eccentricity=float(np.std(eccentricities)),
        mean_solidity=float(np.mean(solidities)),
        std_solidity=float(np.std(solidities)),
    )
