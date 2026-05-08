"""Watershed segmentation backend."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import clear_border, watershed

from cellvision_qc.segmentation.base import SegmentationResult, Segmenter


class WatershedSegmenter(Segmenter):
    """Distance-transform watershed segmentation.

    Algorithm:
    1. Otsu threshold produces a foreground binary mask.
    2. Distance transform of the binary mask emphasises object interiors.
    3. Local maxima of the distance map seed individual cell markers.
    4. Watershed fills from seeds, bounded by the binary mask.

    This approach handles touching/overlapping cells better than plain
    thresholding at the cost of slightly higher compute.

    Parameters
    ----------
    min_distance:
        Minimum number of pixels separating watershed seed peaks.
    min_object_size:
        Minimum object area in pixels.
    remove_border_objects:
        Remove objects that touch the image border.
    """

    name = "watershed"

    def __init__(
        self,
        min_distance: int = 10,
        min_object_size: int = 50,
        remove_border_objects: bool = False,
    ) -> None:
        self.min_distance = min_distance
        self.min_object_size = min_object_size
        self.remove_border_objects = remove_border_objects

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment via distance-transform watershed."""
        thresh_val = threshold_otsu(image)
        binary = image > thresh_val
        binary = morphology.opening(binary, morphology.disk(2))
        binary = morphology.remove_small_objects(binary, max_size=self.min_object_size)

        distance = ndi.distance_transform_edt(binary)

        coords = peak_local_max(
            distance,
            min_distance=self.min_distance,
            labels=binary,
        )
        markers = np.zeros(distance.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        markers, _ = ndi.label(markers)

        labels = watershed(-distance, markers, mask=binary)

        if self.remove_border_objects:
            labels = clear_border(labels)

        labels = label(labels > 0) if labels.max() == 0 else labels

        return SegmentationResult(
            label_image=labels,
            method=self.name,
            metadata={
                "min_distance": self.min_distance,
                "n_seeds": int(coords.shape[0]),
            },
        )
