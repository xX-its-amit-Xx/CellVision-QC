"""Classical intensity-thresholding segmentation backend."""

from __future__ import annotations

import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label
from skimage.segmentation import clear_border

from cellvision_qc.segmentation.base import SegmentationResult, Segmenter


class ThresholdSegmenter(Segmenter):
    """Otsu global thresholding with optional morphological cleanup.

    This is the simplest baseline segmentation strategy. An Otsu threshold
    separates foreground (cells) from background, followed by binary
    opening to remove small debris and then connected-component labelling.

    Parameters
    ----------
    min_object_size:
        Minimum object area in pixels. Smaller objects are discarded.
    remove_border_objects:
        If True, objects touching the image border are removed.
    use_local:
        If True, use adaptive (local) thresholding instead of Otsu.
    local_block_size:
        Block size for adaptive thresholding (must be odd).
    """

    name = "threshold"

    def __init__(
        self,
        min_object_size: int = 50,
        remove_border_objects: bool = False,
        use_local: bool = False,
        local_block_size: int = 51,
    ) -> None:
        self.min_object_size = min_object_size
        self.remove_border_objects = remove_border_objects
        self.use_local = use_local
        self.local_block_size = local_block_size

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment via Otsu (or adaptive) thresholding."""
        if self.use_local:
            thresh = threshold_local(image, block_size=self.local_block_size)
            binary = image > thresh
        else:
            thresh_val = threshold_otsu(image)
            binary = image > thresh_val

        binary = morphology.opening(binary, morphology.disk(2))
        binary = morphology.remove_small_objects(binary, max_size=self.min_object_size)

        if self.remove_border_objects:
            binary = clear_border(binary)

        labels = label(binary)

        return SegmentationResult(
            label_image=labels,
            method=self.name,
            metadata={
                "use_local": self.use_local,
                "min_object_size": self.min_object_size,
            },
        )
