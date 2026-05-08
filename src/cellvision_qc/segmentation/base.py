"""Abstract base class for all segmentation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Container for a segmentation output.

    Attributes
    ----------
    label_image:
        Integer-valued label image where 0 = background and each positive
        integer corresponds to a unique segmented object.
    method:
        Name of the segmentation method that produced this result.
    metadata:
        Arbitrary key-value pairs for method-specific diagnostics.
    """

    label_image: np.ndarray
    method: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        """Number of segmented objects (excluding background)."""
        return int(self.label_image.max())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.label_image.shape


class Segmenter(ABC):
    """Abstract base class for segmentation backends.

    Subclasses must implement :meth:`segment`.  The :meth:`segment_from_file`
    convenience method loads and preprocesses a TIFF/PNG/JPEG before calling
    :meth:`segment`.
    """

    name: str = "base"

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Segment a preprocessed 2-D float image.

        Parameters
        ----------
        image:
            Preprocessed float array in [0, 1], shape (H, W).

        Returns
        -------
        SegmentationResult
        """

    def segment_from_file(
        self,
        path: Path | str,
        preprocessing_config: Optional[object] = None,
    ) -> SegmentationResult:
        """Load, preprocess, and segment an image from disk."""
        from cellvision_qc.preprocessing import (
            PreprocessingConfig,
            load_and_preprocess,
        )

        config = preprocessing_config or PreprocessingConfig()
        image = load_and_preprocess(path, config)
        return self.segment(image)
