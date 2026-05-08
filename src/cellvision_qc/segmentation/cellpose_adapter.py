"""Placeholder adapter for Cellpose / Cellpose-SAM segmentation outputs.

This module defines the interface through which CellVision-QC can ingest
segmentation results produced by Cellpose or its SAM-augmented variants.
The adapter is intentionally minimal — the heavy Cellpose/SAM runtime is
**not** a dependency of this package.

Usage (user-provided workflow)
-------------------------------
1. Run Cellpose (or Cellpose-SAM) externally and save the integer label
   mask as a TIFF:

       from cellpose import models
       model = models.Cellpose(gpu=False, model_type="cyto3")
       masks, _, _, _ = model.eval(image, diameter=None, channels=[0, 0])
       tifffile.imwrite("cellpose_mask.tif", masks.astype(np.uint16))

2. Load the mask into CellVision-QC:

       from cellvision_qc.segmentation import CellposeAdapter
       adapter = CellposeAdapter()
       result = adapter.load_mask("cellpose_mask.tif")

Disclaimer
----------
Cellpose and Cellpose-SAM are third-party tools not affiliated with this
project. This adapter provides structural interoperability only and does not
ship or invoke those models directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellvision_qc.segmentation.base import SegmentationResult, Segmenter


class CellposeAdapter(Segmenter):
    """Adapter for externally generated Cellpose / Cellpose-SAM label masks.

    Parameters
    ----------
    min_object_size:
        Objects smaller than this pixel count are discarded on load.
    """

    name = "cellpose"

    def __init__(self, min_object_size: int = 50) -> None:
        self.min_object_size = min_object_size

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Not applicable — Cellpose requires its own runtime.

        Raises
        ------
        NotImplementedError
            Always. Use :meth:`load_mask` instead.
        """
        raise NotImplementedError(
            "CellposeAdapter cannot segment images directly. "
            "Run Cellpose externally and load the mask with load_mask()."
        )

    def load_mask(self, mask_path: Path | str) -> SegmentationResult:
        """Load a Cellpose integer label mask from a TIFF file.

        Parameters
        ----------
        mask_path:
            Path to a TIFF containing a uint16 integer label image output
            by Cellpose (e.g. ``*_cp_masks.tif``).

        Returns
        -------
        SegmentationResult
        """
        import tifffile

        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        labels = tifffile.imread(str(mask_path)).astype(np.int32)

        if labels.ndim != 2:
            raise ValueError(
                f"Expected 2-D label image, got shape {labels.shape}. "
                "Ensure you are loading a single-channel mask."
            )

        labels = self._remove_small(labels)

        return SegmentationResult(
            label_image=labels,
            method=self.name,
            metadata={"source": str(mask_path)},
        )

    def _remove_small(self, labels: np.ndarray) -> np.ndarray:
        from skimage.measure import regionprops

        for region in regionprops(labels):
            if region.area < self.min_object_size:
                labels[labels == region.label] = 0
        return labels
