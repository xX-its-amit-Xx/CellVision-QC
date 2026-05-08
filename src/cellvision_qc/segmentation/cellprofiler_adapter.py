"""Placeholder adapter for CellProfiler segmentation exports.

CellProfiler produces per-object CSV tables and optionally exports binary or
label TIFF masks. This adapter provides the interface for ingesting those
exports into CellVision-QC for downstream QC and comparison.

Usage (user-provided workflow)
-------------------------------
Export from CellProfiler:
  - In your pipeline add "SaveImages" (masks) or "ExportToSpreadsheet" (CSVs).
  - Recommended: export per-object outlines as 16-bit label TIFFs.

Load into CellVision-QC:

    from cellvision_qc.segmentation import CellProfilerAdapter
    adapter = CellProfilerAdapter()

    # Option A: from a label mask TIFF
    result = adapter.load_mask("cp_nuclei.tif")

    # Option B: from a CellProfiler object CSV
    result_df = adapter.load_object_csv("Nuclei.csv")

Disclaimer
----------
CellProfiler is a third-party tool maintained by the Broad Institute Imaging
Platform and is not affiliated with or bundled in this package. This adapter
provides structural interoperability only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cellvision_qc.segmentation.base import SegmentationResult, Segmenter


class CellProfilerAdapter(Segmenter):
    """Adapter for CellProfiler label mask and CSV exports.

    Parameters
    ----------
    object_name:
        CellProfiler object name (e.g. ``"Nuclei"``, ``"Cells"``). Used
        to validate CSV column prefixes.
    min_object_size:
        Objects smaller than this pixel count are discarded on mask load.
    """

    name = "cellprofiler"

    def __init__(
        self,
        object_name: str = "Cells",
        min_object_size: int = 50,
    ) -> None:
        self.object_name = object_name
        self.min_object_size = min_object_size

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Not applicable — CellProfiler requires its own runtime.

        Raises
        ------
        NotImplementedError
            Always. Use :meth:`load_mask` or :meth:`load_object_csv` instead.
        """
        raise NotImplementedError(
            "CellProfilerAdapter cannot segment images directly. "
            "Run CellProfiler externally and load results with "
            "load_mask() or load_object_csv()."
        )

    def load_mask(self, mask_path: Path | str) -> SegmentationResult:
        """Load a CellProfiler exported label mask TIFF.

        Parameters
        ----------
        mask_path:
            Path to a 16-bit TIFF label image produced by CellProfiler's
            SaveImages module (e.g. ``CellsLabels.tif``).

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
                f"Expected 2-D label image, got shape {labels.shape}."
            )

        labels = self._remove_small(labels)

        return SegmentationResult(
            label_image=labels,
            method=self.name,
            metadata={"source": str(mask_path), "object_name": self.object_name},
        )

    def load_object_csv(self, csv_path: Path | str) -> pd.DataFrame:
        """Load a CellProfiler per-object measurement CSV.

        Returns a :class:`pandas.DataFrame` with measurements indexed by
        ``ObjectNumber``. Column names are preserved as exported by
        CellProfiler (e.g. ``AreaShape_Area``, ``Intensity_MeanIntensity``).

        Parameters
        ----------
        csv_path:
            Path to the CellProfiler object-level CSV.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "ObjectNumber" in df.columns:
            df = df.set_index("ObjectNumber")

        return df

    def _remove_small(self, labels: np.ndarray) -> np.ndarray:
        from skimage.measure import regionprops

        for region in regionprops(labels):
            if region.area < self.min_object_size:
                labels[labels == region.label] = 0
        return labels
