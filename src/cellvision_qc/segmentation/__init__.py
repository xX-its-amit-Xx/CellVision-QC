"""Segmentation backends for CellVision-QC."""

from cellvision_qc.segmentation.base import SegmentationResult, Segmenter
from cellvision_qc.segmentation.cellprofiler_adapter import CellProfilerAdapter
from cellvision_qc.segmentation.cellpose_adapter import CellposeAdapter
from cellvision_qc.segmentation.threshold import ThresholdSegmenter
from cellvision_qc.segmentation.watershed import WatershedSegmenter

__all__ = [
    "Segmenter",
    "SegmentationResult",
    "ThresholdSegmenter",
    "WatershedSegmenter",
    "CellposeAdapter",
    "CellProfilerAdapter",
]


def get_segmenter(method: str) -> "Segmenter":
    """Factory: return a Segmenter instance by name."""
    registry: dict[str, type[Segmenter]] = {
        "threshold": ThresholdSegmenter,
        "watershed": WatershedSegmenter,
        "cellpose": CellposeAdapter,
        "cellprofiler": CellProfilerAdapter,
    }
    if method not in registry:
        raise ValueError(
            f"Unknown segmentation method '{method}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[method]()
