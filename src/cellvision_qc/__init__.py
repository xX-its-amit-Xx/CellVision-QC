"""CellVision-QC: Fluorescence microscopy quality control and segmentation comparison."""

__version__ = "0.1.0"
__author__ = "Amit Shenoy"
__email__ = "shenoy.am@husky.neu.edu"

from cellvision_qc import features, metrics, preprocessing, segmentation, visualization
from cellvision_qc.data import synthetic

__all__ = [
    "__version__",
    "features",
    "metrics",
    "preprocessing",
    "segmentation",
    "visualization",
    "synthetic",
]
