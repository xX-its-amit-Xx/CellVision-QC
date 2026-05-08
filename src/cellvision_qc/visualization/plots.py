"""Publication-adjacent visualization for segmentation QC and phenotype analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

_PALETTE = sns.color_palette("Set2", n_colors=8)


def plot_segmentation_overlay(
    image: np.ndarray,
    label_image: np.ndarray,
    method: str,
    output_path: Path | str,
    alpha: float = 0.4,
) -> None:
    """Save a false-colour segmentation overlay PNG.

    Each object is rendered in a distinct colour overlaid on the grayscale
    image. Object boundaries are drawn as thin white lines.

    Parameters
    ----------
    image:
        Preprocessed grayscale image, float [0, 1], shape (H, W).
    label_image:
        Integer label image, shape (H, W).
    method:
        Method name for the title.
    output_path:
        Destination PNG path.
    alpha:
        Transparency of the colour overlay.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_image = np.stack([image] * 3, axis=-1)
    overlay = label2rgb(label_image, image=rgb_image, alpha=alpha, bg_label=0)

    boundaries = find_boundaries(label_image, mode="outer")
    overlay[boundaries] = [1.0, 1.0, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Preprocessed image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    n_obj = int(label_image.max())
    axes[1].set_title(
        f"{method} segmentation  ({n_obj} objects)", fontsize=11, fontweight="bold"
    )
    axes[1].axis("off")

    fig.suptitle(output_path.stem, fontsize=9, color="gray", y=0.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_distributions(
    features_df: pd.DataFrame,
    output_path: Path | str,
    label_col: str = "phenotype",
    feature_cols: Optional[Sequence[str]] = None,
) -> None:
    """Plot per-phenotype feature distribution violin plots.

    Parameters
    ----------
    features_df:
        Per-object feature table with a phenotype label column.
    output_path:
        Destination PNG path.
    label_col:
        Column containing phenotype labels.
    feature_cols:
        Which feature columns to plot. Defaults to the standard six.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = [
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "mean_intensity",
            "integrated_intensity",
        ]
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.8 * n_rows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, feature_cols):
        sns.violinplot(
            data=features_df,
            x=label_col,
            y=feat,
            ax=ax,
            palette=_PALETTE,
            inner="box",
            linewidth=0.8,
        )
        ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("")

    for ax in axes[len(feature_cols) :]:
        ax.set_visible(False)

    method = (
        features_df["method"].iloc[0] if "method" in features_df.columns else ""
    )
    fig.suptitle(
        f"Feature distributions by phenotype  |  method: {method}",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_qc_radar(
    qc_rows: list[dict],
    output_path: Path | str,
    metrics: Optional[Sequence[str]] = None,
) -> None:
    """Radar (spider) chart comparing QC metrics across methods.

    Parameters
    ----------
    qc_rows:
        List of QC metric dicts (from :class:`SegmentationQCMetrics`).
    output_path:
        Destination PNG.
    metrics:
        Metric names to include on the radar axes.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = [
            "n_objects",
            "mean_area",
            "cv_area",
            "coverage_fraction",
            "mean_eccentricity",
            "mean_solidity",
        ]
    metrics = list(metrics)
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for i, row in enumerate(qc_rows):
        values = []
        for m in metrics:
            v = row.get(m, 0.0)
            values.append(float(v) if v is not None else 0.0)

        raw = np.array(values)
        normed = raw / (raw.max() + 1e-9)
        normed = normed.tolist() + normed[:1].tolist()

        ax.plot(angles, normed, color=_PALETTE[i], linewidth=2, label=row.get("method", f"method {i}"))
        ax.fill(angles, normed, color=_PALETTE[i], alpha=0.1)

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [m.replace("_", "\n") for m in metrics],
        fontsize=8,
    )
    ax.set_yticklabels([])
    ax.set_title("QC Metrics Comparison (normalised)", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_comparison_bar(
    summary_df: pd.DataFrame,
    output_path: Path | str,
    metrics: Optional[Sequence[str]] = None,
) -> None:
    """Grouped bar chart comparing classifier performance across methods.

    Parameters
    ----------
    summary_df:
        DataFrame output of :func:`compare_phenotypes` with one row per method.
    output_path:
        Destination PNG.
    metrics:
        Which performance columns to plot. Defaults to accuracy, F1, ROC-AUC.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = ["accuracy_mean", "f1_mean", "roc_auc_mean"]
    metrics = [m for m in metrics if m in summary_df.columns]

    melted = summary_df.melt(
        id_vars=["method"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    err_map = {"accuracy_mean": "accuracy_std", "f1_mean": "f1_std", "roc_auc_mean": "roc_auc_std"}
    err_col = melted["metric"].map(err_map)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=melted,
        x="metric",
        y="score",
        hue="method",
        ax=ax,
        palette=_PALETTE,
        edgecolor="white",
        linewidth=0.6,
    )

    for i, bar in enumerate(ax.patches):
        h = bar.get_height()
        if np.isnan(h):
            continue

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(
        "Phenotype classifier performance by segmentation method",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticklabels(
        [m.replace("_mean", "").replace("_", " ").upper() for m in metrics]
    )
    ax.legend(title="Method", loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
