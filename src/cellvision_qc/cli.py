"""Command-line interface for CellVision-QC.

Commands
--------
generate-demo   Generate synthetic demo data.
run             Run preprocessing, segmentation, QC, and feature extraction.
compare         Compare classification performance across multiple runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from cellvision_qc import __version__


@click.group()
@click.version_option(version=__version__, prog_name="cellvision-qc")
def main() -> None:
    """CellVision-QC: fluorescence microscopy image quality control and
    segmentation comparison."""


# ---------------------------------------------------------------------------
# generate-demo
# ---------------------------------------------------------------------------


@main.command("generate-demo")
@click.option(
    "--out",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to write synthetic demo data.",
)
@click.option("--n-images", default=10, show_default=True, help="Number of images.")
@click.option("--image-size", default=256, show_default=True, help="Image side length (px).")
@click.option("--cells-per-image", default=20, show_default=True, help="Target cells per image.")
@click.option("--seed", default=0, show_default=True, help="Random seed.")
def generate_demo(
    output_dir: str,
    n_images: int,
    image_size: int,
    cells_per_image: int,
    seed: int,
) -> None:
    """Generate synthetic fluorescence microscopy demo data."""
    from cellvision_qc.data.synthetic import generate_demo_dataset

    generate_demo_dataset(
        output_dir=Path(output_dir),
        n_images=n_images,
        image_size=image_size,
        cells_per_image=cells_per_image,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@main.command("run")
@click.option(
    "--images",
    "images_dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing TIFF/PNG/JPEG images.",
)
@click.option(
    "--labels",
    "labels_csv",
    required=True,
    type=click.Path(exists=True),
    help="CSV with columns: filename, phenotype.",
)
@click.option(
    "--out",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to write results.",
)
@click.option(
    "--method",
    default="threshold",
    show_default=True,
    type=click.Choice(["threshold", "watershed"], case_sensitive=False),
    help="Segmentation method.",
)
@click.option(
    "--background-radius",
    default=50,
    show_default=True,
    help="Radius for rolling-ball background subtraction.",
)
@click.option(
    "--gaussian-sigma",
    default=None,
    type=float,
    help="Gaussian smoothing sigma (disabled if omitted).",
)
@click.option(
    "--min-object-size",
    default=50,
    show_default=True,
    help="Minimum object area in pixels.",
)
@click.option(
    "--overlays/--no-overlays",
    default=True,
    show_default=True,
    help="Save segmentation overlay PNGs.",
)
def run(
    images_dir: str,
    labels_csv: str,
    output_dir: str,
    method: str,
    background_radius: int,
    gaussian_sigma: Optional[float],
    min_object_size: int,
    overlays: bool,
) -> None:
    """Run the full CellVision-QC pipeline on a set of images."""
    import numpy as np

    from cellvision_qc.features.extraction import aggregate_features, extract_features
    from cellvision_qc.metrics.qc import compute_qc_metrics
    from cellvision_qc.phenotype.analysis import PhenotypeClassifier
    from cellvision_qc.preprocessing import PreprocessingConfig, load_and_preprocess, load_image
    from cellvision_qc.segmentation import get_segmenter
    from cellvision_qc.visualization.plots import (
        plot_feature_distributions,
        plot_segmentation_overlay,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    if overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(images_dir)
    labels_df = pd.read_csv(labels_csv)
    label_map: dict[str, str] = dict(zip(labels_df["filename"], labels_df["phenotype"]))

    prep_config = PreprocessingConfig(
        background_radius=background_radius,
        gaussian_sigma=gaussian_sigma,
    )
    segmenter = get_segmenter(method)

    click.echo(f"[cellvision-qc] method={method}  images={images_dir}  out={output_dir}")

    all_features: list[pd.DataFrame] = []
    all_qc: list[dict] = []

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    )

    if not image_paths:
        click.echo(f"[warn] No images found in {images_dir}", err=True)
        return

    for img_path in image_paths:
        phenotype = label_map.get(img_path.name, "unknown")
        click.echo(f"  processing {img_path.name}  (phenotype={phenotype})")

        raw = load_image(img_path)
        preprocessed = load_and_preprocess(img_path, prep_config)

        result = segmenter.segment(preprocessed)
        click.echo(f"    → {result.n_objects} objects segmented")

        qc = compute_qc_metrics(result, image_name=img_path.name)
        qc_dict = qc.to_dict()
        qc_dict["phenotype"] = phenotype
        all_qc.append(qc_dict)

        feats = extract_features(
            result,
            preprocessed,
            image_name=img_path.name,
            label_column="phenotype",
            label_value=phenotype,
        )
        all_features.append(feats)

        if overlays and result.n_objects > 0:
            overlay_path = overlays_dir / f"{img_path.stem}_overlay.png"
            plot_segmentation_overlay(preprocessed, result.label_image, method, overlay_path)

    qc_df = pd.DataFrame(all_qc)
    qc_df.to_csv(output_dir / "qc_metrics.csv", index=False)
    click.echo(f"\n  QC metrics → {output_dir / 'qc_metrics.csv'}")

    features_df = aggregate_features(all_features)
    if not features_df.empty:
        features_df.to_csv(output_dir / "features.csv", index=False)
        click.echo(f"  Features   → {output_dir / 'features.csv'}")

        if "phenotype" in features_df.columns and features_df["phenotype"].nunique() > 1:
            clf = PhenotypeClassifier()
            report = clf.evaluate(features_df, label_col="phenotype", method_name=method)
            report_dict = report.to_dict()
            (output_dir / "phenotype_report.json").write_text(
                json.dumps(report_dict, indent=2)
            )
            click.echo(f"  Phenotype  → {output_dir / 'phenotype_report.json'}")
            click.echo(
                f"  Classifier accuracy: {report.accuracy_mean:.3f} ± {report.accuracy_std:.3f}"
            )

            feat_dist_path = output_dir / "feature_distributions.png"
            plot_feature_distributions(features_df, feat_dist_path, label_col="phenotype")
            click.echo(f"  Plot       → {feat_dist_path}")

    click.echo(f"\n[cellvision-qc] Run complete. Results in {output_dir}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@main.command("compare")
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to write comparison results.",
)
@click.option(
    "--classifier",
    default="random_forest",
    show_default=True,
    type=click.Choice(["logistic", "random_forest"], case_sensitive=False),
)
def compare(run_dirs: tuple[str, ...], output_dir: str, classifier: str) -> None:
    """Compare phenotype classification performance across segmentation runs.

    RUN_DIRS are directories produced by `cellvision-qc run`. Each must
    contain features.csv.
    """
    from cellvision_qc.phenotype.analysis import compare_phenotypes
    from cellvision_qc.visualization.plots import plot_comparison_bar, plot_qc_radar

    dirs = [Path(d) for d in run_dirs]
    out = Path(output_dir)

    click.echo(f"[cellvision-qc compare] Comparing {len(dirs)} runs → {out}")

    summary_df = compare_phenotypes(dirs, out, classifier_type=classifier)

    if summary_df.empty:
        click.echo("[warn] No valid runs to compare.")
        return

    click.echo(summary_df.to_string(index=False))

    bar_path = out / "performance_comparison.png"
    plot_comparison_bar(summary_df, bar_path)
    click.echo(f"  Bar chart  → {bar_path}")

    qc_rows = []
    for d in dirs:
        qc_path = d / "qc_metrics.csv"
        if qc_path.exists():
            qdf = pd.read_csv(qc_path)
            agg = qdf.mean(numeric_only=True).to_dict()
            agg["method"] = d.name
            qc_rows.append(agg)

    if len(qc_rows) > 1:
        radar_path = out / "qc_radar.png"
        plot_qc_radar(qc_rows, radar_path)
        click.echo(f"  Radar      → {radar_path}")

    click.echo(f"\n[cellvision-qc compare] Done. Results in {out}")
