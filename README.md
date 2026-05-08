# CellVision-QC

**Fluorescence microscopy image quality control, segmentation comparison, and segmentation-sensitive phenotypic analysis.**

[![CI](https://github.com/ashenoy/CellVision-QC/actions/workflows/ci.yml/badge.svg)](https://github.com/ashenoy/CellVision-QC/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CellVision-QC is a Python package for evaluating how **segmentation method choice** propagates into **downstream cell-level feature extraction** and the ability to distinguish healthy from unhealthy cellular phenotypes in fluorescence microscopy data.

The project compares a classical thresholding workflow with a distance-transform watershed approach, quantifies segmentation quality through an extensible set of QC metrics, extracts per-object morphological and intensity features, and benchmarks phenotype classification performance across segmentation strategies.

CellVision-QC is designed to be honest and reproducible. It ships with a **synthetic data generator** that requires no real biological imaging data to run. Adapters for Cellpose-SAM and CellProfiler provide interoperability hooks for users who bring their own segmentation outputs.

---

## Motivation

Segmentation quality is often treated as a binary pass/fail rather than a continuous variable that quantitatively affects scientific conclusions. In practice, the choice of segmentation algorithm — and the quality of its outputs — shapes which cells are counted, how their morphology is measured, and ultimately whether a downstream classifier can distinguish experimental conditions.

CellVision-QC makes this dependency explicit. By running multiple segmentation strategies on the same image set and comparing QC metrics alongside classification performance, it exposes how segmentation-sensitive a given phenotypic readout actually is. This is particularly relevant for high-content screening workflows where CellProfiler-style classical pipelines and deep-learning methods such as Cellpose are often used interchangeably without systematic comparison.

---

## Features

- **Preprocessing pipeline** — rolling-ball background subtraction, percentile normalization, optional Gaussian smoothing
- **Segmentation backends** — Otsu thresholding, distance-transform watershed, adapter stubs for Cellpose-SAM and CellProfiler
- **QC metrics** — object count, mean area, area CV, coverage fraction, border-touching object count, eccentricity and solidity summaries
- **Feature extraction** — area, perimeter, eccentricity, solidity, mean intensity, integrated intensity, centroid coordinates
- **Phenotype analysis** — stratified cross-validated logistic regression and random forest classifiers; accuracy, F1, ROC-AUC reporting
- **Visualization** — segmentation overlays, violin-plot feature distributions, radar QC comparison, bar-chart classifier comparison
- **Synthetic data generator** — fully parameterized, reproducible, no external data required
- **CLI** — four composable commands covering generation, analysis, and comparison
- **Docker** — single-container reproducibility

---

## Installation

```bash
# From PyPI (once published)
pip install cellvision-qc

# From source (recommended for development)
git clone https://github.com/ashenoy/CellVision-QC.git
cd CellVision-QC
pip install -e ".[dev]"
```

**Python ≥ 3.9** is required. Dependencies are listed in `pyproject.toml`.

---

## Quickstart

### 1. Generate synthetic demo data

```bash
cellvision-qc generate-demo \
    --out data/demo \
    --n-images 16 \
    --image-size 256 \
    --cells-per-image 25
```

Creates `data/demo/images/` (16 TIFF files), `data/demo/labels.csv`, and `data/demo/metadata.json`.

### 2. Run threshold segmentation

```bash
cellvision-qc run \
    --images data/demo/images \
    --labels data/demo/labels.csv \
    --out results/demo \
    --method threshold
```

### 3. Run watershed segmentation

```bash
cellvision-qc run \
    --images data/demo/images \
    --labels data/demo/labels.csv \
    --out results/demo_watershed \
    --method watershed
```

### 4. Compare methods

```bash
cellvision-qc compare \
    results/demo \
    results/demo_watershed \
    --out results/comparison \
    --classifier random_forest
```

Or run all four steps with the provided shell script:

```bash
bash examples/run_demo.sh
```

---

## Example Outputs

After a full demo run, `results/` contains:

```
results/
├── demo/
│   ├── overlays/                  # segmentation overlay PNGs (one per image)
│   ├── qc_metrics.csv             # per-image QC metrics
│   ├── features.csv               # per-object feature table
│   ├── phenotype_report.json      # classification accuracy/F1/ROC-AUC
│   └── feature_distributions.png  # violin plots by phenotype
├── demo_watershed/
│   └── ...                        # same structure, watershed method
└── comparison/
    ├── comparison_summary.csv      # cross-method classifier comparison
    ├── comparison_report.json
    ├── performance_comparison.png  # bar chart
    └── qc_radar.png               # radar chart of QC metrics
```

---

## CLI Reference

```
Usage: cellvision-qc [OPTIONS] COMMAND [ARGS]...

Commands:
  generate-demo   Generate synthetic demo dataset
  run             Preprocess, segment, extract QC + features, classify
  compare         Compare classifier performance across runs
```

### `generate-demo`

| Option | Default | Description |
|--------|---------|-------------|
| `--out` | required | Output directory |
| `--n-images` | 10 | Number of images |
| `--image-size` | 256 | Image side length (px) |
| `--cells-per-image` | 20 | Target cells per image |
| `--seed` | 0 | Random seed |

### `run`

| Option | Default | Description |
|--------|---------|-------------|
| `--images` | required | Image directory |
| `--labels` | required | CSV with `filename`, `phenotype` columns |
| `--out` | required | Output directory |
| `--method` | `threshold` | `threshold` or `watershed` |
| `--background-radius` | 50 | Rolling-ball radius (px) |
| `--gaussian-sigma` | None | Smoothing sigma (disabled if omitted) |
| `--min-object-size` | 50 | Minimum object area (px²) |
| `--overlays/--no-overlays` | on | Save overlay PNGs |

### `compare`

| Argument/Option | Description |
|-----------------|-------------|
| `RUN_DIRS` | One or more `run` output directories |
| `--out` | Comparison output directory |
| `--classifier` | `logistic` or `random_forest` |

---

## Architecture

```
src/cellvision_qc/
├── cli.py                        # Click CLI entry point
├── preprocessing.py              # Background subtraction, normalization
├── segmentation/
│   ├── base.py                   # Abstract Segmenter + SegmentationResult
│   ├── threshold.py              # Otsu / adaptive thresholding
│   ├── watershed.py              # Distance-transform watershed
│   ├── cellpose_adapter.py       # Interoperability stub (Cellpose-SAM)
│   └── cellprofiler_adapter.py   # Interoperability stub (CellProfiler)
├── metrics/
│   └── qc.py                     # SegmentationQCMetrics, compute_qc_metrics
├── features/
│   └── extraction.py             # extract_features, aggregate_features
├── phenotype/
│   └── analysis.py               # PhenotypeClassifier, ClassificationReport
├── visualization/
│   └── plots.py                  # Overlays, violin plots, radar, bar charts
└── data/
    └── synthetic.py              # Synthetic image + label generation
```

---

## Adapter Stubs: Cellpose-SAM and CellProfiler

CellVision-QC includes adapter classes that define a consistent interface for external segmentation tools:

**`CellposeAdapter`** — loads a Cellpose-exported label TIFF (e.g., `*_cp_masks.tif`) and converts it into a `SegmentationResult`. It does **not** install or invoke Cellpose/SAM; the user runs those tools independently and provides the mask.

**`CellProfilerAdapter`** — loads CellProfiler label mask TIFFs or per-object measurement CSVs. It does **not** bundle or invoke CellProfiler; that tool is run externally via its own pipeline.

These adapters are designed for **structural interoperability**: they establish a shared data contract that allows any segmentation source — classical, deep-learning, or commercial — to feed into the same downstream QC and feature-extraction pipeline.

---

## Docker

```bash
# Build
docker build -t cellvision-qc .

# Generate demo data inside the container
docker run --rm -v $(pwd)/data:/workspace/data cellvision-qc \
    generate-demo --out data/demo

# Run segmentation
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/results:/workspace/results \
    cellvision-qc run \
    --images data/demo/images \
    --labels data/demo/labels.csv \
    --out results/demo \
    --method threshold
```

---

## Running Tests

```bash
pytest                       # all tests
pytest -v                    # verbose
pytest --cov=cellvision_qc   # with coverage
```

Tests use only synthetic fixtures; no external data or network access is required.

---

## Scientific Rationale

### Why does segmentation quality matter?

Every measurement in a high-content imaging experiment depends on the quality of the upstream segmentation. Object count, morphology features (area, eccentricity, solidity), and intensity measurements are all directly derived from the segmentation mask. A systematic bias in cell detection — for instance, missed cells, split objects, or merged cell clusters — will propagate as a systematic offset in every downstream measurement.

### Why compare threshold vs. watershed?

Global Otsu thresholding is the standard baseline: fast, interpretable, and parameter-light. It fails when cells are touching or overlapping, because connected foreground regions are treated as a single object. The distance-transform watershed addresses this by seeding individual objects from the interior of the distance map, allowing separation of touching cells.

This comparison is practically relevant because CellProfiler pipelines routinely use both approaches in different modules, and Cellpose was specifically designed to improve on classical methods for densely packed cell cultures.

### Classifier as a downstream proxy

Phenotype classification accuracy is used here as a proxy for how well-preserved biological signal is after a given segmentation step. If a classifier can distinguish healthy from unhealthy phenotypes with higher ROC-AUC using watershed features than threshold features, that suggests the watershed segmentation is capturing morphological and intensity information more faithfully on this image set.

This is a workflow demonstration, not a controlled biological experiment.

---

## Roadmap

- [ ] Cellpose runtime integration (optional dependency)
- [ ] CellProfiler CSV import with automatic column normalisation
- [ ] UMAP embedding visualisation of per-object features
- [ ] Batched processing for large image sets
- [ ] Jupyter Book documentation
- [ ] Snakemake workflow definition

---

## Disclaimer

CellVision-QC is a **workflow scaffold and demonstration tool**. It operates on synthetic data by default. The Cellpose-SAM and CellProfiler adapters are structural stubs that provide a consistent data contract for interoperability; they do not reproduce the algorithms or models of those tools, which are developed and maintained by their respective authors.

No biological validity is claimed for any result produced by this software on synthetic data. Users bringing real experimental data are responsible for validating results against appropriate biological controls.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

If you use CellVision-QC in your research, please cite:

```
Shenoy, A. (2025). CellVision-QC: Fluorescence microscopy image quality control
and segmentation comparison. GitHub. https://github.com/ashenoy/CellVision-QC
```
