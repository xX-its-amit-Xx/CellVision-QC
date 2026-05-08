#!/usr/bin/env bash
# run_demo.sh — End-to-end CellVision-QC demonstration using synthetic data.
#
# Prerequisites:
#   pip install -e ".[dev]"   (from the repository root)
#
# Usage:
#   bash examples/run_demo.sh
#
# Outputs (all under results/):
#   demo/          — threshold segmentation results
#   demo_watershed/— watershed segmentation results
#   comparison/    — cross-method performance comparison

set -euo pipefail

DATA_DIR="data/demo"
RESULTS_DIR="results"

echo "============================================================"
echo "  CellVision-QC  |  synthetic end-to-end demonstration"
echo "============================================================"
echo ""

# ── Step 1: generate demo data ──────────────────────────────────
echo "[1/4] Generating synthetic demo dataset..."
cellvision-qc generate-demo \
    --out "${DATA_DIR}" \
    --n-images 16 \
    --image-size 256 \
    --cells-per-image 25 \
    --seed 42

echo ""

# ── Step 2: run threshold segmentation ──────────────────────────
echo "[2/4] Running threshold segmentation..."
cellvision-qc run \
    --images "${DATA_DIR}/images" \
    --labels "${DATA_DIR}/labels.csv" \
    --out "${RESULTS_DIR}/demo" \
    --method threshold \
    --background-radius 50 \
    --min-object-size 50

echo ""

# ── Step 3: run watershed segmentation ──────────────────────────
echo "[3/4] Running watershed segmentation..."
cellvision-qc run \
    --images "${DATA_DIR}/images" \
    --labels "${DATA_DIR}/labels.csv" \
    --out "${RESULTS_DIR}/demo_watershed" \
    --method watershed \
    --background-radius 50 \
    --min-object-size 50

echo ""

# ── Step 4: compare methods ──────────────────────────────────────
echo "[4/4] Comparing segmentation methods..."
cellvision-qc compare \
    "${RESULTS_DIR}/demo" \
    "${RESULTS_DIR}/demo_watershed" \
    --out "${RESULTS_DIR}/comparison" \
    --classifier random_forest

echo ""
echo "============================================================"
echo "  Done! Key outputs:"
echo "    ${RESULTS_DIR}/demo/qc_metrics.csv"
echo "    ${RESULTS_DIR}/demo/features.csv"
echo "    ${RESULTS_DIR}/demo/phenotype_report.json"
echo "    ${RESULTS_DIR}/demo_watershed/phenotype_report.json"
echo "    ${RESULTS_DIR}/comparison/comparison_summary.csv"
echo "    ${RESULTS_DIR}/comparison/performance_comparison.png"
echo "    ${RESULTS_DIR}/comparison/qc_radar.png"
echo "============================================================"
