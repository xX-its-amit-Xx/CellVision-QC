"""Tests for cellvision_qc.data.synthetic."""

from __future__ import annotations

import pandas as pd
import pytest

from cellvision_qc.data.synthetic import generate_demo_dataset


class TestGenerateDemoDataset:
    def test_creates_expected_files(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "demo"
        generate_demo_dataset(out, n_images=4, image_size=64, cells_per_image=5)
        assert (out / "labels.csv").exists()
        assert (out / "metadata.json").exists()
        assert (out / "cell_ground_truth.csv").exists()

    def test_correct_image_count(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "demo"
        generate_demo_dataset(out, n_images=6, image_size=64, cells_per_image=5)
        images = list((out / "images").glob("*.tif"))
        assert len(images) == 6

    def test_labels_csv_has_correct_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "demo"
        generate_demo_dataset(out, n_images=4, image_size=64, cells_per_image=5)
        df = pd.read_csv(out / "labels.csv")
        assert "filename" in df.columns
        assert "phenotype" in df.columns

    def test_balanced_phenotypes(self, tmp_path: pytest.TempPathFactory) -> None:
        out = tmp_path / "demo"
        generate_demo_dataset(
            out, n_images=10, image_size=64, cells_per_image=5, healthy_fraction=0.5
        )
        df = pd.read_csv(out / "labels.csv")
        counts = df["phenotype"].value_counts()
        assert counts["healthy"] == 5
        assert counts["unhealthy"] == 5

    def test_reproducible_with_same_seed(self, tmp_path: pytest.TempPathFactory) -> None:
        out1 = tmp_path / "d1"
        out2 = tmp_path / "d2"
        generate_demo_dataset(out1, n_images=3, image_size=32, cells_per_image=3, seed=7)
        generate_demo_dataset(out2, n_images=3, image_size=32, cells_per_image=3, seed=7)
        df1 = pd.read_csv(out1 / "labels.csv")
        df2 = pd.read_csv(out2 / "labels.csv")
        assert list(df1["phenotype"]) == list(df2["phenotype"])
