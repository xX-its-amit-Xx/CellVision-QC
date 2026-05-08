"""Tests for cellvision_qc.phenotype.analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellvision_qc.phenotype.analysis import (
    ClassificationReport,
    PhenotypeClassifier,
)


def _make_synthetic_features(n_healthy: int = 60, n_unhealthy: int = 60, seed: int = 0) -> pd.DataFrame:
    """Create a small synthetic feature table with separable phenotypes."""
    rng = np.random.default_rng(seed)

    healthy = pd.DataFrame(
        {
            "area": rng.normal(700, 50, n_healthy),
            "perimeter": rng.normal(100, 8, n_healthy),
            "eccentricity": rng.uniform(0.0, 0.3, n_healthy),
            "solidity": rng.uniform(0.85, 1.0, n_healthy),
            "mean_intensity": rng.uniform(0.55, 0.85, n_healthy),
            "integrated_intensity": rng.uniform(400, 600, n_healthy),
            "phenotype": "healthy",
            "method": "threshold",
        }
    )
    unhealthy = pd.DataFrame(
        {
            "area": rng.normal(400, 100, n_unhealthy),
            "perimeter": rng.normal(140, 20, n_unhealthy),
            "eccentricity": rng.uniform(0.5, 0.9, n_unhealthy),
            "solidity": rng.uniform(0.5, 0.75, n_unhealthy),
            "mean_intensity": rng.uniform(0.2, 0.45, n_unhealthy),
            "integrated_intensity": rng.uniform(100, 250, n_unhealthy),
            "phenotype": "unhealthy",
            "method": "threshold",
        }
    )
    return pd.concat([healthy, unhealthy], ignore_index=True)


class TestPhenotypeClassifier:
    def test_random_forest_returns_report(self) -> None:
        df = _make_synthetic_features()
        clf = PhenotypeClassifier(classifier_type="random_forest", cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        assert isinstance(report, ClassificationReport)

    def test_logistic_returns_report(self) -> None:
        df = _make_synthetic_features()
        clf = PhenotypeClassifier(classifier_type="logistic", cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        assert isinstance(report, ClassificationReport)

    def test_accuracy_above_chance(self) -> None:
        df = _make_synthetic_features(n_healthy=80, n_unhealthy=80)
        clf = PhenotypeClassifier(classifier_type="random_forest", cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        assert report.accuracy_mean > 0.6, f"Expected >0.6 accuracy, got {report.accuracy_mean:.3f}"

    def test_roc_auc_in_range(self) -> None:
        df = _make_synthetic_features()
        clf = PhenotypeClassifier(classifier_type="random_forest", cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        assert 0.0 <= report.roc_auc_mean <= 1.0

    def test_report_dict_keys(self) -> None:
        df = _make_synthetic_features()
        clf = PhenotypeClassifier(cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        d = report.to_dict()
        for key in ["method", "classifier", "n_objects", "accuracy_mean", "f1_mean", "roc_auc_mean"]:
            assert key in d

    def test_n_objects_matches_input(self) -> None:
        df = _make_synthetic_features(n_healthy=30, n_unhealthy=30)
        clf = PhenotypeClassifier(cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype")
        assert report.n_objects == 60

    def test_invalid_classifier_raises(self) -> None:
        with pytest.raises(ValueError, match="classifier_type"):
            PhenotypeClassifier(classifier_type="svm")

    def test_method_name_override(self) -> None:
        df = _make_synthetic_features()
        clf = PhenotypeClassifier(cv_folds=3)
        report = clf.evaluate(df, label_col="phenotype", method_name="my_method")
        assert report.method == "my_method"
