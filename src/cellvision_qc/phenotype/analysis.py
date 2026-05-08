"""Phenotype classification and segmentation-sensitivity analysis.

This module demonstrates how segmentation method choice propagates into
downstream phenotype discrimination. All analysis is performed on
synthetic demonstration data and should be interpreted as a workflow
scaffold rather than a validated biological assay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLS = [
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "integrated_intensity",
]


@dataclass
class ClassificationReport:
    """Classification performance summary for a single method.

    Attributes
    ----------
    method:
        Segmentation method name.
    classifier:
        Classifier type used.
    n_objects:
        Total number of objects (cells) used for classification.
    n_features:
        Number of features used.
    cv_folds:
        Number of cross-validation folds.
    accuracy_mean:
        Mean cross-validated accuracy.
    accuracy_std:
        Std of cross-validated accuracy.
    f1_mean:
        Mean cross-validated macro F1 score.
    f1_std:
        Std of cross-validated F1 score.
    roc_auc_mean:
        Mean cross-validated ROC-AUC (binary only; NaN for multi-class).
    roc_auc_std:
        Std of cross-validated ROC-AUC.
    """

    method: str
    classifier: str
    n_objects: int
    n_features: int
    cv_folds: int
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    roc_auc_mean: float
    roc_auc_std: float

    def to_dict(self) -> dict:
        return asdict(self)


class PhenotypeClassifier:
    """Train and evaluate a phenotype classifier on per-object features.

    Parameters
    ----------
    classifier_type:
        ``"logistic"`` or ``"random_forest"``.
    cv_folds:
        Number of stratified cross-validation folds.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        classifier_type: str = "random_forest",
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        if classifier_type not in ("logistic", "random_forest"):
            raise ValueError(
                f"classifier_type must be 'logistic' or 'random_forest', "
                f"got '{classifier_type}'"
            )
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _build_pipeline(self) -> Pipeline:
        if self.classifier_type == "logistic":
            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    def evaluate(
        self,
        features_df: pd.DataFrame,
        label_col: str = "phenotype",
        method_name: Optional[str] = None,
    ) -> ClassificationReport:
        """Run stratified k-fold cross-validation and return a report.

        Parameters
        ----------
        features_df:
            Per-object feature table. Must contain FEATURE_COLS and label_col.
        label_col:
            Column containing phenotype labels.
        method_name:
            Override the method name in the report; defaults to the unique
            value of ``features_df["method"]`` if present.
        """
        available = [c for c in FEATURE_COLS if c in features_df.columns]
        X = features_df[available].values
        y_raw = features_df[label_col].values

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        is_binary = n_classes == 2

        pipeline = self._build_pipeline()
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        scoring = ["accuracy", "f1_macro"]
        if is_binary:
            scoring.append("roc_auc")

        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
        )

        if method_name is None:
            method_name = (
                str(features_df["method"].iloc[0])
                if "method" in features_df.columns
                else "unknown"
            )

        roc_mean = float(np.mean(cv_results.get("test_roc_auc", [np.nan])))
        roc_std = float(np.std(cv_results.get("test_roc_auc", [np.nan])))

        return ClassificationReport(
            method=method_name,
            classifier=self.classifier_type,
            n_objects=len(X),
            n_features=len(available),
            cv_folds=self.cv_folds,
            accuracy_mean=float(np.mean(cv_results["test_accuracy"])),
            accuracy_std=float(np.std(cv_results["test_accuracy"])),
            f1_mean=float(np.mean(cv_results["test_f1_macro"])),
            f1_std=float(np.std(cv_results["test_f1_macro"])),
            roc_auc_mean=roc_mean,
            roc_auc_std=roc_std,
        )


def compare_phenotypes(
    run_dirs: list[Path],
    output_dir: Path,
    label_col: str = "phenotype",
    classifier_type: str = "random_forest",
) -> pd.DataFrame:
    """Compare classification performance across multiple segmentation runs.

    Reads ``features.csv`` from each run directory, evaluates a classifier
    for each, and writes a comparison CSV + JSON to ``output_dir``.

    Parameters
    ----------
    run_dirs:
        Directories produced by ``cellvision-qc run``. Each must contain
        ``features.csv``.
    output_dir:
        Destination for comparison outputs.
    label_col:
        Column in features CSV containing phenotype labels.
    classifier_type:
        Classifier to use (``"logistic"`` or ``"random_forest"``).

    Returns
    -------
    pd.DataFrame
        One row per run with classification metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clf = PhenotypeClassifier(classifier_type=classifier_type)
    reports = []

    for run_dir in run_dirs:
        csv_path = run_dir / "features.csv"
        if not csv_path.exists():
            print(f"[warn] No features.csv in {run_dir}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            print(f"[warn] Column '{label_col}' not found in {csv_path}, skipping.")
            continue

        method_name = run_dir.name
        report = clf.evaluate(df, label_col=label_col, method_name=method_name)
        reports.append(report)

    if not reports:
        return pd.DataFrame()

    summary = pd.DataFrame([r.to_dict() for r in reports])
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    report_json = [r.to_dict() for r in reports]
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report_json, indent=2)
    )

    return summary
