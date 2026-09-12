"""
Automated unit and integration test suite for Healthcare Diagnostics Benchmark.
"""

import pytest
import numpy as np
import pandas as pd
from src.models import build_models
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_models


def test_models_registry():
    models = build_models(fast_mode=True)
    assert len(models) == 10
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost" in models
    assert "SVM" in models


@pytest.mark.parametrize("dataset_key", list(DATASET_REGISTRY.keys()))
def test_data_loaders(dataset_key):
    meta = DATASET_REGISTRY[dataset_key]
    X, y = meta["loader"]()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 50  # Must contain real clinical samples
    assert not X.isnull().any().any(), f"NaNs found in features for {dataset_key}"
    assert set(y.unique()).issubset({0, 1}), f"Non-binary target in {dataset_key}"


def test_fast_benchmark_run():
    # Smoke test fast evaluation on breast cancer
    meta = DATASET_REGISTRY["breast_cancer"]
    X, y = meta["loader"]()
    pipelines = build_models(fast_mode=True)
    results_df, roc_data = evaluate_models(
        pipelines, X, y, test_size=0.2, random_state=42
    )

    assert len(results_df) == 10
    assert "Accuracy" in results_df.columns
    assert "F1-Score" in results_df.columns
    assert "ROC-AUC" in results_df.columns
    assert (results_df["Accuracy"] >= 80.0).any()
    assert len(roc_data) == 10
