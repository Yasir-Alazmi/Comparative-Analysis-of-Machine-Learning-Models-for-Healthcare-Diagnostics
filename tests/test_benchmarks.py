import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from src.models import build_models, build_clinical_pipelines
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_models, evaluate_cross_validation
from src.robustness import evaluate_sensor_noise_drift


def test_models_registry():
    models = build_models(fast_mode=True, include_stacking=True)
    assert len(models) >= 10
    assert "CatBoost" in models
    assert "LightGBM" in models
    assert "XGBoost" in models
    assert "Random Forest" in models
    assert "Logistic Regression" in models
    assert "Stacking Ensemble (Super Learner)" in models


@pytest.mark.parametrize("dataset_key", list(DATASET_REGISTRY.keys()))
def test_data_loaders(dataset_key):
    meta = DATASET_REGISTRY[dataset_key]
    X, y = meta["loader"]()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) >= 50
    assert set(y.dropna().unique()).issubset({0, 1}), f"Non-binary target in {dataset_key}"


def test_nhanes_cardiovascular_loader():
    meta = DATASET_REGISTRY["nhanes_cardiovascular"]
    X, y = meta["loader"]()
    assert len(X) >= 5000
    # Assert engineered features exist
    assert "AIP" in X.columns
    assert "MAP" in X.columns
    assert "Pulse_Pressure" in X.columns
    assert "TyG_Index" in X.columns
    assert "Total_HDL_Ratio" in X.columns


def test_fast_benchmark_run():
    meta = DATASET_REGISTRY["heart_failure"]
    X, y = meta["loader"]()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    models = {"Logistic Regression": build_models(fast_mode=True)["Logistic Regression"]}
    pipelines = build_clinical_pipelines(models, num_cols, cat_cols, use_smote=False)

    results_df, _ = evaluate_models(pipelines, X, y, test_size=0.2, random_state=42)
    assert len(results_df) == 1
    assert "ROC-AUC" in results_df.columns
    assert "Sensitivity (Recall)" in results_df.columns
    assert "ECE" in results_df.columns
