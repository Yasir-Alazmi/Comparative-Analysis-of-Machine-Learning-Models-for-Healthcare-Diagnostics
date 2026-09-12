import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from src.models import build_models
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_models, evaluate_cross_validation
from src.robustness import evaluate_seed_stability, evaluate_noise_resilience


def test_models_registry():
    models = build_models(fast_mode=True)
    assert len(models) == 10
    assert 'Logistic Regression' in models
    assert 'Random Forest' in models
    assert 'XGBoost' in models
    assert 'SVM' in models


@pytest.mark.parametrize('dataset_key', list(DATASET_REGISTRY.keys()))
def test_data_loaders(dataset_key):
    meta = DATASET_REGISTRY[dataset_key]
    X, y = meta['loader']()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 50
    assert not X.isnull().any().any(), f'NaNs found in features for {dataset_key}'
    assert set(y.unique()).issubset({0, 1}), f'Non-binary target in {dataset_key}'


def test_fast_benchmark_run():
    meta = DATASET_REGISTRY['breast_cancer']
    X, y = meta['loader']()
    pipelines = build_models(fast_mode=True)
    results_df, roc_data = evaluate_models(
        pipelines, X, y, test_size=0.2, random_state=42
    )

    assert len(results_df) == 10
    assert 'Accuracy' in results_df.columns
    assert 'Balanced Acc' in results_df.columns
    assert 'MCC' in results_df.columns
    assert 'Brier Score' in results_df.columns
    assert 'F1-Score' in results_df.columns
    assert 'ROC-AUC' in results_df.columns
    assert len(roc_data) == 10


def test_cross_validation():
    meta = DATASET_REGISTRY['breast_cancer']
    X, y = meta['loader']()
    pipelines = {'Logistic Regression': build_models(fast_mode=True)['Logistic Regression']}
    df_cv = evaluate_cross_validation(pipelines, X, y, n_splits=3, random_state=42)
    assert len(df_cv) == 1
    assert 'Accuracy (Mean±Std)' in df_cv.columns
    assert 'Balanced Acc (Mean±Std)' in df_cv.columns


def test_noise_and_seed_robustness():
    meta = DATASET_REGISTRY['breast_cancer']
    X, y = meta['loader']()
    pipelines = {'Logistic Regression': build_models(fast_mode=True)['Logistic Regression']}
    df_stab = evaluate_seed_stability(pipelines, X, y, seeds=[42, 99])
    assert len(df_stab) == 1
    assert df_stab.loc['Logistic Regression', 'Mean Accuracy'] > 90.0

    df_noise = evaluate_noise_resilience(pipelines, X, y, noise_levels=[0.0, 0.05])
    assert len(df_noise) == 1
    assert df_noise.loc['Logistic Regression', 'Noise 0%'] > 90.0
