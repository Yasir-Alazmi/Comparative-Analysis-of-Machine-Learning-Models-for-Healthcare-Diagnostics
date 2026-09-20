"""
Unit Tests for Conformal Prediction and Uncertainty Calibration.
"""

import pytest
import numpy as np
import pandas as pd
from src.conformal import compute_expected_calibration_error, evaluate_conformal_prediction
from src.preprocessor import build_leakage_free_pipeline
from sklearn.linear_model import LogisticRegression


def test_ece_computation():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95, 0.9, 0.88, 0.92])
    ece, _, _ = compute_expected_calibration_error(y_true, y_proba, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.20  # Well-calibrated toy probabilities


def test_conformal_coverage_guarantee():
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame({"feat": rng.normal(0, 1, size=n)})
    # Logistic relationship
    p = 1.0 / (1.0 + np.exp(-1.5 * X["feat"]))
    y = pd.Series(rng.binomial(1, p))

    pipe = build_leakage_free_pipeline(LogisticRegression(), num_cols=["feat"], cat_cols=[])
    
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    res = evaluate_conformal_prediction(pipe, X_tr, y_tr, X_te, y_te, alpha=0.05, random_state=42)
    
    # Assert empirical coverage satisfies guaranteed level within small finite-sample margin
    assert res["Empirical_Coverage"] >= 92.0
    assert res["Mean_Set_Size"] >= 1.0
    assert 0.0 <= res["Expected_Calibration_Error"] <= 1.0
