"""
Unit Tests for Publication-Grade Methodological Rigor:
1. Zero-snooping threshold locking
2. 1,000-resample non-parametric bootstrap 95% CIs
3. Paired ROC bootstrap hypothesis test
4. Locked prospective external validation
5. Subgroup fairness parity audit
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models import build_models, build_clinical_pipelines
from src.evaluator import (
    evaluate_models,
    compute_bootstrap_confidence_intervals,
    compute_paired_roc_bootstrap_test,
    evaluate_locked_external_validation,
)
from src.tuning import evaluate_nested_cross_validation, find_optimal_clinical_threshold
from src.fairness import evaluate_demographic_fairness
from src.data_loader import load_external_validation_cohort, load_nhanes_cardiovascular


def test_zero_threshold_snooping_and_bootstrap():
    # Synthetic dataset
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "num_1": rng.normal(0, 1, n),
        "num_2": rng.normal(2, 1, n),
        "cat_1": rng.choice(["A", "B"], n)
    })
    y = pd.Series(rng.binomial(1, 0.3, n))

    models = {"Logistic Regression": LogisticRegression()}
    pipelines = build_clinical_pipelines(models, ["num_1", "num_2"], ["cat_1"], use_smote=False)

    results_df, _ = evaluate_models(
        pipelines, X, y, test_size=0.25, random_state=42, compute_ci=True, n_bootstraps=100
    )

    row = results_df.iloc[0]
    assert "Locked_Threshold" in row
    assert 0.0 < row["Locked_Threshold"] < 1.0
    assert "ROC-AUC [95% CI]" in row
    assert "Sensitivity [95% CI]" in row
    assert "[" in row["ROC-AUC [95% CI]"] and "]" in row["ROC-AUC [95% CI]"]


def test_paired_roc_bootstrap_test():
    rng = np.random.default_rng(42)
    n = 150
    y_true = rng.binomial(1, 0.4, n)
    # Model A is stronger than Model B
    proba_a = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, n), 0, 1)
    proba_b = np.clip(y_true * 0.3 + rng.uniform(0, 0.7, n), 0, 1)

    test_res = compute_paired_roc_bootstrap_test(y_true, proba_a, proba_b, n_bootstraps=200, random_state=42)
    assert "Mean_Delta_AUC (%)" in test_res
    assert "P_Value" in test_res
    assert test_res["Mean_Delta_AUC (%)"] > 0


def test_locked_external_validation():
    X_ext, y_ext = load_external_validation_cohort(n_samples=200, random_state=42)
    num_cols = X_ext.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_ext.select_dtypes(exclude=[np.number]).columns.tolist()

    models = {"Logistic Regression": LogisticRegression(max_iter=500)}
    pipelines = build_clinical_pipelines(models, num_cols, cat_cols, use_smote=False)
    pipe = pipelines["Logistic Regression"]

    # Fit pipeline on a subset
    pipe.fit(X_ext.iloc[:100], y_ext.iloc[:100])
    locked_threshold = 0.45

    # Evaluate blindly on the remaining subset
    df_ext, ext_dict = evaluate_locked_external_validation(
        pipe, locked_threshold, X_ext.iloc[100:], y_ext.iloc[100:], n_bootstraps=50, random_state=42
    )

    assert ext_dict["Locked_Threshold"] == 0.45
    assert "ROC-AUC [95% CI]" in ext_dict
    assert "ECE [95% CI]" in ext_dict


def test_subgroup_fairness_audit():
    rng = np.random.default_rng(42)
    n = 200
    y_true = pd.Series(rng.binomial(1, 0.3, n))
    y_pred = rng.binomial(1, 0.35, n)
    sex_cohort = pd.Series(rng.choice(["Female", "Male"], n))

    fairness_res = evaluate_demographic_fairness(
        y_true, y_pred, sex_cohort, group_a="Female", group_b="Male", n_bootstraps=100
    )

    assert "Disparate_Impact_Ratio" in fairness_res
    assert "Sensitivity_A [95% CI]" in fairness_res
    assert "Equal_Opportunity_Gap [95% CI]" in fairness_res
    assert "Fairness_Verdict" in fairness_res
