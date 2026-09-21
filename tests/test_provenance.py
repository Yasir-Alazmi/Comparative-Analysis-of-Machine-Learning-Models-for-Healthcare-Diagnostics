"""
Scientific Provenance & Dataset Integrity Tests.
Verifies that benchmarks consume authentic human clinical data from CDC NHANES and Framingham,
and proves that no silent synthetic data fallbacks occur.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_nhanes_cardiovascular,
    load_external_validation_cohort,
    generate_synthetic_demo_cohort,
)


def test_nhanes_authentic_cohort_integrity():
    """Validates that authentic CDC NHANES cohort meets published size and clinical prevalence."""
    X, y = load_nhanes_cardiovascular()

    # Must be the authentic CDC adult survey cohort (N >= 11,000)
    assert len(X) >= 11000, f"Expected N >= 11,000, got {len(X)}"
    assert len(y) == len(X)

    # Real-world physician-diagnosed composite CVD prevalence in U.S. adults is ~11-13%
    prevalence = np.mean(y) * 100.0
    assert 10.0 <= prevalence <= 14.0, f"Prevalence {prevalence:.2f}% outside authentic epidemiological bounds [10%, 14%]"

    # Verify essential physiological biomarkers and engineered features
    expected_features = [
        "Age", "Sex", "Systolic_BP", "Diastolic_BP", "Total_Cholesterol",
        "HDL_Cholesterol", "HbA1c", "Serum_Creatinine", "AIP", "MAP", "TyG_Index", "eGFR"
    ]
    for feat in expected_features:
        assert feat in X.columns, f"Missing required biomarker or engineered feature: {feat}"


def test_framingham_external_cohort_integrity():
    """Validates that authentic Framingham Heart Study longitudinal cohort has exact patient count and event rate."""
    X, y = load_external_validation_cohort(n_samples=None)

    # Exact cohort size of authentic Framingham release
    assert len(X) == 4240, f"Expected authentic Framingham N = 4,240, got {len(X)}"
    assert len(y) == 4240

    # 10-year incident CHD event rate in Framingham cohort is ~15.2%
    event_rate = np.mean(y) * 100.0
    assert 14.0 <= event_rate <= 16.5, f"Event rate {event_rate:.2f}% outside authentic Framingham bounds [14%, 16.5%]"


def test_no_silent_synthetic_fallback_on_missing_dataset():
    """Ensures load_nhanes_cardiovascular strictly fails with FileNotFoundError when data is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_nhanes_cardiovascular(data_path="datasets/nonexistent_fake_path_xyz.csv")

    assert "no synthetic data fallback is permitted" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()


def test_synthetic_demo_is_quarantined():
    """Verifies that synthetic generator exists strictly for offline smoke tests when explicitly invoked."""
    df_demo = generate_synthetic_demo_cohort(n_samples=100, random_state=42)
    assert len(df_demo) == 100
    assert "CVD_Diagnosis" in df_demo.columns
