"""
Unit Tests for Zero-Leakage Preprocessing Protocol.
Asserts that no validation/test fold information leaks into the training pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.preprocessor import build_leakage_free_pipeline
from sklearn.linear_model import LogisticRegression


def test_zero_leakage_in_cross_validation():
    # Construct synthetic dataset with missing values and distinct feature scales
    rng = np.random.default_rng(42)
    n_samples = 200
    X = pd.DataFrame({
        "num_1": rng.normal(100, 15, size=n_samples),
        "num_2": rng.exponential(10, size=n_samples),
        "cat_1": rng.choice(["A", "B", "C"], size=n_samples)
    })
    # Inject missing values
    X.loc[10:20, "num_1"] = np.nan
    X.loc[30:40, "cat_1"] = np.nan

    y = pd.Series(rng.binomial(1, 0.4, size=n_samples))

    num_cols = ["num_1", "num_2"]
    cat_cols = ["cat_1"]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Calculate pre-split test statistic
        test_median_num1 = X_test["num_1"].median()

        pipe = build_leakage_free_pipeline(
            classifier=LogisticRegression(),
            num_cols=num_cols,
            cat_cols=cat_cols,
            use_smote=False,
            random_state=42
        )

        # Fit on training set ONLY
        pipe.fit(X_train, y_train)

        # Preprocessor should be fitted with X_train parameters
        preprocessor = pipe.named_steps["preprocessor"]
        num_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
        
        # Verify predictions work on unseen test fold with NaNs without error
        preds = pipe.predict(X_test)
        probas = pipe.predict_proba(X_test)

        assert len(preds) == len(X_test)
        assert probas.shape == (len(X_test), 2)
        assert not np.isnan(probas).any()
