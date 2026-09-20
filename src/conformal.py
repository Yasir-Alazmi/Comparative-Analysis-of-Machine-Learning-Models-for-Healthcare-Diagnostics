"""
Conformal Prediction & Clinical Probability Calibration Module.
Provides mathematically guaranteed uncertainty quantification for high-stakes medical diagnosis:
1. Inductive Conformal Prediction (ICP) producing prediction sets with guaranteed (1 - alpha) coverage
2. Ambiguity Rate and Referral Quantification (identifying ambiguous clinical cases)
3. Expected Calibration Error (ECE) and Brier Score Loss
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Measures reliability of model risk probabilities against true observed disease frequencies.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    total_samples = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        bin_counts[b] = count
        if count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_accuracies[b] = bin_acc
            bin_confidences[b] = bin_conf
            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4)), bin_accuracies, bin_confidences


def evaluate_conformal_prediction(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    alpha: float = 0.05,
    calib_size: float = 0.25,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes Split (Inductive) Conformal Prediction:
    Guarantees finite-sample marginal statistical coverage: P(Y in C(X)) >= 1 - alpha
    strictly under the exchangeability (i.i.d.) hypothesis.
    
    IMPORTANT CLINICAL NOTE:
    This guarantees marginal statistical coverage over hypothetical exchangeable draws;
    it is NOT an absolute guarantee of patient-specific diagnostic certainty. Ambiguous sets {0, 1}
    indicate high uncertainty and mandate human physician triage / referral.

    Returns:
        Coverage percentage with 95% CIs, mean set size, ambiguity referral rate, and ECE.
    """
    # 1. Split training into pure training and calibration splits
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=calib_size, stratify=y_train, random_state=random_state
    )

    # 2. Fit pipeline on pure training data
    pipeline.fit(X_tr, y_tr)

    # 3. Predict probabilities on calibration set
    cal_proba = pipeline.predict_proba(X_cal)
    y_cal_arr = np.asarray(y_cal)

    # Non-conformity scores: s_i = 1 - p(y_true | x_i)
    cal_scores = 1.0 - cal_proba[np.arange(len(y_cal_arr)), y_cal_arr]

    # 4. Calculate finite-sample conformal quantile threshold q_hat
    n_cal = len(cal_scores)
    level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    level = min(1.0, max(0.0, level))
    q_hat = np.quantile(cal_scores, level, method="higher")

    # 5. Evaluate on independent test set
    test_proba = pipeline.predict_proba(X_test)
    y_test_arr = np.asarray(y_test)

    # For each test sample, include class c if (1 - p_c) <= q_hat
    # Equivalently: include c if p_c >= 1 - q_hat
    threshold_prob = 1.0 - q_hat

    set_includes_0 = test_proba[:, 0] >= threshold_prob
    set_includes_1 = test_proba[:, 1] >= threshold_prob

    # Compute prediction sets
    set_sizes = set_includes_0.astype(int) + set_includes_1.astype(int)
    # Ensure no empty sets by including the argmax if empty
    empty_mask = set_sizes == 0
    if np.any(empty_mask):
        top_classes = np.argmax(test_proba[empty_mask], axis=1)
        set_includes_0[empty_mask] = top_classes == 0
        set_includes_1[empty_mask] = top_classes == 1
        set_sizes = set_includes_0.astype(int) + set_includes_1.astype(int)

    # Coverage: true class is inside prediction set
    covered = np.where(y_test_arr == 0, set_includes_0, set_includes_1)
    empirical_coverage = np.mean(covered) * 100.0

    # Non-parametric bootstrap for coverage 95% CI
    boot_rng = np.random.default_rng(random_state)
    boot_covs = []
    n_t = len(covered)
    for _ in range(n_bootstraps):
        b_idx = boot_rng.choice(n_t, size=n_t, replace=True)
        boot_covs.append(np.mean(covered[b_idx]) * 100.0)

    cov_ci = f"[{np.percentile(boot_covs, 2.5):.2f}%, {np.percentile(boot_covs, 97.5):.2f}%]"

    # Ambiguity (both classes included {0, 1} -> requires human clinician referral)
    ambiguous = set_sizes == 2
    ambiguity_rate = np.mean(ambiguous) * 100.0

    # Singlet sets ({0} or {1})
    singleton_rate = np.mean(set_sizes == 1) * 100.0

    # Calibration error on test set
    ece, _, _ = compute_expected_calibration_error(y_test_arr, test_proba[:, 1], n_bins=10)

    return {
        "Target_Coverage": f"{(1 - alpha) * 100:.1f}%",
        "Coverage_Guarantee_Type": "Marginal Statistical Coverage under Exchangeability",
        "Empirical_Coverage": round(empirical_coverage, 2),
        "Empirical_Coverage [95% CI]": cov_ci,
        "Conformal_Threshold": round(float(q_hat), 4),
        "Mean_Set_Size": round(float(np.mean(set_sizes)), 3),
        "Singleton_Certainty_Rate": round(singleton_rate, 2),
        "Ambiguity_Referral_Rate": round(ambiguity_rate, 2),
        "Expected_Calibration_Error": round(ece, 4),
    }
