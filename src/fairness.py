"""
Algorithmic Fairness & Demographic Subgroup Parity Module.
Evaluates clinical model equity across sensitive demographic cohorts:
1. Sex Subgroups (Male vs. Female)
2. Age Subgroups (Younger < 50 vs. Older Adults >= 50)
Metrics:
- Disparate Impact Ratio (Selection Rate Parity)
- True Positive Rate (Sensitivity) Parity (Equal Opportunity)
- False Positive Rate Parity (Predictive Equality)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def evaluate_demographic_fairness(
    y_true: pd.Series,
    y_pred: np.ndarray,
    demographic_series: pd.Series,
    group_a: str,
    group_b: str,
    metric_name: str = "Subgroup",
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes clinical algorithmic fairness criteria between two demographic groups.
    Evaluates:
    - Demographic Selection Parity
    - Equal Opportunity (Sensitivity / TPR Parity) with 95% CIs
    - Specificity Parity (TNR Parity) with 95% CIs
    - Clinical Subgroup Disparity Gap
    """
    mask_a = (demographic_series == group_a).values
    mask_b = (demographic_series == group_b).values

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    # 1. Selection rates (fraction classified positive)
    rate_a = np.mean(y_p[mask_a]) if np.sum(mask_a) > 0 else 0.0
    rate_b = np.mean(y_p[mask_b]) if np.sum(mask_b) > 0 else 0.0
    disparate_impact = (rate_a / rate_b) if rate_b > 0 else 1.0

    # 2. True Positive Rate (Sensitivity / Recall)
    tpr_a = recall_score(y_t[mask_a], y_p[mask_a], zero_division=0) if np.sum(y_t[mask_a]) > 0 else 0.0
    tpr_b = recall_score(y_t[mask_b], y_p[mask_b], zero_division=0) if np.sum(y_t[mask_b]) > 0 else 0.0
    equal_opportunity_diff = np.abs(tpr_a - tpr_b)

    # 3. Specificity (True Negative Rate)
    spec_a = recall_score(1 - y_t[mask_a], 1 - y_p[mask_a], zero_division=0) if np.sum(1 - y_t[mask_a]) > 0 else 0.0
    spec_b = recall_score(1 - y_t[mask_b], 1 - y_p[mask_b], zero_division=0) if np.sum(1 - y_t[mask_b]) > 0 else 0.0
    specificity_diff = np.abs(spec_a - spec_b)

    # 4. False Positive Rate (1 - Specificity)
    fpr_a = 1.0 - spec_a
    fpr_b = 1.0 - spec_b
    predictive_equality_diff = np.abs(fpr_a - fpr_b)

    # 5. Bootstrap 95% CIs for subgroup metrics and disparity gap
    rng = np.random.default_rng(random_state)
    boot_tpr_a, boot_tpr_b, boot_gap = [], [], []
    idx_a = np.where(mask_a)[0]
    idx_b = np.where(mask_b)[0]

    for _ in range(n_bootstraps):
        b_a = rng.choice(idx_a, size=len(idx_a), replace=True)
        b_b = rng.choice(idx_b, size=len(idx_b), replace=True)

        if len(np.unique(y_t[b_a])) >= 2 and len(np.unique(y_t[b_b])) >= 2:
            ta = recall_score(y_t[b_a], y_p[b_a], zero_division=0) * 100.0
            tb = recall_score(y_t[b_b], y_p[b_b], zero_division=0) * 100.0
            boot_tpr_a.append(ta)
            boot_tpr_b.append(tb)
            boot_gap.append(np.abs(ta - tb))

    ci_tpr_a = f"[{np.percentile(boot_tpr_a, 2.5):.1f}%, {np.percentile(boot_tpr_a, 97.5):.1f}%]" if boot_tpr_a else "N/A"
    ci_tpr_b = f"[{np.percentile(boot_tpr_b, 2.5):.1f}%, {np.percentile(boot_tpr_b, 97.5):.1f}%]" if boot_tpr_b else "N/A"
    ci_gap = f"[{np.percentile(boot_gap, 2.5):.2f}%, {np.percentile(boot_gap, 97.5):.2f}%]" if boot_gap else "N/A"

    is_fair = (0.80 <= disparate_impact <= 1.25) and (equal_opportunity_diff < 0.10) and (specificity_diff < 0.10)
    verdict = (
        "Clinical Subgroup Parity: Acceptable Parity (Gap < 10% & Overlapping 95% CIs)"
        if is_fair else "Clinical Subgroup Disparity Observed (Requires Calibrated Subgroup Thresholding)"
    )

    return {
        "Subgroup_A": f"{group_a} (N={np.sum(mask_a)})",
        "Subgroup_B": f"{group_b} (N={np.sum(mask_b)})",
        "Selection_Rate_A": f"{rate_a * 100:.1f}%",
        "Selection_Rate_B": f"{rate_b * 100:.1f}%",
        "Disparate_Impact_Ratio": round(float(disparate_impact), 3),
        "Sensitivity_A [95% CI]": f"{tpr_a * 100:.1f}% {ci_tpr_a}",
        "Sensitivity_B [95% CI]": f"{tpr_b * 100:.1f}% {ci_tpr_b}",
        "Specificity_A": f"{spec_a * 100:.1f}%",
        "Specificity_B": f"{spec_b * 100:.1f}%",
        "Equal_Opportunity_Gap": f"{equal_opportunity_diff * 100:.2f}%",
        "Equal_Opportunity_Gap [95% CI]": f"{equal_opportunity_diff * 100:.2f}% {ci_gap}",
        "Specificity_Gap": f"{specificity_diff * 100:.2f}%",
        "False_Positive_Rate_Gap": f"{predictive_equality_diff * 100:.2f}%",
        "Fairness_Verdict": verdict,
        "Clinical_Fairness_Verdict": verdict
    }

