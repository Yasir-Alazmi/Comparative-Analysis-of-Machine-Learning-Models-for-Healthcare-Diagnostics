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
    metric_name: str = "Subgroup"
) -> Dict[str, Any]:
    """
    Computes algorithmic fairness criteria between two demographic groups.
    """
    mask_a = (demographic_series == group_a).values
    mask_b = (demographic_series == group_b).values

    y_t = np.asarray(y_true)

    # 1. Selection rates (fraction classified positive)
    rate_a = np.mean(y_pred[mask_a]) if np.sum(mask_a) > 0 else 0.0
    rate_b = np.mean(y_pred[mask_b]) if np.sum(mask_b) > 0 else 0.0
    disparate_impact = (rate_a / rate_b) if rate_b > 0 else 1.0

    # 2. True Positive Rate (Sensitivity / Recall)
    tpr_a = recall_score(y_t[mask_a], y_pred[mask_a], zero_division=0) if np.sum(y_t[mask_a]) > 0 else 0.0
    tpr_b = recall_score(y_t[mask_b], y_pred[mask_b], zero_division=0) if np.sum(y_t[mask_b]) > 0 else 0.0
    equal_opportunity_diff = np.abs(tpr_a - tpr_b)

    # 3. False Positive Rate (1 - Specificity)
    neg_a = (y_t[mask_a] == 0)
    neg_b = (y_t[mask_b] == 0)
    fpr_a = np.mean(y_pred[mask_a][neg_a]) if np.sum(neg_a) > 0 else 0.0
    fpr_b = np.mean(y_pred[mask_b][neg_b]) if np.sum(neg_b) > 0 else 0.0
    predictive_equality_diff = np.abs(fpr_a - fpr_b)

    fairness_verdict = "FAIR (Meets 80% Rule)" if 0.80 <= disparate_impact <= 1.25 and equal_opportunity_diff < 0.10 else "DISPARITY OBSERVED"

    return {
        "Subgroup_A": f"{group_a} (N={np.sum(mask_a)})",
        "Subgroup_B": f"{group_b} (N={np.sum(mask_b)})",
        "Selection_Rate_A": f"{rate_a * 100:.1f}%",
        "Selection_Rate_B": f"{rate_b * 100:.1f}%",
        "Disparate_Impact_Ratio": round(float(disparate_impact), 3),
        "Sensitivity_A (TPR)": f"{tpr_a * 100:.1f}%",
        "Sensitivity_B (TPR)": f"{tpr_b * 100:.1f}%",
        "Equal_Opportunity_Gap": f"{equal_opportunity_diff * 100:.2f}%",
        "False_Positive_Rate_Gap": f"{predictive_equality_diff * 100:.2f}%",
        "Fairness_Verdict": fairness_verdict
    }
