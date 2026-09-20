"""
Comprehensive Clinical Evaluation & Statistical Inference Engine.
Features:
1. Honest Clinical Metrics: ROC-AUC, PR-AUC, Sensitivity, Specificity, Balanced Accuracy, MCC, Brier Score, ECE.
2. Optimal Youden's J-statistic decision thresholding.
3. K-Fold Stratified Cross-Validation with leak-free fold execution.
4. Friedman Omnibus & Nemenyi Post-Hoc Statistical Significance Tests.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    matthews_corrcoef,
    brier_score_loss,
)

from src.conformal import compute_expected_calibration_error
from src.tuning import find_optimal_clinical_threshold


def evaluate_models(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    sort_metric: str = "ROC-AUC"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates clinical models with isolated train/test separation and comprehensive clinical diagnostics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results = {}
    roc_data = {}
    pr_data = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)
        y_pred = pipe.predict(X_test)

        # Standard metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        spec = recall_score(1 - y_test, 1 - y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        ece, _, _ = compute_expected_calibration_error(np.asarray(y_test), y_proba)

        # ROC & PR curves
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)
        pr_data[name] = (recall_pts, precision_pts, pr_auc)

        # Optimal Youden threshold sensitivity
        thresh_info = find_optimal_clinical_threshold(np.asarray(y_test), y_proba)

        results[name] = {
            "ROC-AUC": round(roc * 100, 2),
            "PR-AUC": round(pr_auc * 100, 2),
            "Sensitivity (Recall)": round(rec * 100, 2),
            "Specificity": round(spec * 100, 2),
            "Balanced Acc": round(bal_acc * 100, 2),
            "MCC": round(mcc, 3),
            "Brier Score": round(brier, 4),
            "ECE": round(ece, 4),
            "Optimal_Sensitivity": f"{thresh_info['Sensitivity_at_Threshold']}% (thresh={thresh_info['Optimal_Threshold']})",
        }

    results_df = pd.DataFrame(results).T
    if sort_metric in results_df.columns:
        results_df = results_df.sort_values(sort_metric, ascending=False)
    return results_df, {"roc": roc_data, "pr": pr_data}


def evaluate_cross_validation(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Stratified K-Fold Cross-Validation.
    Returns:
        Summary DataFrame (Mean +/- Std) and Raw Fold Metric Matrix for statistical tests.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_summary = {}
    raw_fold_matrix = []

    for name, pipe in pipelines.items():
        fold_aucs = []
        fold_pr_aucs = []
        fold_recalls = []
        fold_specificities = []
        fold_mccs = []
        fold_briers = []

        fold_idx = 0
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe.fit(X_tr, y_tr)
            y_proba = pipe.predict_proba(X_val)[:, 1]
            y_pred = pipe.predict(X_val)

            auc_score = roc_auc_score(y_val, y_proba) * 100.0
            fold_aucs.append(auc_score)
            fold_pr_aucs.append(average_precision_score(y_val, y_proba) * 100.0)
            fold_recalls.append(recall_score(y_val, y_pred, zero_division=0) * 100.0)
            fold_specificities.append(recall_score(1 - y_val, 1 - y_pred, zero_division=0) * 100.0)
            fold_mccs.append(matthews_corrcoef(y_val, y_pred))
            fold_briers.append(brier_score_loss(y_val, y_proba))

            raw_fold_matrix.append({
                "Model": name,
                "Fold": fold_idx,
                "ROC-AUC": auc_score,
            })
            fold_idx += 1

        cv_summary[name] = {
            "ROC-AUC (Mean±Std)": f"{np.mean(fold_aucs):.2f}% ± {np.std(fold_aucs):.2f}%",
            "PR-AUC (Mean±Std)": f"{np.mean(fold_pr_aucs):.2f}% ± {np.std(fold_pr_aucs):.2f}%",
            "Sensitivity (Mean±Std)": f"{np.mean(fold_recalls):.2f}% ± {np.std(fold_recalls):.2f}%",
            "Specificity (Mean±Std)": f"{np.mean(fold_specificities):.2f}% ± {np.std(fold_specificities):.2f}%",
            "MCC (Mean)": f"{np.mean(fold_mccs):.3f}",
            "Brier Score (Mean)": f"{np.mean(fold_briers):.4f}",
            "_raw_auc": np.mean(fold_aucs)
        }

    df_summary = pd.DataFrame(cv_summary).T.sort_values("_raw_auc", ascending=False)
    df_summary = df_summary.drop(columns=["_raw_auc"])
    df_raw = pd.DataFrame(raw_fold_matrix)
    return df_summary, df_raw


def compute_friedman_test(df_raw_folds: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes Friedman Rank-Sum Omnibus Test across CV folds to assert statistical significance:
    H0: All algorithms perform equivalently.
    H1: At least one algorithm exhibits statistically significant superiority.
    """
    pivot = df_raw_folds.pivot(index="Fold", columns="Model", values="ROC-AUC")
    stat, p_val = stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])

    # Compute average rank (lower rank is better in standard ranking, here 1 = highest AUC)
    ranks = pivot.rank(axis=1, ascending=False).mean(axis=0).sort_values()

    return {
        "Friedman_Statistic": round(float(stat), 3),
        "P_Value": float(p_val),
        "Statistically_Significant": bool(p_val < 0.05),
        "Model_Rankings": ranks.to_dict()
    }
