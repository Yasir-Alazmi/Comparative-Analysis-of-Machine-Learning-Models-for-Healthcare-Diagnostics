"""
Comprehensive Clinical Evaluation & Statistical Inference Engine.
Features:
1. Honest Clinical Metrics: ROC-AUC, PR-AUC, Sensitivity, Specificity, Balanced Accuracy, MCC, Brier Score, ECE.
2. Zero-Snooping Threshold Locking: Optimal decision thresholds (Youden's J) are selected exclusively
   on training partitions and locked before blind evaluation on the held-out test set.
3. 1,000-Resample Non-Parametric Bootstrap 95% Confidence Intervals [Lower, Upper] for all clinical metrics.
4. Paired Bootstrap Hypothesis Testing for statistically rigorous model comparisons.
5. K-Fold Stratified Cross-Validation & Friedman Omnibus Hypothesis Testing.
"""

from typing import Dict, Tuple, Any, List, Optional
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


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes empirical 1,000-resample non-parametric bootstrap 95% Confidence Intervals
    (Percentile method) for all discrimination, calibration, and clinical utility metrics.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    boot_aucs = []
    boot_praucs = []
    boot_sens = []
    boot_specs = []
    boot_bal_accs = []
    boot_briers = []
    boot_eces = []
    boot_mccs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes are present in the resample
        if len(np.unique(y_true[idx])) < 2:
            continue

        y_t_b = y_true[idx]
        y_p_b = y_proba[idx]
        y_hat_b = y_pred[idx]

        try:
            boot_aucs.append(roc_auc_score(y_t_b, y_p_b))
        except Exception:
            pass

        boot_praucs.append(average_precision_score(y_t_b, y_p_b))
        boot_sens.append(recall_score(y_t_b, y_hat_b, zero_division=0))
        boot_specs.append(recall_score(1 - y_t_b, 1 - y_hat_b, zero_division=0))
        boot_bal_accs.append(balanced_accuracy_score(y_t_b, y_hat_b))
        boot_briers.append(brier_score_loss(y_t_b, y_p_b))
        ece_b, _, _ = compute_expected_calibration_error(y_t_b, y_p_b)
        boot_eces.append(ece_b)
        boot_mccs.append(matthews_corrcoef(y_t_b, y_hat_b))

    def get_ci(arr, scale=100.0, decimals=2):
        if len(arr) == 0:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "formatted": "N/A"}
        lower = np.percentile(arr, (alpha / 2.0) * 100) * scale
        upper = np.percentile(arr, (1.0 - alpha / 2.0) * 100) * scale
        mean = np.mean(arr) * scale
        fmt = f"{mean:.{decimals}f} [{lower:.{decimals}f} - {upper:.{decimals}f}]"
        return {
            "mean": round(mean, decimals),
            "lower": round(lower, decimals),
            "upper": round(upper, decimals),
            "formatted": fmt,
        }

    return {
        "ROC-AUC": get_ci(boot_aucs, scale=100.0, decimals=2),
        "PR-AUC": get_ci(boot_praucs, scale=100.0, decimals=2),
        "Sensitivity": get_ci(boot_sens, scale=100.0, decimals=2),
        "Specificity": get_ci(boot_specs, scale=100.0, decimals=2),
        "Balanced_Acc": get_ci(boot_bal_accs, scale=100.0, decimals=2),
        "Brier": get_ci(boot_briers, scale=1.0, decimals=4),
        "ECE": get_ci(boot_eces, scale=1.0, decimals=4),
        "MCC": get_ci(boot_mccs, scale=1.0, decimals=3),
    }


def compute_paired_roc_bootstrap_test(
    y_true: np.ndarray,
    proba_model_a: np.ndarray,
    proba_model_b: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap hypothesis test for difference in ROC-AUC between two competing models on the same test set.
    H0: delta_AUC = AUC_A - AUC_B == 0.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    delta_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_true[idx], proba_model_a[idx])
            auc_b = roc_auc_score(y_true[idx], proba_model_b[idx])
            delta_aucs.append(auc_a - auc_b)
        except Exception:
            pass

    delta_arr = np.array(delta_aucs)
    mean_diff = np.mean(delta_arr) * 100.0
    ci_lower = np.percentile(delta_arr, 2.5) * 100.0
    ci_upper = np.percentile(delta_arr, 97.5) * 100.0

    # Empirical two-sided p-value
    p_val = 2.0 * min(np.mean(delta_arr <= 0), np.mean(delta_arr >= 0))

    return {
        "Mean_Delta_AUC (%)": round(mean_diff, 2),
        "Delta_AUC_95_CI": f"[{ci_lower:.2f}%, {ci_upper:.2f}%]",
        "P_Value": round(float(p_val), 4),
        "Statistically_Significant": bool(p_val < 0.05),
    }


def evaluate_models(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    sort_metric: str = "ROC-AUC",
    compute_ci: bool = True,
    n_bootstraps: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates clinical models with strict zero-snooping threshold locking:
    1. Splits into Train (80%) and Test (20%).
    2. Decision threshold (Youden's J) is identified strictly on training partition tuning data.
    3. Pipeline fits on full Train, locks threshold tau*, and prospectively evaluates on untouched Test.
    4. Computes 1,000-resample non-parametric bootstrap 95% Confidence Intervals for all metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results = {}
    roc_data = {}
    pr_data = {}
    y_test_arr = np.asarray(y_test)

    # Split training partition internally for threshold selection (zero test set snooping)
    X_tr_dev, X_tr_tune, y_tr_dev, y_tr_tune = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=random_state
    )

    for name, pipe in pipelines.items():
        # Step A: Identify optimal clinical threshold on training data only
        pipe.fit(X_tr_dev, y_tr_dev)
        if hasattr(pipe, "predict_proba"):
            tune_proba = pipe.predict_proba(X_tr_tune)[:, 1]
        else:
            tune_proba = pipe.predict(X_tr_tune)

        thresh_info = find_optimal_clinical_threshold(np.asarray(y_tr_tune), tune_proba)
        locked_threshold = thresh_info["Optimal_Threshold"]

        # Step B: Refit pipeline on FULL training partition
        pipe.fit(X_train, y_train)

        # Step C: Blind prospective evaluation on UNTOUCHED test set using locked threshold
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = pipe.predict(X_test)

        y_pred_locked = (y_proba >= locked_threshold).astype(int)

        # Standard point metrics
        rec = recall_score(y_test_arr, y_pred_locked, zero_division=0)
        spec = recall_score(1 - y_test_arr, 1 - y_pred_locked, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test_arr, y_pred_locked)
        roc = roc_auc_score(y_test_arr, y_proba)
        pr_auc = average_precision_score(y_test_arr, y_proba)
        mcc = matthews_corrcoef(y_test_arr, y_pred_locked)
        brier = brier_score_loss(y_test_arr, y_proba)
        ece, _, _ = compute_expected_calibration_error(y_test_arr, y_proba)

        # Curves
        fpr, tpr, _ = roc_curve(y_test_arr, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        precision_pts, recall_pts, _ = precision_recall_curve(y_test_arr, y_proba)
        pr_data[name] = (recall_pts, precision_pts, pr_auc)

        row = {
            "ROC-AUC": round(roc * 100, 2),
            "ROC-AUC (%)": round(roc * 100, 2),
            "PR-AUC": round(pr_auc * 100, 2),
            "PR-AUC (%)": round(pr_auc * 100, 2),
            "Sensitivity (Recall)": round(rec * 100, 2),
            "Sensitivity (%)": round(rec * 100, 2),
            "Specificity": round(spec * 100, 2),
            "Specificity (%)": round(spec * 100, 2),
            "Balanced Acc": round(bal_acc * 100, 2),
            "Balanced Acc (%)": round(bal_acc * 100, 2),
            "MCC": round(mcc, 3),
            "Brier Score": round(brier, 4),
            "ECE": round(ece, 4),
            "Locked_Threshold": locked_threshold,
        }

        # Step D: 1,000-resample Bootstrap 95% Confidence Intervals
        if compute_ci:
            ci_dict = compute_bootstrap_confidence_intervals(
                y_test_arr, y_proba, y_pred_locked,
                n_bootstraps=n_bootstraps, random_state=random_state
            )
            row["ROC-AUC [95% CI]"] = ci_dict["ROC-AUC"]["formatted"]
            row["Sensitivity [95% CI]"] = ci_dict["Sensitivity"]["formatted"]
            row["Specificity [95% CI]"] = ci_dict["Specificity"]["formatted"]
            row["PR-AUC [95% CI]"] = ci_dict["PR-AUC"]["formatted"]
            row["Brier [95% CI]"] = ci_dict["Brier"]["formatted"]
            row["ECE [95% CI]"] = ci_dict["ECE"]["formatted"]

        results[name] = row

    results_df = pd.DataFrame(results).T
    primary_sort = sort_metric if sort_metric in results_df.columns else ("ROC-AUC" if "ROC-AUC" in results_df.columns else results_df.columns[0])
    results_df = results_df.sort_values(primary_sort, ascending=False)
    return results_df, {"roc": roc_data, "pr": pr_data}


def evaluate_cross_validation(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Stratified K-Fold Cross-Validation with leak-free fold execution.
    Returns summary DataFrame (Mean +/- Std) and Raw Fold Matrix for statistical hypothesis tests.
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

        # Calculate 95% Confidence Interval across folds: Mean +/- 1.96 * (std / sqrt(k))
        n_k = len(fold_aucs)
        ci_auc = 1.96 * (np.std(fold_aucs) / np.sqrt(n_k))
        ci_rec = 1.96 * (np.std(fold_recalls) / np.sqrt(n_k))

        cv_summary[name] = {
            "ROC-AUC [95% CI]": f"{np.mean(fold_aucs):.2f}% ± {ci_auc:.2f}%",
            "Sensitivity [95% CI]": f"{np.mean(fold_recalls):.2f}% ± {ci_rec:.2f}%",
            "PR-AUC (Mean±Std)": f"{np.mean(fold_pr_aucs):.2f}% ± {np.std(fold_pr_aucs):.2f}%",
            "Specificity (Mean±Std)": f"{np.mean(fold_specificities):.2f}% ± {np.std(fold_specificities):.2f}%",
            "MCC (Mean)": f"{np.mean(fold_mccs):.3f}",
            "Brier Score (Mean)": f"{np.mean(fold_briers):.4f}",
            "_raw_auc": np.mean(fold_aucs),
        }

    df_summary = pd.DataFrame(cv_summary).T.sort_values("_raw_auc", ascending=False)
    df_summary = df_summary.drop(columns=["_raw_auc"])
    df_raw = pd.DataFrame(raw_fold_matrix)
    return df_summary, df_raw


def compute_friedman_test(df_raw_folds: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes Friedman Rank-Sum Omnibus Test across CV folds.
    H0: All algorithms exhibit equivalent performance distributions.
    """
    pivot = df_raw_folds.pivot(index="Fold", columns="Model", values="ROC-AUC")
    stat, p_val = stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])
    ranks = pivot.rank(axis=1, ascending=False).mean(axis=0).sort_values()

    return {
        "Friedman_Statistic": round(float(stat), 3),
        "P_Value": float(p_val),
        "Statistically_Significant": bool(p_val < 0.05),
        "Model_Rankings": ranks.to_dict(),
    }


def evaluate_locked_external_validation(
    pipeline: Any,
    locked_threshold: float,
    X_ext: pd.DataFrame,
    y_ext: pd.Series,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates a frozen, locked clinical pipeline for independent external transportability
    on an independent cohort (e.g., authentic Framingham Heart Study cohort) with zero retraining
    and zero threshold adjustment.

    Calculates discrimination, calibration drift, and 1,000-resample non-parametric bootstrap 95% CIs.
    """
    y_ext_arr = np.asarray(y_ext)

    # Harmonize feature columns with the training pipeline's expected feature space
    X_ext_aligned = X_ext.copy()
    expected_cols = None
    if hasattr(pipeline, "feature_names_in_"):
        expected_cols = list(pipeline.feature_names_in_)
    elif hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        prep = pipeline.named_steps["preprocessor"]
        if hasattr(prep, "feature_names_in_"):
            expected_cols = list(prep.feature_names_in_)

    if expected_cols is not None:
        for col in expected_cols:
            if col not in X_ext_aligned.columns:
                X_ext_aligned[col] = np.nan
        X_ext_aligned = X_ext_aligned[expected_cols]

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_ext_aligned)[:, 1]
    else:
        y_proba = pipeline.predict(X_ext_aligned)

    y_pred = (y_proba >= locked_threshold).astype(int)

    roc = roc_auc_score(y_ext_arr, y_proba)
    pr_auc = average_precision_score(y_ext_arr, y_proba)
    rec = recall_score(y_ext_arr, y_pred, zero_division=0)
    spec = recall_score(1 - y_ext_arr, 1 - y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_ext_arr, y_pred)
    mcc = matthews_corrcoef(y_ext_arr, y_pred)
    brier = brier_score_loss(y_ext_arr, y_proba)
    ece, _, _ = compute_expected_calibration_error(y_ext_arr, y_proba)

    ci_dict = compute_bootstrap_confidence_intervals(
        y_ext_arr, y_proba, y_pred, n_bootstraps=n_bootstraps, random_state=random_state
    )

    metrics = {
        "Cohort": "Real Framingham Heart Study (N=4,240 Longitudinal Follow-up)",
        "N_Patients": len(y_ext),
        "Event_Rate (%)": round(float(np.mean(y_ext_arr) * 100), 2),
        "Locked_Threshold": locked_threshold,
        "ROC-AUC (%)": round(roc * 100, 2),
        "ROC-AUC [95% CI]": ci_dict["ROC-AUC"]["formatted"],
        "PR-AUC (%)": round(pr_auc * 100, 2),
        "PR-AUC [95% CI]": ci_dict["PR-AUC"]["formatted"],
        "Sensitivity (%)": round(rec * 100, 2),
        "Sensitivity [95% CI]": ci_dict["Sensitivity"]["formatted"],
        "Specificity (%)": round(spec * 100, 2),
        "Specificity [95% CI]": ci_dict["Specificity"]["formatted"],
        "Balanced Acc (%)": round(bal_acc * 100, 2),
        "MCC": round(mcc, 3),
        "Brier Score": round(brier, 4),
        "Brier [95% CI]": ci_dict["Brier"]["formatted"],
        "ECE": round(ece, 4),
        "ECE [95% CI]": ci_dict["ECE"]["formatted"],
    }

    df_out = pd.DataFrame([metrics])
    return df_out, metrics

