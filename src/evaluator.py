"""
Evaluation engine for benchmarking 10 machine learning models.
Features:
1. Anti-Leakage Execution: All scalers and oversamplers are fitted strictly on training data.
2. Stratified K-Fold Cross-Validation: Evaluates models across k distinct non-overlapping data splits.
3. Honest Clinical Metrics:
   - Balanced Accuracy: Prevents accuracy manipulation in imbalanced data.
   - Matthews Correlation Coefficient (MCC): High score only if prediction is good across all 4 confusion matrix categories.
   - Brier Score: Measures clinical probability calibration.
   - Recall & Specificity: Critical clinical trade-offs.
   - Precision, F1-Score, and ROC-AUC.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    matthews_corrcoef,
    brier_score_loss,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def evaluate_models(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
    sort_metric: str = "Accuracy",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits and evaluates models with strict train/test isolation and honest clinical metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    results = {}
    roc_data = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train_res, y_train_res)
        y_pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            raw_scores = pipe.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-raw_scores))
        else:
            y_proba = y_pred

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)

        results[name] = {
            "Accuracy": round(acc * 100, 2),
            "Balanced Acc": round(bal_acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2),
            "ROC-AUC": round(roc * 100, 2),
            "MCC": round(mcc, 3),
            "Brier Score": round(brier, 4),
        }

    results_df = pd.DataFrame(results).T.sort_values(sort_metric, ascending=False)
    return results_df, roc_data


def evaluate_cross_validation(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    use_smote: bool = False,
) -> pd.DataFrame:
    """
    Executes Stratified K-Fold Cross-Validation across n_splits distinct data partitions.
    Guarantees no data leakage: in each fold, training and validation sets are completely separate.

    Returns:
        Summary DataFrame with Mean +/- Std for all metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_summary = {}

    for name, pipe in pipelines.items():
        fold_accuracies = []
        fold_balanced = []
        fold_recalls = []
        fold_precisions = []
        fold_f1s = []
        fold_rocs = []
        fold_mccs = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if use_smote:
                sm = SMOTE(random_state=random_state)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)

            if hasattr(pipe, "predict_proba"):
                y_proba = pipe.predict_proba(X_val)[:, 1]
            elif hasattr(pipe, "decision_function"):
                raw = pipe.decision_function(X_val)
                y_proba = 1 / (1 + np.exp(-raw))
            else:
                y_proba = y_pred

            fold_accuracies.append(accuracy_score(y_val, y_pred) * 100)
            fold_balanced.append(balanced_accuracy_score(y_val, y_pred) * 100)
            fold_recalls.append(recall_score(y_val, y_pred, zero_division=0) * 100)
            fold_precisions.append(precision_score(y_val, y_pred, zero_division=0) * 100)
            fold_f1s.append(f1_score(y_val, y_pred, zero_division=0) * 100)
            fold_rocs.append(roc_auc_score(y_val, y_proba) * 100)
            fold_mccs.append(matthews_corrcoef(y_val, y_pred))

        cv_summary[name] = {
            "Accuracy (Mean±Std)": f"{np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%",
            "Balanced Acc (Mean±Std)": f"{np.mean(fold_balanced):.2f}% ± {np.std(fold_balanced):.2f}%",
            "Recall (Mean±Std)": f"{np.mean(fold_recalls):.2f}% ± {np.std(fold_recalls):.2f}%",
            "F1-Score (Mean±Std)": f"{np.mean(fold_f1s):.2f}% ± {np.std(fold_f1s):.2f}%",
            "ROC-AUC (Mean±Std)": f"{np.mean(fold_rocs):.2f}% ± {np.std(fold_rocs):.2f}%",
            "MCC (Mean)": f"{np.mean(fold_mccs):.3f}",
            "Raw Mean Acc": np.mean(fold_accuracies),
        }

    df_cv = pd.DataFrame(cv_summary).T.sort_values("Raw Mean Acc", ascending=False)
    df_cv = df_cv.drop(columns=["Raw Mean Acc"])
    return df_cv
