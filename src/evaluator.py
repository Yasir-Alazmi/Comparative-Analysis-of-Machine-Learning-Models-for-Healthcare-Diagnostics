"""
Evaluation engine for benchmarking 10 machine learning models.
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Handles SMOTE oversampling for imbalanced medical data.
"""

from typing import Dict, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
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
    Fits and evaluates a dictionary of models on the given dataset.

    Returns:
        results_df: DataFrame with metrics for all algorithms.
        roc_data: Dictionary mapping model names to (fpr, tpr, roc_auc) tuples.
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
            y_proba = pipe.decision_function(X_test)
        else:
            y_proba = y_pred

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba) * 100, 2),
        }

    results_df = pd.DataFrame(results).T.sort_values(sort_metric, ascending=False)
    return results_df, roc_data
