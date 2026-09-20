"""
Bayesian Hyperparameter Optimization & Clinical Threshold Tuning Module.
Features:
1. Optuna-driven Bayesian Optimization across model parameter search spaces
2. Clinical Threshold Optimization via Youden's J-statistic (maximizing Sensitivity)
3. 5x5 Nested Stratified Cross-Validation for unbiased publication benchmarks
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.preprocessor import build_leakage_free_pipeline
from src.evaluator import compute_expected_calibration_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def find_optimal_clinical_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Identifies the optimal diagnostic probability threshold maximizing Youden's J-Index:
    J = Sensitivity + Specificity - 1
    Prioritizes clinical Sensitivity (Recall) while penalizing excessive false alarms.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_j = -1.0
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        # Specificity = recall of negative class
        spec = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_thresh = t
            best_metrics = {
                "Optimal_Threshold": round(float(t), 4),
                "Sensitivity_at_Threshold": round(float(sens) * 100, 2),
                "Specificity_at_Threshold": round(float(spec) * 100, 2),
                "Balanced_Acc_at_Threshold": round(float(balanced_accuracy_score(y_true, y_pred)) * 100, 2),
                "Youden_J": round(float(j), 4),
            }

    return best_metrics


def sample_hyperparameters_optuna(trial: optuna.Trial, model_name: str, random_state: int = 42) -> Any:
    """
    Defines search spaces for primary clinical classifiers.
    """
    if model_name == "CatBoost":
        depth = trial.suggest_int("depth", 4, 8)
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        l2 = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        return CatBoostClassifier(iterations=200, depth=depth, learning_rate=lr, l2_leaf_reg=l2, verbose=0, random_seed=random_state)

    elif model_name == "LightGBM":
        num_leaves = trial.suggest_int("num_leaves", 15, 63)
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        min_child = trial.suggest_int("min_child_samples", 10, 50)
        return LGBMClassifier(n_estimators=200, num_leaves=num_leaves, learning_rate=lr, min_child_samples=min_child, random_state=random_state, verbose=-1)

    elif model_name == "XGBoost":
        max_depth = trial.suggest_int("max_depth", 3, 7)
        lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        return XGBClassifier(n_estimators=200, max_depth=max_depth, learning_rate=lr, subsample=subsample, colsample_bytree=colsample, eval_metric="logloss", random_state=random_state)

    elif model_name == "Random Forest":
        max_depth = trial.suggest_int("max_depth", 6, 16)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
        return RandomForestClassifier(n_estimators=200, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)

    else:
        # Logistic Regression
        c_val = trial.suggest_float("C", 0.01, 10.0, log=True)
        return LogisticRegression(C=c_val, max_iter=1500, random_state=random_state)


def tune_model_optuna(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: list,
    cat_cols: list,
    n_trials: int = 30,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executes Bayesian optimization with Stratified K-Fold cross-validation to maximize ROC-AUC.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        clf = sample_hyperparameters_optuna(trial, model_name, random_state=random_state)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = build_leakage_free_pipeline(clf, num_cols, cat_cols, use_smote=False, random_state=random_state)
            pipe.fit(X_tr, y_tr)
            y_proba = pipe.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_proba))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_clf = sample_hyperparameters_optuna(study.best_trial, model_name, random_state=random_state)
    return best_clf, study.best_params


def evaluate_nested_cross_validation(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: list,
    cat_cols: list,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_trials: int = 15,
    use_smote: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes true 5x5 Nested Stratified Cross-Validation with inner-loop Optuna Bayesian optimization.
    Guarantees zero selection bias: hyperparameter tuning is restricted strictly to inner folds,
    and clinical decision thresholds are derived on training partitions without test snooping.

    Returns:
        df_summary: DataFrame with mean metrics, standard deviations, and 95% Confidence Intervals.
        df_outer_folds: DataFrame with per-fold prospective outer test evaluation metrics.
    """
    outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    outer_records = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_skf.split(X, y), 1):
        X_tr_outer, X_te_outer = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_outer, y_te_outer = y.iloc[tr_idx], y.iloc[te_idx]

        # Inner loop: Bayesian Hyperparameter Optimization on outer training split only
        inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold_idx)

        def objective(trial):
            clf = sample_hyperparameters_optuna(trial, model_name, random_state=random_state)
            inner_scores = []
            for in_tr_idx, in_val_idx in inner_skf.split(X_tr_outer, y_tr_outer):
                in_X_tr, in_X_val = X_tr_outer.iloc[in_tr_idx], X_tr_outer.iloc[in_val_idx]
                in_y_tr, in_y_val = y_tr_outer.iloc[in_tr_idx], y_tr_outer.iloc[in_val_idx]

                pipe = build_leakage_free_pipeline(clf, num_cols, cat_cols, use_smote=use_smote, random_state=random_state)
                pipe.fit(in_X_tr, in_y_tr)
                if hasattr(pipe, "predict_proba"):
                    in_proba = pipe.predict_proba(in_X_val)[:, 1]
                else:
                    in_proba = pipe.predict(in_X_val)
                inner_scores.append(roc_auc_score(in_y_val, in_proba))
            return float(np.mean(inner_scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state + fold_idx))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_clf = sample_hyperparameters_optuna(study.best_trial, model_name, random_state=random_state)

        # Threshold derivation on outer training partition only (zero outer test snooping)
        X_dev, X_tune, y_dev, y_tune = train_test_split(
            X_tr_outer, y_tr_outer, test_size=0.25, stratify=y_tr_outer, random_state=random_state + fold_idx
        )
        tune_pipe = build_leakage_free_pipeline(best_clf, num_cols, cat_cols, use_smote=use_smote, random_state=random_state)
        tune_pipe.fit(X_dev, y_dev)
        tune_proba = tune_pipe.predict_proba(X_tune)[:, 1] if hasattr(tune_pipe, "predict_proba") else tune_pipe.predict(X_tune)
        thresh_info = find_optimal_clinical_threshold(np.asarray(y_tune), tune_proba)
        locked_thresh = thresh_info["Optimal_Threshold"]

        # Refit best model on FULL outer training fold
        final_pipe = build_leakage_free_pipeline(best_clf, num_cols, cat_cols, use_smote=use_smote, random_state=random_state)
        final_pipe.fit(X_tr_outer, y_tr_outer)

        # Blind prospective evaluation on untouched outer test partition
        y_te_proba = final_pipe.predict_proba(X_te_outer)[:, 1] if hasattr(final_pipe, "predict_proba") else final_pipe.predict(X_te_outer)
        y_te_arr = np.asarray(y_te_outer)
        y_te_pred = (y_te_proba >= locked_thresh).astype(int)

        roc = roc_auc_score(y_te_arr, y_te_proba)
        pr_auc = average_precision_score(y_te_arr, y_te_proba)
        sens = recall_score(y_te_arr, y_te_pred, zero_division=0)
        spec = recall_score(1 - y_te_arr, 1 - y_te_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_te_arr, y_te_pred)
        brier = brier_score_loss(y_te_arr, y_te_proba)
        ece, _, _ = compute_expected_calibration_error(y_te_arr, y_te_proba)
        mcc = matthews_corrcoef(y_te_arr, y_te_pred)

        outer_records.append({
            "Fold": fold_idx,
            "Model": model_name,
            "ROC-AUC": roc * 100.0,
            "PR-AUC": pr_auc * 100.0,
            "Sensitivity": sens * 100.0,
            "Specificity": spec * 100.0,
            "Balanced_Acc": bal_acc * 100.0,
            "Brier_Score": brier,
            "ECE": ece,
            "MCC": mcc,
            "Locked_Threshold": locked_thresh,
            "Best_Params": str(study.best_params),
        })

    df_outer_folds = pd.DataFrame(outer_records)

    # Compute outer summary statistics with 95% Confidence Intervals
    k = outer_splits
    summary_data = {}
    for col in ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Balanced_Acc", "Brier_Score", "ECE", "MCC"]:
        vals = df_outer_folds[col].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        ci_95 = 1.96 * (std_val / np.sqrt(k))
        if col in ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Balanced_Acc"]:
            summary_data[f"{col} [95% CI]"] = f"{mean_val:.2f}% ± {ci_95:.2f}%"
            summary_data[f"{col} (Mean±Std)"] = f"{mean_val:.2f}% ± {std_val:.2f}%"
        else:
            summary_data[f"{col} [95% CI]"] = f"{mean_val:.4f} ± {ci_95:.4f}"
            summary_data[f"{col} (Mean±Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"

    df_summary = pd.DataFrame([summary_data], index=[model_name])
    return df_summary, df_outer_folds

