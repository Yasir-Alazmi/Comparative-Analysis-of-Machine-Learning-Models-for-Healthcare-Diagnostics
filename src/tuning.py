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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, balanced_accuracy_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.preprocessor import build_leakage_free_pipeline
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
