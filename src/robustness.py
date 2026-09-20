"""
Clinical Stress-Testing & Robustness Verification Module.
Features:
1. Laboratory Sensor Perturbation Drift (0% to 20% Gaussian noise)
2. Missing Clinical Feature Stress-Test (simulating resource-constrained rural clinics)
3. Multi-Seed Generalization Stability across 10 distinct random initializations
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score


def evaluate_sensor_noise_drift(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Simulates clinical laboratory instrument calibration drift by injecting Gaussian perturbation
    into continuous clinical covariates of test subjects.
    """
    pipeline.fit(X_train, y_train)
    resilience = {}

    # Identify continuous numerical columns
    num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    stds = X_test[num_cols].std().replace(0, 1.0)

    for noise in noise_levels:
        if noise == 0.0:
            X_noisy = X_test.copy()
        else:
            rng = np.random.default_rng(random_state)
            X_noisy = X_test.copy()
            noise_matrix = rng.normal(0.0, noise, size=(len(X_test), len(num_cols))) * stds.values
            X_noisy[num_cols] = X_noisy[num_cols] + noise_matrix

        y_pred = pipeline.predict(X_noisy)
        y_proba = pipeline.predict_proba(X_noisy)[:, 1]

        auc_val = roc_auc_score(y_test, y_proba) * 100.0
        rec_val = recall_score(y_test, y_pred, zero_division=0) * 100.0
        bal_acc = balanced_accuracy_score(y_test, y_pred) * 100.0

        label = f"Noise_{int(noise * 100)}%"
        resilience[label] = {
            "ROC-AUC": round(auc_val, 2),
            "Sensitivity (Recall)": round(rec_val, 2),
            "Balanced Accuracy": round(bal_acc, 2),
            "Retention_Ratio": f"{(auc_val / resilience.get('Noise_0%', {}).get('ROC-AUC', auc_val)) * 100:.1f}%" if noise > 0 else "100.0%"
        }

    return pd.DataFrame(resilience).T


def evaluate_missing_feature_stress(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    drop_fractions: List[float] = [0.0, 0.10, 0.20, 0.30],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Simulates incomplete diagnostic panels in rural or emergency clinical settings
    by randomly masking feature values with NaN to test pipeline imputation resilience.
    """
    pipeline.fit(X_train, y_train)
    stress_results = {}
    num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

    for frac in drop_fractions:
        X_masked = X_test.copy()
        if frac > 0.0:
            rng = np.random.default_rng(random_state)
            mask = rng.uniform(0.0, 1.0, size=X_masked[num_cols].shape) < frac
            X_masked_num = X_masked[num_cols].copy()
            X_masked_num[mask] = np.nan
            X_masked[num_cols] = X_masked_num

        y_proba = pipeline.predict_proba(X_masked)[:, 1]
        y_pred = pipeline.predict(X_masked)

        auc_val = roc_auc_score(y_test, y_proba) * 100.0
        rec_val = recall_score(y_test, y_pred, zero_division=0) * 100.0

        label = f"Missing_{int(frac * 100)}%"
        stress_results[label] = {
            "ROC-AUC": round(auc_val, 2),
            "Sensitivity": round(rec_val, 2),
        }

    return pd.DataFrame(stress_results).T


def evaluate_multi_seed_stability(
    pipelines: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    seeds: List[int] = [42, 1337, 2024, 7, 99, 123, 256, 512, 777, 1024]
) -> pd.DataFrame:
    """
    Calculates 95% Confidence Intervals over 10 random independent train/test splits.
    """
    stability = {}
    for name, pipe in pipelines.items():
        aucs = []
        recalls = []
        for s in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
            pipe.fit(X_tr, y_tr)
            y_proba = pipe.predict_proba(X_te)[:, 1]
            y_pred = pipe.predict(X_te)

            aucs.append(roc_auc_score(y_te, y_proba) * 100.0)
            recalls.append(recall_score(y_te, y_pred, zero_division=0) * 100.0)

        mean_auc = np.mean(aucs)
        ci95_auc = 1.96 * (np.std(aucs) / np.sqrt(len(seeds)))

        stability[name] = {
            "ROC-AUC (Mean ± 95% CI)": f"{mean_auc:.2f}% ± {ci95_auc:.2f}%",
            "Sensitivity (Mean ± Std)": f"{np.mean(recalls):.2f}% ± {np.std(recalls):.2f}%",
            "Min ROC-AUC": round(np.min(aucs), 2),
            "Max ROC-AUC": round(np.max(aucs), 2),
            "_sort_auc": mean_auc
        }

    df_out = pd.DataFrame(stability).T.sort_values("_sort_auc", ascending=False)
    return df_out.drop(columns=["_sort_auc"])
