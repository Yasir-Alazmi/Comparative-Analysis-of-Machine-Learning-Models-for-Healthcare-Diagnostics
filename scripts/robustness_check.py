#!/usr/bin/env python3
"""
Standalone robustness check for one dataset (a CLI script, not a test).

Runs:
1. 5-fold stratified cross-validation.
2. Multi-seed stability over several random 80/20 splits.
3. Simulated Gaussian noise on numeric test features for logistic regression
   (noise SD = 0-20% of each feature's test-set SD; not real sensor drift).

The published results come from scripts/reproduce_benchmarks.py, not from this script.

Usage:
    python scripts/robustness_check.py --dataset breast_cancer --fast
"""

import os
import sys
import argparse

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split

from src.models import build_models, build_clinical_pipelines
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_cross_validation
from src.robustness import evaluate_multi_seed_stability, evaluate_sensor_noise_drift


def run_robustness(dataset_key: str, fast: bool = False):
    meta = DATASET_REGISTRY[dataset_key]
    print("\n" + "=" * 85)
    print(f"  ROBUSTNESS CHECK: {meta['name'].upper()}")
    print("=" * 85)

    X, y = meta["loader"]()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    models = build_models(random_state=42, fast_mode=fast, include_stacking=False)
    pipelines = build_clinical_pipelines(
        models=models, num_cols=num_cols, cat_cols=cat_cols,
        use_smote=meta["use_smote"], random_state=42,
    )

    print("\n[1/3] 5-fold stratified cross-validation...")
    df_cv, _ = evaluate_cross_validation(pipelines, X, y, n_splits=5, random_state=42)
    print(df_cv.to_string())

    print("\n[2/3] Multi-seed stability over random 80/20 splits...")
    seeds = [7, 13, 21, 42, 99] if fast else [7, 13, 21, 42, 77, 99, 123, 256, 512, 1024]
    df_stability = evaluate_multi_seed_stability(pipelines, X, y, seeds=seeds)
    print(df_stability.to_string())

    print("\n[3/3] Simulated Gaussian noise on numeric features (logistic regression)...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    df_noise = evaluate_sensor_noise_drift(
        pipelines["Logistic Regression"], X_tr, y_tr, X_te, y_te,
        noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20],
        n_bootstraps=200 if fast else 1000,
    )
    print(df_noise[["Perturbation Level", "ROC-AUC [95% CI]", "Sensitivity [95% CI]", "Retention_Ratio"]].to_string())
    print("\n[+] Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Standalone robustness check for one dataset")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="breast_cancer")
    parser.add_argument("--fast", action="store_true", help="Quick run mode")
    args = parser.parse_args()
    run_robustness(args.dataset, fast=args.fast)


if __name__ == "__main__":
    main()
