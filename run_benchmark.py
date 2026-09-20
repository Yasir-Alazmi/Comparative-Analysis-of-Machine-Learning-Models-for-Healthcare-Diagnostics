#!/usr/bin/env python3
"""
Master CLI Benchmark Runner for Trustworthy Healthcare Machine Learning.
Evaluates 10 clinical models plus the Level-1 Stacking Super Learner across clinical cohorts:
- Flagship: CDC NHANES Cardiovascular & Metabolic Cohort (N >= 10,000)
- Benchmarks: Heart Failure, Stroke Prediction, Pima Diabetes, Breast Cancer, CKD

Usage:
    python run_benchmark.py --dataset nhanes_cardiovascular
    python run_benchmark.py --dataset all --cv 5
    python run_benchmark.py --dataset nhanes_cardiovascular --conformal --robustness --shap
    python run_benchmark.py --fast --no-plot
"""

import os
import sys
import argparse

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.models import build_models, build_clinical_pipelines
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_models, evaluate_cross_validation, compute_friedman_test
from src.conformal import evaluate_conformal_prediction
from src.robustness import evaluate_sensor_noise_drift, evaluate_multi_seed_stability
from src.explainability import explain_clinical_model, audit_clinical_guideline_alignment, render_shap_summary_plot
from src.fairness import evaluate_demographic_fairness
from src.publication_report import plot_publication_discrimination, plot_publication_robustness, generate_tripod_checklist_markdown


def run_benchmark_dataset(
    key: str,
    cv_folds: int = 0,
    fast: bool = False,
    run_conformal: bool = True,
    run_robustness: bool = True,
    run_shap: bool = True,
    no_plot: bool = False
):
    meta = DATASET_REGISTRY[key]
    print("\n" + "=" * 85)
    print(f"  CLINICAL BENCHMARK: {meta['name'].upper()}")
    print("=" * 85)

    print(f"[*] Loading and engineering features for: {key}...")
    X, y = meta["loader"]()
    
    # Partition column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"[*] Cohort Size: {len(X)} patients | Features: {X.shape[1]} (Num: {len(num_cols)}, Cat: {len(cat_cols)})")
    print(f"[*] Target Condition: {meta['target_name']} | Event Rate (Prevalence): {np.mean(y) * 100:.2f}%")

    print("[*] Assembling 10 Machine Learning Models & Stacking Super Learner...")
    models = build_models(random_state=42, fast_mode=fast, include_stacking=True)
    pipelines = build_clinical_pipelines(
        models=models,
        num_cols=num_cols,
        cat_cols=cat_cols,
        use_smote=meta["use_smote"],
        random_state=42
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", key)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Main Standard Hold-out Evaluation
    print("\n[*] [1/5] Training and Evaluating Models on Hold-Out Test Set (80/20 Stratified)...")
    results_df, curve_data = evaluate_models(
        pipelines=pipelines,
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        sort_metric=meta["sort_metric"]
    )

    print("-" * 85)
    print(f"{'Algorithm':<32} {'ROC-AUC':>10} {'PR-AUC':>10} {'Recall':>10} {'Spec':>10} {'ECE':>10}")
    print("-" * 85)
    for alg, row in results_df.iterrows():
        print(f"{str(alg):<32} {row['ROC-AUC']:>9.2f}% {row['PR-AUC']:>9.2f}% {row['Sensitivity (Recall)']:>9.2f}% {row['Specificity']:>9.2f}% {row['ECE']:>10.4f}")
    print("-" * 85)

    best_model_name = results_df.index[0]
    best_auc = results_df.loc[best_model_name, "ROC-AUC"]
    print(f"[+] Top Performer ({meta['sort_metric']}): {best_model_name} ({best_auc:.2f}%)")

    csv_path = os.path.join(out_dir, "benchmark_metrics.csv")
    results_df.to_csv(csv_path)
    print(f"[+] Metrics saved to: {csv_path}")

    # 2. Stratified Cross-Validation & Statistical Inference (Friedman Test)
    if cv_folds > 1:
        print(f"\n[*] [2/5] Executing {cv_folds}-Fold Stratified Cross-Validation & Friedman Test...")
        df_cv, df_raw_folds = evaluate_cross_validation(pipelines, X, y, n_splits=cv_folds, random_state=42)
        print("-" * 85)
        print(df_cv.to_string())
        print("-" * 85)
        df_cv.to_csv(os.path.join(out_dir, "cross_validation_metrics.csv"))

        friedman_res = compute_friedman_test(df_raw_folds)
        print(f"[+] Friedman Omnibus Test: Chi-Square = {friedman_res['Friedman_Statistic']}, p-value = {friedman_res['P_Value']:.4e}")
        print(f"[+] Statistical Significance Confirmed: {friedman_res['Statistically_Significant']}")

    # 3. Inductive Conformal Prediction & Calibration (Uncertainty Quantification)
    if run_conformal:
        print("\n[*] [3/5] Computing Inductive Conformal Prediction Sets (Guaranteed 95% Coverage)...")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        top_pipe = pipelines[best_model_name]
        conformal_res = evaluate_conformal_prediction(top_pipe, X_tr, y_tr, X_te, y_te, alpha=0.05, random_state=42)
        print("-" * 85)
        for k, v in conformal_res.items():
            print(f"  - {k:<30}: {v}")
        print("-" * 85)
        pd.DataFrame([conformal_res]).to_csv(os.path.join(out_dir, "conformal_prediction.csv"), index=False)

    # 4. Clinical Sensor Perturbation Stress-Testing
    if run_robustness:
        print("\n[*] [4/5] Executing Clinical Sensor Noise Stress-Test (+0% to +20% Gaussian Drift)...")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        top_pipe = pipelines[best_model_name]
        df_noise = evaluate_sensor_noise_drift(top_pipe, X_tr, y_tr, X_te, y_te, noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20])
        print("-" * 85)
        print(df_noise.to_string())
        print("-" * 85)
        df_noise.to_csv(os.path.join(out_dir, "sensor_noise_stress_test.csv"))

        if not no_plot:
            plot_publication_robustness(df_noise, os.path.join(out_dir, "figure2_sensor_robustness.png"))

    # 5. Explainable AI (SHAP) & Clinical Guideline Alignment
    if run_shap:
        print("\n[*] [5/5] Computing SHAP Attributions & Verifying AHA/ACC Biomarker Alignment...")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # Use a tree-based model for SHAP (XGBoost or LightGBM or CatBoost) for fast exact TreeSHAP
        shap_candidate = "XGBoost" if "XGBoost" in pipelines else best_model_name
        shap_pipe = pipelines[shap_candidate]
        shap_pipe.fit(X_tr, y_tr)
        shap_vals, importance_df = explain_clinical_model(shap_pipe, X_tr, X_te, max_eval_samples=150)
        
        guideline_audit = audit_clinical_guideline_alignment(importance_df, top_k=5)
        print("-" * 85)
        print(f"[+] Top Clinical Biomarkers Identified by Model: {', '.join(guideline_audit['Top_Features'])}")
        print(f"[+] AHA/ACC Guideline Alignment Rate: {guideline_audit['Guideline_Alignment_Rate']}")
        print(f"[+] Clinical Safety Audit: {guideline_audit['Clinical_Safety_Verdict']}")
        print("-" * 85)
        importance_df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)

        if not no_plot:
            render_shap_summary_plot(shap_vals, importance_df, os.path.join(out_dir, "figure3_shap_biomarkers.png"))

    # Publication Figures & TRIPOD+AI Documentation
    if not no_plot:
        plot_publication_discrimination(curve_data["roc"], curve_data["pr"], os.path.join(out_dir, "figure1_discrimination.png"))
        generate_tripod_checklist_markdown(os.path.join(out_dir, "tripod_ai_checklist.md"))
        print(f"[+] Publication-grade 300 DPI figures and TRIPOD+AI appendix generated in: {out_dir}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Trustworthy Clinical Machine Learning Benchmark Suite")
    parser.add_argument("--dataset", choices=["all"] + list(DATASET_REGISTRY.keys()), default="nhanes_cardiovascular",
                        help="Target clinical dataset (default: nhanes_cardiovascular)")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation (e.g. --cv 5)")
    parser.add_argument("--fast", action="store_true", help="Fast smoke run with fewer estimators")
    parser.add_argument("--conformal", action="store_true", default=True, help="Run Inductive Conformal Prediction")
    parser.add_argument("--robustness", action="store_true", default=True, help="Run Sensor Noise Stress-Test")
    parser.add_argument("--shap", action="store_true", default=True, help="Run SHAP Explainability & Clinical Audit")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot figures")
    args = parser.parse_args()

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    for t in targets:
        run_benchmark_dataset(
            key=t,
            cv_folds=args.cv,
            fast=args.fast,
            run_conformal=args.conformal,
            run_robustness=args.robustness,
            run_shap=args.shap,
            no_plot=args.no_plot
        )

    print("\n" + "=" * 85)
    print("  ALL BENCHMARKS & PUBLICATION PROTOCOLS COMPLETED SUCCESSFULLY!")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
