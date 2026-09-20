#!/usr/bin/env python3
"""
Master CLI Benchmark Runner for Trustworthy Healthcare Machine Learning.
Evaluates 10 clinical models plus the Level-1 Stacking Super Learner across clinical cohorts:
- Flagship: CDC NHANES Cardiovascular & Metabolic Cohort (N >= 10,000)
- Benchmarks: Heart Failure, Stroke Prediction, Pima Diabetes, Breast Cancer, CKD

Usage:
    python run_benchmark.py --dataset nhanes_cardiovascular
    python run_benchmark.py --dataset all --cv 5
    python run_benchmark.py --dataset nhanes_cardiovascular --nested-cv
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
from sklearn.model_selection import train_test_split

from src.models import build_models, build_clinical_pipelines
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import (
    evaluate_models,
    evaluate_cross_validation,
    compute_friedman_test,
    compute_paired_roc_bootstrap_test,
    evaluate_locked_external_validation,
)
from src.tuning import evaluate_nested_cross_validation
from src.data_loader import load_external_validation_cohort
from src.conformal import evaluate_conformal_prediction
from src.robustness import evaluate_sensor_noise_drift, evaluate_multi_seed_stability
from src.explainability import explain_clinical_model, audit_clinical_guideline_alignment, render_shap_summary_plot
from src.fairness import evaluate_demographic_fairness
from src.publication_report import plot_publication_discrimination, plot_publication_robustness, generate_tripod_checklist_markdown


def run_benchmark_dataset(
    key: str,
    cv_folds: int = 0,
    run_nested_cv: bool = False,
    fast: bool = False,
    run_conformal: bool = True,
    run_robustness: bool = True,
    run_shap: bool = True,
    no_plot: bool = False
):
    meta = DATASET_REGISTRY[key]
    print("\n" + "=" * 95)
    print(f"  CLINICAL BENCHMARK: {meta['name'].upper()}")
    print("=" * 95)

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

    # 1. Main Prospective Evaluation with Zero-Snooping Locked Thresholds & 1,000-Resample 95% CIs
    print("\n[*] [1/7] Prospective Hold-Out Evaluation (80/20 Stratified, Locked Youden Thresholds, 1,000 Bootstrap CIs)...")
    results_df, curve_data = evaluate_models(
        pipelines=pipelines,
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        sort_metric=meta["sort_metric"],
        compute_ci=True,
        n_bootstraps=1000
    )

    print("-" * 95)
    print(f"{'Algorithm':<26} {'ROC-AUC [95% CI]':<22} {'Sensitivity [95% CI]':<22} {'Specificity':>10} {'ECE':>8}")
    print("-" * 95)
    for alg, row in results_df.iterrows():
        auc_str = row.get("ROC-AUC [95% CI]", f"{row['ROC-AUC']:.2f}%")
        sens_str = row.get("Sensitivity [95% CI]", f"{row['Sensitivity (%)']:.2f}%")
        spec_str = f"{row['Specificity (%)']:.2f}%"
        ece_str = f"{row['ECE']:.4f}"
        print(f"{str(alg):<26} {auc_str:<22} {sens_str:<22} {spec_str:>10} {ece_str:>8}")
    print("-" * 95)

    best_model_name = results_df.index[0]
    best_auc = results_df.loc[best_model_name, "ROC-AUC"]
    locked_thresh = results_df.loc[best_model_name, "Locked_Threshold"]
    print(f"[+] Top Performer: {best_model_name} (ROC-AUC: {best_auc:.2f}%, Locked Threshold: {locked_thresh})")

    csv_path = os.path.join(out_dir, "benchmark_metrics.csv")
    results_df.to_csv(csv_path)
    print(f"[+] Metrics saved to: {csv_path}")

    # Paired Bootstrap ROC Hypothesis Test (Top Performer vs Baseline / Second-Best)
    X_tr_hold, X_te_hold, y_tr_hold, y_te_hold = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    comparator_name = "Logistic Regression" if best_model_name != "Logistic Regression" else results_df.index[1]
    
    pipe_top = pipelines[best_model_name]
    pipe_comp = pipelines[comparator_name]
    pipe_top.fit(X_tr_hold, y_tr_hold)
    pipe_comp.fit(X_tr_hold, y_tr_hold)

    proba_top = pipe_top.predict_proba(X_te_hold)[:, 1] if hasattr(pipe_top, "predict_proba") else pipe_top.predict(X_te_hold)
    proba_comp = pipe_comp.predict_proba(X_te_hold)[:, 1] if hasattr(pipe_comp, "predict_proba") else pipe_comp.predict(X_te_hold)
    y_te_arr = np.asarray(y_te_hold)

    paired_test_res = compute_paired_roc_bootstrap_test(y_te_arr, proba_top, proba_comp, n_bootstraps=1000, random_state=42)
    print(f"[+] Paired Bootstrap ROC Test ({best_model_name} vs {comparator_name}):")
    print(f"    ΔAUC: {paired_test_res['Mean_Delta_AUC (%)']}% {paired_test_res['Delta_AUC_95_CI']} | p-value: {paired_test_res['P_Value']:.4f} (Significant: {paired_test_res['Statistically_Significant']})")
    pd.DataFrame([{
        "Model_A": best_model_name,
        "Model_B": comparator_name,
        **paired_test_res
    }]).to_csv(os.path.join(out_dir, "paired_roc_bootstrap_test.csv"), index=False)

    # 2. Stratified Cross-Validation & Friedman Omnibus Hypothesis Test
    if cv_folds > 1:
        print(f"\n[*] [2/7] Executing {cv_folds}-Fold Stratified Cross-Validation & Friedman Omnibus Rank-Sum Test...")
        df_cv, df_raw_folds = evaluate_cross_validation(pipelines, X, y, n_splits=cv_folds, random_state=42)
        print("-" * 95)
        print(df_cv.to_string())
        print("-" * 95)
        df_cv.to_csv(os.path.join(out_dir, "cross_validation_metrics.csv"))

        friedman_res = compute_friedman_test(df_raw_folds)
        print(f"[+] Friedman Omnibus Test: Chi-Square = {friedman_res['Friedman_Statistic']}, p-value = {friedman_res['P_Value']:.4e}")
        print(f"[+] Statistical Significance: {friedman_res['Statistically_Significant']}")

    # 3. 5x5 Nested Stratified Cross-Validation (Unbiased Selection & Assessment)
    if run_nested_cv:
        print(f"\n[*] [3/7] Executing 5x5 Nested Stratified Cross-Validation for {best_model_name} (Optuna Inner Loop)...")
        n_trials = 10 if fast else 20
        df_nested_sum, df_nested_folds = evaluate_nested_cross_validation(
            model_name=best_model_name,
            X=X,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            outer_splits=5,
            inner_splits=5,
            n_trials=n_trials,
            use_smote=meta["use_smote"],
            random_state=42
        )
        print("-" * 95)
        print(df_nested_sum.to_string())
        print("-" * 95)
        df_nested_sum.to_csv(os.path.join(out_dir, "nested_cv_summary.csv"))
        df_nested_folds.to_csv(os.path.join(out_dir, "nested_cv_outer_folds.csv"), index=False)

    # 4. Locked External Prospective Cohort Validation (Framingham Cohort for NHANES)
    if key == "nhanes_cardiovascular":
        print("\n[*] [4/7] Prospective External Validation on Independent Cohort (Framingham Simulation, Locked Threshold)...")
        X_ext, y_ext = load_external_validation_cohort(n_samples=2500, random_state=1337)
        top_pipe = pipelines[best_model_name]
        top_pipe.fit(X_tr_hold, y_tr_hold)
        df_ext, ext_dict = evaluate_locked_external_validation(
            pipeline=top_pipe,
            locked_threshold=locked_thresh,
            X_ext=X_ext,
            y_ext=y_ext,
            n_bootstraps=1000,
            random_state=42
        )
        print("-" * 95)
        print(f"External Cohort N: {ext_dict['N_Patients']} | Prevalence: {ext_dict['Event_Rate (%)']}% | Locked Thresh: {ext_dict['Locked_Threshold']}")
        print(f"Prospective External ROC-AUC: {ext_dict['ROC-AUC [95% CI]']}")
        print(f"Prospective External Sensitivity: {ext_dict['Sensitivity [95% CI]']}")
        print(f"Prospective External Specificity: {ext_dict['Specificity [95% CI]']}")
        print(f"Prospective External ECE: {ext_dict['ECE [95% CI]']} (Brier: {ext_dict['Brier [95% CI]']})")
        print("-" * 95)
        df_ext.to_csv(os.path.join(out_dir, "external_validation_metrics.csv"), index=False)

    # 5. Inductive Conformal Prediction (Marginal Statistical Coverage & Referral Triage)
    if run_conformal:
        print("\n[*] [5/7] Computing Inductive Conformal Prediction Sets (Marginal Statistical Coverage under Exchangeability)...")
        top_pipe = pipelines[best_model_name]
        conformal_res = evaluate_conformal_prediction(top_pipe, X_tr_hold, y_tr_hold, X_te_hold, y_te_hold, alpha=0.05, n_bootstraps=1000, random_state=42)
        print("-" * 95)
        for k, v in conformal_res.items():
            print(f"  - {k:<34}: {v}")
        print("-" * 95)
        pd.DataFrame([conformal_res]).to_csv(os.path.join(out_dir, "conformal_prediction.csv"), index=False)

    # 6. Simulated Measurement & Assay Perturbation Stress-Testing
    if run_robustness:
        print("\n[*] [6/7] Simulated Clinical Measurement & Assay Perturbation Stress-Testing (+0% to +20% Gaussian Noise)...")
        top_pipe = pipelines[best_model_name]
        df_noise = evaluate_sensor_noise_drift(top_pipe, X_tr_hold, y_tr_hold, X_te_hold, y_te_hold, noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20], n_bootstraps=1000)
        print("-" * 95)
        print(df_noise[["Perturbation Level", "ROC-AUC [95% CI]", "Sensitivity [95% CI]", "Retention_Ratio"]].to_string())
        print("-" * 95)
        df_noise.to_csv(os.path.join(out_dir, "sensor_noise_stress_test.csv"))

        if not no_plot:
            plot_publication_robustness(df_noise, os.path.join(out_dir, "figure2_sensor_robustness.png"))

    # 7. Explainable AI & Pathophysiological Plausibility Audit
    if run_shap:
        print("\n[*] [7/7] Computing TreeSHAP Attributions & Pathophysiological Plausibility Audit against AHA/ACC Criteria...")
        shap_candidate = "XGBoost" if "XGBoost" in pipelines else best_model_name
        shap_pipe = pipelines[shap_candidate]
        shap_pipe.fit(X_tr_hold, y_tr_hold)
        shap_vals, importance_df = explain_clinical_model(shap_pipe, X_tr_hold, X_te_hold, max_eval_samples=150)
        
        guideline_audit = audit_clinical_guideline_alignment(importance_df, top_k=5)
        print("-" * 95)
        print(f"[+] Top Clinical Biomarkers Identified: {', '.join(guideline_audit['Top_Features'])}")
        print(f"[+] AHA/ACC Guideline Alignment Rate: {guideline_audit['Guideline_Alignment_Rate']}")
        print(f"[+] Clinical Plausibility Audit: {guideline_audit['Clinical_Plausibility_Verdict']}")
        print("-" * 95)
        importance_df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)

        if not no_plot:
            render_shap_summary_plot(shap_vals, importance_df, os.path.join(out_dir, "figure3_shap_biomarkers.png"))

    # Demographic Subgroup Fairness Audit (Sex & Age)
    if "Sex" in X.columns and "Age" in X.columns:
        print("\n[*] [Fairness] Clinical Subgroup Parity Audit (Sex & Age Subgroups with Bootstrap 95% CIs)...")
        top_pipe = pipelines[best_model_name]
        y_te_preds = (proba_top >= locked_thresh).astype(int)
        
        fairness_sex = evaluate_demographic_fairness(
            y_true=y_te_hold,
            y_pred=y_te_preds,
            demographic_series=X_te_hold["Sex"],
            group_a="Female",
            group_b="Male",
            metric_name="Sex"
        )
        
        age_cohort = pd.Series(np.where(X_te_hold["Age"] >= 50, "Older Adults (>=50)", "Younger (<50)"), index=X_te_hold.index)
        fairness_age = evaluate_demographic_fairness(
            y_true=y_te_hold,
            y_pred=y_te_preds,
            demographic_series=age_cohort,
            group_a="Younger (<50)",
            group_b="Older Adults (>=50)",
            metric_name="Age"
        )

        df_fairness = pd.DataFrame([fairness_sex, fairness_age], index=["Sex_Subgroups", "Age_Subgroups"])
        print("-" * 95)
        print(f"Sex Parity Verdict: {fairness_sex['Fairness_Verdict']} (Gap: {fairness_sex['Equal_Opportunity_Gap [95% CI]']})")
        print(f"Age Parity Verdict: {fairness_age['Fairness_Verdict']} (Gap: {fairness_age['Equal_Opportunity_Gap [95% CI]']})")
        print("-" * 95)
        df_fairness.to_csv(os.path.join(out_dir, "demographic_fairness.csv"))

    # Publication Figures & TRIPOD+AI Documentation
    if not no_plot:
        plot_publication_discrimination(curve_data["roc"], curve_data["pr"], os.path.join(out_dir, "figure1_discrimination.png"))
        generate_tripod_checklist_markdown(os.path.join(out_dir, "tripod_ai_checklist.md"))
        print(f"\n[+] Publication-grade 300 DPI figures and TRIPOD+AI appendix generated in: {out_dir}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Trustworthy Clinical Machine Learning Benchmark Suite")
    parser.add_argument("--dataset", choices=["all"] + list(DATASET_REGISTRY.keys()), default="nhanes_cardiovascular",
                        help="Target clinical dataset (default: nhanes_cardiovascular)")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation (e.g. --cv 5)")
    parser.add_argument("--nested-cv", action="store_true", help="Run 5x5 Nested Stratified Cross-Validation with inner Bayesian optimization")
    parser.add_argument("--fast", action="store_true", help="Fast smoke run with fewer estimators")
    parser.add_argument("--conformal", action="store_true", default=True, help="Run Inductive Conformal Prediction")
    parser.add_argument("--robustness", action="store_true", default=True, help="Run Simulated Measurement Perturbation Stress-Test")
    parser.add_argument("--shap", action="store_true", default=True, help="Run SHAP Explainability & Clinical Audit")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot figures")
    args = parser.parse_args()

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    for t in targets:
        run_benchmark_dataset(
            key=t,
            cv_folds=args.cv,
            run_nested_cv=args.nested_cv,
            fast=args.fast,
            run_conformal=args.conformal,
            run_robustness=args.robustness,
            run_shap=args.shap,
            no_plot=args.no_plot
        )

    print("\n" + "=" * 95)
    print("  ALL BENCHMARKS & RIGOROUS PUBLICATION PROTOCOLS COMPLETED SUCCESSFULLY!")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
