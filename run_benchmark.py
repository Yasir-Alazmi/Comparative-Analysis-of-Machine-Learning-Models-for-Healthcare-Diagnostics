#!/usr/bin/env python3
"""
CLI Runner for the Medical Healthcare Diagnostics Benchmark Suite.

Usage:
    python run_benchmark.py --dataset all
    python run_benchmark.py --dataset breast_cancer
    python run_benchmark.py --dataset chronic_kidney
    python run_benchmark.py --dataset heart_failure
    python run_benchmark.py --dataset pima_diabetes
    python run_benchmark.py --dataset stroke_prediction
    python run_benchmark.py --fast  (fast smoke run with fewer estimators)
"""

import os
import sys
import argparse

# Reconfigure stdout for Windows console UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models import build_models
from src.preprocessor import DATASET_REGISTRY
from src.evaluator import evaluate_models, evaluate_cross_validation
from src.visualizer import plot_bar_chart, plot_heatmap, plot_radar_chart, plot_roc_curves


def run_single_dataset(key: str, fast: bool = False, no_plot: bool = False):
    meta = DATASET_REGISTRY[key]
    print("\n" + "=" * 80)
    print(f"  BENCHMARK: {meta['name'].upper()}")
    print("=" * 80)

    print(f"[*] Loading and preprocessing dataset: {key}...")
    X, y = meta["loader"]()
    print(f"[*] Samples: {len(X)}, Features: {X.shape[1]}, Classes: {y.nunique()}")

    print("[*] Building 10 Machine Learning Pipelines...")
    pipelines = build_models(random_state=42, fast_mode=fast)

    print("[*] Training and Evaluating models...")
    results_df, roc_data = evaluate_models(
        pipelines,
        X,
        y,
        use_smote=meta["use_smote"],
        sort_metric=meta["sort_metric"]
    )

    print("\n" + "-" * 80)
    print(f"{'Algorithm':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for alg, row in results_df.iterrows():
        print(f"{str(alg):<24} {row['Accuracy']:>9.2f}% {row['Precision']:>9.2f}% {row['Recall']:>9.2f}% {row['F1-Score']:>9.2f}% {row['ROC-AUC']:>9.2f}%")
    print("-" * 80)

    best = results_df.index[0]
    print(f"[+] Top Performer ({meta['sort_metric']}): {best} ({results_df.loc[best, meta['sort_metric']]:.2f}%)")

    # Save results and plots
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", key)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "metrics.csv")
    results_df.to_csv(csv_path)
    print(f"[+] Metrics saved to: {csv_path}")

    if not no_plot:
        print("[*] Generating visualizations...")
        plot_bar_chart(results_df, os.path.join(out_dir, "bar_chart.png"), f"{meta['name']} - Accuracy")
        plot_heatmap(results_df, os.path.join(out_dir, "heatmap.png"), f"{meta['name']} - Metrics Heatmap")
        plot_radar_chart(results_df, os.path.join(out_dir, "radar_chart.png"), f"{meta['name']} - Radar Analysis")
        plot_roc_curves(roc_data, os.path.join(out_dir, "roc_curves.png"), f"{meta['name']} - ROC Curves")
        print(f"[+] Visualizations saved to: {out_dir}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Healthcare Diagnostics ML Comparative Suite")
    parser.add_argument("--dataset", choices=["all"] + list(DATASET_REGISTRY.keys()), default="all",
                        help="Choose specific medical dataset or 'all'")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation instead of single split (e.g. --cv 5)")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (reduced estimators for testing)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot images")
    args = parser.parse_args()

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    for t in targets:
        run_single_dataset(t, fast=args.fast, no_plot=args.no_plot)

    print("\n" + "=" * 80)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
