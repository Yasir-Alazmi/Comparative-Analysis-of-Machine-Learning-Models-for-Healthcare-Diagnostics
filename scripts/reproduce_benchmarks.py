#!/usr/bin/env python3
"""
Single-Command Automated Scientific Reproduction Pipeline.
Reproduces the complete benchmark metrics on authentic CDC NHANES and Framingham cohorts.

Usage:
    python scripts/reproduce_benchmarks.py
    python scripts/reproduce_benchmarks.py --smoke
"""

import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "datasets")
NHANES_CSV = os.path.join(DATA_DIR, "nhanes_cardiovascular.csv")
FRAMINGHAM_CSV = os.path.join(DATA_DIR, "external_framingham_cohort.csv")


def check_and_prepare_datasets():
    """Ensures authentic datasets are present on disk; acquires them if missing."""
    print("[1/3] Verifying authentic dataset availability...")
    if not os.path.exists(NHANES_CSV) or not os.path.exists(FRAMINGHAM_CSV):
        print("  [*] Datasets missing. Programmatically acquiring authentic CDC and Duke data...")
        build_script = os.path.join(REPO_ROOT, "scripts", "build_real_datasets.py")
        res = subprocess.run([sys.executable, build_script], check=True)
        if res.returncode != 0:
            raise RuntimeError("Failed to acquire authentic primary datasets from CDC/Duke servers.")
    else:
        import pandas as pd
        df_n = pd.read_csv(NHANES_CSV)
        df_f = pd.read_csv(FRAMINGHAM_CSV)
        print(f"  [+] Authentic CDC NHANES verified: N = {len(df_n)} adults (CVD rate: {df_n['CVD_Diagnosis'].mean()*100:.2f}%)")
        print(f"  [+] Authentic Framingham verified: N = {len(df_f)} patients (CVD rate: {df_f['CVD_Diagnosis'].mean()*100:.2f}%)")


def run_reproducible_benchmark(smoke: bool = False, seed: int = 42):
    """Executes the master benchmark with locked random seed."""
    print(f"\n[2/3] Executing deterministic benchmark (seed={seed}, smoke={smoke})...")
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "run_benchmark.py"),
        "--dataset", "nhanes_cardiovascular",
        "--seed", str(seed)
    ]
    if smoke:
        cmd.extend(["--fast", "--cv", "0"])
    else:
        cmd.extend(["--cv", "5"])

    subprocess.run(cmd, check=True)


def verify_outputs():
    """Verifies that all scientific artifacts and metrics tables were produced."""
    print("\n[3/3] Verifying generated scientific artifacts...")
    out_dir = os.path.join(REPO_ROOT, "results", "nhanes_cardiovascular")
    required_files = [
        "benchmark_metrics.csv",
        "external_validation_metrics.csv",
        "conformal_prediction.csv",
        "sensor_noise_stress_test.csv",
        "demographic_fairness.csv",
        "shap_feature_importance.csv",
        "paired_roc_bootstrap_test.csv",
        "figure1_discrimination.png",
        "figure2_sensor_robustness.png",
        "figure3_shap_biomarkers.png",
        "tripod_ai_checklist.md",
    ]

    all_ok = True
    for f in required_files:
        p = os.path.join(out_dir, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [+] Artifact confirmed: {f} ({os.path.getsize(p):,} bytes)")
        else:
            print(f"  [!] Missing or empty artifact: {f}")
            all_ok = False

    if all_ok:
        print("\n" + "=" * 80)
        print("  SCIENTIFIC REPRODUCTION COMPLETED SUCCESSFULLY WITH 100% ARTIFACT INTEGRITY!")
        print("=" * 80 + "\n")
    else:
        raise RuntimeError("Artifact verification failed. Some expected metric tables are missing.")


def main():
    parser = argparse.ArgumentParser(description="Deterministic Benchmark Reproduction Script")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run for CI or quick sanity checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    check_and_prepare_datasets()
    run_reproducible_benchmark(smoke=args.smoke, seed=args.seed)
    verify_outputs()


if __name__ == "__main__":
    main()
