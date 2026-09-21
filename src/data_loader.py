"""
Data Loader & Clinical Cohort Acquisition Engine.
Provides standard access to:
1. CDC NHANES (National Health and Nutrition Examination Survey) Continuous Cardiovascular Cohort
2. Framingham Heart Study / External Validation Cohort
3. Integration with benchmark clinical datasets with domain feature engineering.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from src.features import engineer_clinical_features

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def generate_synthetic_demo_cohort(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    [OFFLINE CI / SMOKE-TEST ONLY]
    WARNING: DO NOT USE FOR SCIENTIFIC BENCHMARKING OR PUBLICATION CLAIMS.
    Generates a synthetic mock tabular cohort strictly for automated offline testing when
    remote CDC SAS XPT data are unavailable in CI sandboxes.
    """
    rng = np.random.default_rng(random_state)

    # 1. Demographics
    age = rng.uniform(20.0, 80.0, size=n_samples)
    sex = rng.choice(["Male", "Female"], size=n_samples, p=[0.49, 0.51])
    is_female = (sex == "Female").astype(int)
    race = rng.choice(
        ["Non-Hispanic White", "Non-Hispanic Black", "Mexican American", "Other Hispanic", "Asian / Other"],
        size=n_samples,
        p=[0.36, 0.22, 0.15, 0.12, 0.15]
    )
    smoking = rng.choice(["Never", "Former", "Current"], size=n_samples, p=[0.55, 0.25, 0.20])
    smoker_score = np.where(smoking == "Current", 1.5, np.where(smoking == "Former", 0.5, 0.0))

    # 2. Anthropometrics
    bmi_base = rng.lognormal(mean=3.35, sigma=0.22, size=n_samples)  # Mean ~29.5 kg/m2
    bmi = np.clip(bmi_base + (age - 20) * 0.04, 16.0, 60.0)
    waist = bmi * 2.8 + rng.normal(0, 6.0, size=n_samples) + (1 - is_female) * 5.0
    height = np.where(is_female == 1, rng.normal(162.0, 6.5, n_samples), rng.normal(175.5, 7.0, n_samples))

    # 3. Hemodynamics (Blood Pressure correlated with Age & BMI)
    sbp_latent = 112.0 + 0.52 * (age - 20) + 0.65 * (bmi - 25) + rng.normal(0, 11.0, n_samples)
    sbp = np.clip(sbp_latent, 85.0, 220.0)
    dbp_latent = 72.0 + 0.18 * (age - 20) + 0.35 * (bmi - 25) - np.maximum(0, (age - 60) * 0.25) + rng.normal(0, 8.0, n_samples)
    dbp = np.clip(dbp_latent, 45.0, 125.0)

    # 4. Metabolic & Glycemic Profile
    glucose_latent = 85.0 + 0.35 * (age - 20) + 0.85 * (bmi - 25) + rng.exponential(scale=12.0, size=n_samples)
    fasting_glucose = np.clip(glucose_latent, 65.0, 380.0)
    hba1c = np.clip(4.8 + (fasting_glucose - 85.0) * 0.024 + rng.normal(0, 0.35, n_samples), 4.2, 14.5)

    # 5. Lipid Profile
    tot_chol = np.clip(rng.normal(195.0, 38.0, n_samples) + (age - 20) * 0.3, 100.0, 420.0)
    hdl_latent = 55.0 - (bmi - 25) * 0.45 + is_female * 7.5 + rng.normal(0, 11.0, n_samples)
    hdl = np.clip(hdl_latent, 20.0, 120.0)
    trig_latent = np.exp(rng.normal(4.85, 0.48, n_samples)) + (bmi - 25) * 2.5
    triglycerides = np.clip(trig_latent, 35.0, 750.0)
    ldl = np.clip(tot_chol - hdl - (triglycerides / 5.0), 30.0, 310.0)

    # 6. Renal Panel
    scr_base = np.where(is_female == 1, rng.normal(0.82, 0.18, n_samples), rng.normal(1.05, 0.24, n_samples))
    serum_creatinine = np.clip(scr_base + np.maximum(0, (age - 55) * 0.008), 0.4, 6.5)
    bun = np.clip(13.0 + (age - 20) * 0.12 + (serum_creatinine - 0.9) * 8.0 + rng.normal(0, 3.5, n_samples), 4.0, 65.0)

    # 7. Ground-Truth Cardiovascular Outcome via ACC/AHA ASCVD Risk Formulation
    log_odds = (
        -3.4
        + 0.092 * (age - 50.0)
        + 0.042 * (sbp - 120.0)
        + 0.022 * (tot_chol - 190.0)
        - 0.048 * (hdl - 50.0)
        + 0.008 * (triglycerides - 140.0)
        + 0.65 * (hba1c - 5.5)
        + 0.35 * (serum_creatinine - 0.9) * 5.0
        + 1.40 * smoker_score
        + 0.70 * (1 - is_female)
        + 0.055 * (bmi - 26.0)
    )
    p_cvd = 1.0 / (1.0 + np.exp(-log_odds))
    y_cvd = rng.binomial(n=1, p=np.clip(p_cvd, 0.01, 0.98), size=n_samples)

    df = pd.DataFrame({
        "Age": np.round(age, 1),
        "Sex": sex,
        "Race": race,
        "Smoking": smoking,
        "BMI": np.round(bmi, 2),
        "Waist_Circumference": np.round(waist, 1),
        "Standing_Height": np.round(height, 1),
        "Systolic_BP": np.round(sbp, 1),
        "Diastolic_BP": np.round(dbp, 1),
        "Fasting_Glucose": np.round(fasting_glucose, 1),
        "HbA1c": np.round(hba1c, 2),
        "Total_Cholesterol": np.round(tot_chol, 1),
        "HDL_Cholesterol": np.round(hdl, 1),
        "LDL_Cholesterol": np.round(ldl, 1),
        "Triglycerides": np.round(triglycerides, 1),
        "Serum_Creatinine": np.round(serum_creatinine, 2),
        "BUN": np.round(bun, 1),
        "CVD_Diagnosis": y_cvd
    })

    return df


def load_nhanes_cardiovascular(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads authentic CDC NHANES Continuous Survey Cardiovascular Cohort (Cycles 2015-2018, N=11,288 adult participants).
    Strictly loads authentic CDC survey data. Raises FileNotFoundError if missing to prevent silent synthetic fallbacks.
    """
    default_nhanes = os.path.join(DEFAULT_DATA_DIR, "nhanes_cardiovascular.csv")
    if data_path is not None and not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Authentic CDC NHANES dataset not found at '{data_path}'. "
            "To ensure strict scientific reproducibility and claim integrity, no synthetic data fallback is permitted. "
            "Please acquire the authentic CDC survey files by executing: python scripts/build_real_datasets.py"
        )

    if data_path is None:
        data_path = default_nhanes

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        try:
            from scripts.build_real_datasets import build_real_nhanes_cohort
            df = build_real_nhanes_cohort()
        except Exception as exc:
            raise FileNotFoundError(
                f"Authentic CDC NHANES dataset not found at '{data_path}' and automated ingestion failed ({exc}). "
                "To ensure strict scientific reproducibility and claim integrity, no synthetic data fallback is permitted. "
                "Please acquire the authentic CDC survey files by executing: python scripts/build_real_datasets.py"
            ) from exc

    target_col = "CVD_Diagnosis"
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Apply row-wise clinical feature engineering (AIP, MAP, TyG, eGFR, etc.)
    X_engineered = engineer_clinical_features(X_raw)

    return X_engineered, y


def load_external_validation_cohort(data_path: str = None, n_samples: int = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the authentic Framingham Heart Study longitudinal cohort (N=4,240 real human patients with 10-year CVD outcome)
    to evaluate independent model transportability across cohorts without refitting.
    
    Target Endpoint: TenYearCHD (10-year incident coronary heart disease event).
    Missing laboratory assays are harmonized and imputed via the training pipeline median imputer.
    """
    default_framingham = os.path.join(DEFAULT_DATA_DIR, "external_framingham_cohort.csv")
    if data_path is not None and not os.path.exists(data_path):
        raise FileNotFoundError(f"Authentic Framingham external cohort not found at '{data_path}'.")

    if data_path is None:
        data_path = default_framingham

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        try:
            from scripts.build_real_datasets import build_real_framingham_cohort
            df = build_real_framingham_cohort()
        except Exception as exc:
            raise FileNotFoundError(
                f"Authentic Framingham external cohort not found at '{data_path}' ({exc}). "
                "Please execute: python scripts/build_real_datasets.py"
            ) from exc

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state) if random_state is not None else df.iloc[:n_samples]

    y = df["CVD_Diagnosis"]
    X = engineer_clinical_features(df.drop(columns=["CVD_Diagnosis"]))
    return X, y


def evaluate_locked_external_validation(*args, **kwargs):
    """
    Convenience proxy re-exporting evaluate_locked_external_validation from src.evaluator.
    """
    from src.evaluator import evaluate_locked_external_validation as _eval_ext
    return _eval_ext(*args, **kwargs)

