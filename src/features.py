"""
Clinical Feature Engineering Module.
Computes evidence-based physiological and metabolic indices from raw clinical measurements:
1. Atherogenic Index of Plasma (AIP) = log10(TG / HDL)
2. Mean Arterial Pressure (MAP) = DBP + (1/3) * (SBP - DBP)
3. Pulse Pressure (PP) = SBP - DBP
4. Triglyceride-Glucose Index (TyG) = ln((TG * Glucose) / 2)
5. Cholesterol Ratios: Total/HDL, LDL/HDL
6. Estimated Glomerular Filtration Rate (eGFR) via CKD-EPI 2021 formula
7. Non-HDL Cholesterol = Total Cholesterol - HDL
8. Anthropometric Ratios: Waist-to-Height Ratio (WHtR)

All transformations are strictly row-wise (within-subject) mathematical mappings,
guaranteeing zero cross-patient data leakage.
"""

import numpy as np
import pandas as pd


def compute_atherogenic_index_of_plasma(triglycerides: pd.Series, hdl: pd.Series) -> pd.Series:
    """
    Computes Atherogenic Index of Plasma (AIP) = log10(TG / HDL).
    TG and HDL in mg/dL are converted to mmol/L internally (TG / 88.57, HDL / 38.67).
    AIP < 0.11: Low CVD risk
    AIP 0.11 - 0.21: Intermediate risk
    AIP > 0.21: High CVD risk
    """
    tg_safe = triglycerides.clip(lower=1.0)
    hdl_safe = hdl.clip(lower=1.0)
    # Convert mg/dL to mmol/L for standardized AIP calculation
    tg_mmol = tg_safe / 88.57
    hdl_mmol = hdl_safe / 38.67
    ratio = (tg_mmol / hdl_mmol).clip(lower=1e-5)
    return np.log10(ratio)


def compute_mean_arterial_pressure(sbp: pd.Series, dbp: pd.Series) -> pd.Series:
    """
    Computes Mean Arterial Pressure (MAP) = DBP + (1/3) * (SBP - DBP).
    Normal resting MAP is typically between 70 and 100 mmHg.
    """
    return dbp + (1.0 / 3.0) * (sbp - dbp)


def compute_pulse_pressure(sbp: pd.Series, dbp: pd.Series) -> pd.Series:
    """
    Computes Pulse Pressure (PP) = SBP - DBP.
    Elevated PP (> 50-60 mmHg) is a strong independent predictor of arterial stiffness and cardiovascular risk.
    """
    return sbp - dbp


def compute_tyg_index(triglycerides: pd.Series, fasting_glucose: pd.Series) -> pd.Series:
    """
    Computes Triglyceride-Glucose Index (TyG) = ln((TG [mg/dL] * FPG [mg/dL]) / 2).
    A validated surrogate marker of insulin resistance and cardiometabolic syndrome.
    """
    tg_safe = triglycerides.clip(lower=1.0)
    glu_safe = fasting_glucose.clip(lower=1.0)
    product = (tg_safe * glu_safe) / 2.0
    return np.log(product.clip(lower=1e-5))


def compute_cholesterol_ratios(total_chol: pd.Series, hdl: pd.Series, ldl: pd.Series = None) -> pd.DataFrame:
    """
    Computes Total/HDL (Castelli Risk Index I) and LDL/HDL (Castelli Risk Index II).
    """
    hdl_safe = hdl.clip(lower=1.0)
    df_ratios = pd.DataFrame(index=total_chol.index)
    df_ratios["Total_HDL_Ratio"] = total_chol / hdl_safe
    if ldl is not None:
        df_ratios["LDL_HDL_Ratio"] = ldl / hdl_safe
    return df_ratios


def compute_egfr_ckdepi_2021(serum_creatinine: pd.Series, age: pd.Series, is_female: pd.Series) -> pd.Series:
    """
    Computes Estimated Glomerular Filtration Rate (eGFR) via the 2021 CKD-EPI Creatinine Equation (Race-Free).
    Scr in mg/dL, Age in years.
    Female: kappa = 0.7, alpha = -0.241
    Male:   kappa = 0.9, alpha = -0.302
    """
    kappa = np.where(is_female == 1, 0.7, 0.9)
    alpha = np.where(is_female == 1, -0.241, -0.302)
    female_factor = np.where(is_female == 1, 1.012, 1.0)

    scr_safe = serum_creatinine.clip(lower=0.1)
    ratio = scr_safe / kappa
    min_ratio = np.minimum(ratio, 1.0)
    max_ratio = np.maximum(ratio, 1.0)

    egfr = 142.0 * (min_ratio ** alpha) * (max_ratio ** (-1.200)) * (0.9938 ** age) * female_factor
    return pd.Series(egfr, index=serum_creatinine.index, name="eGFR")


def engineer_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detects available clinical measurements and derives all feasible physiological indices.
    """
    out = df.copy()
    cols_lower = {col.lower(): col for col in out.columns}

    # 1. Blood pressure indices
    sbp_col = (
        cols_lower.get("sbp")
        or cols_lower.get("systolic_bp")
        or cols_lower.get("restingbp")
        or cols_lower.get("bloodpressure")
    )
    dbp_col = cols_lower.get("dbp") or cols_lower.get("diastolic_bp")
    if sbp_col and dbp_col:
        out["MAP"] = compute_mean_arterial_pressure(out[sbp_col], out[dbp_col])
        out["Pulse_Pressure"] = compute_pulse_pressure(out[sbp_col], out[dbp_col])

    # 2. Lipid and glycemic indices
    tg_col = cols_lower.get("triglycerides") or cols_lower.get("tg")
    hdl_col = cols_lower.get("hdl") or cols_lower.get("hdl_cholesterol")
    glu_col = cols_lower.get("glucose") or cols_lower.get("fasting_glucose") or cols_lower.get("fpg") or cols_lower.get("avg_glucose_level")
    tot_chol = cols_lower.get("cholesterol") or cols_lower.get("tot_chol") or cols_lower.get("total_cholesterol")
    ldl_col = cols_lower.get("ldl") or cols_lower.get("ldl_cholesterol")

    if tg_col and hdl_col:
        out["AIP"] = compute_atherogenic_index_of_plasma(out[tg_col], out[hdl_col])

    if tg_col and glu_col:
        out["TyG_Index"] = compute_tyg_index(out[tg_col], out[glu_col])

    if tot_chol and hdl_col:
        out["Total_HDL_Ratio"] = out[tot_chol] / out[hdl_col].clip(lower=1.0)
        out["Non_HDL_Cholesterol"] = out[tot_chol] - out[hdl_col]

    if ldl_col and hdl_col:
        out["LDL_HDL_Ratio"] = out[ldl_col] / out[hdl_col].clip(lower=1.0)

    # 3. Renal function (eGFR)
    cr_col = cols_lower.get("creatinine") or cols_lower.get("serum_creatinine") or cols_lower.get("sc")
    age_col = cols_lower.get("age")
    sex_col = cols_lower.get("sex") or cols_lower.get("gender")

    if cr_col and age_col and sex_col:
        # Determine female indicator
        if out[sex_col].dtype == object:
            is_female = out[sex_col].astype(str).str.lower().str.startswith("f").astype(int)
        else:
            is_female = (out[sex_col] == 0).astype(int)  # Many datasets code 1=male, 0=female
        out["eGFR"] = compute_egfr_ckdepi_2021(out[cr_col], out[age_col], is_female)

    # 4. Anthropometric ratios
    waist_col = cols_lower.get("waist") or cols_lower.get("waist_circumference")
    height_col = cols_lower.get("height")
    if waist_col and height_col:
        out["Waist_Height_Ratio"] = out[waist_col] / out[height_col].clip(lower=50.0)

    return out
