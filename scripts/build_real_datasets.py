"""
Real-World Clinical Data Ingestion Script.
Downloads and processes:
1. Authentic CDC NHANES continuous survey cohorts (Cycles 2015-2016 and 2017-2018):
   - Merges demographics, blood pressure, anthropometrics, lipid panels, glycemic markers,
     renal panel, smoking status, and physician-diagnosed cardiovascular condition outcomes (N > 11,000 adults).
2. Authentic Framingham Heart Study longitudinal cohort (N = 4,240 real patients with 10-year CVD outcome).
"""

import os
import io
import requests
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_cdc_xpt(cycle_year: int, filename: str) -> pd.DataFrame:
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{cycle_year}/DataFiles/{filename}.xpt"
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url}: status {r.status_code}")
    return pd.read_sas(io.BytesIO(r.content), format='xport')


def extract_real_nhanes_cycle(cycle_year: int, suffix: str) -> pd.DataFrame:
    print(f"[*] Downloading real CDC NHANES {cycle_year} (suffix _{suffix})...")
    demo = fetch_cdc_xpt(cycle_year, f"DEMO_{suffix}")
    bpx = fetch_cdc_xpt(cycle_year, f"BPX_{suffix}")
    bmx = fetch_cdc_xpt(cycle_year, f"BMX_{suffix}")
    tchol = fetch_cdc_xpt(cycle_year, f"TCHOL_{suffix}")
    hdl = fetch_cdc_xpt(cycle_year, f"HDL_{suffix}")
    trig = fetch_cdc_xpt(cycle_year, f"TRIGLY_{suffix}")
    glu = fetch_cdc_xpt(cycle_year, f"GLU_{suffix}")
    ghb = fetch_cdc_xpt(cycle_year, f"GHB_{suffix}")
    biopro = fetch_cdc_xpt(cycle_year, f"BIOPRO_{suffix}")
    smq = fetch_cdc_xpt(cycle_year, f"SMQ_{suffix}")
    mcq = fetch_cdc_xpt(cycle_year, f"MCQ_{suffix}")

    # Merge on participant SEQN
    dfs = [demo, bpx, bmx, tchol, hdl, trig, glu, ghb, biopro, smq, mcq]
    merged = dfs[0]
    for d in dfs[1:]:
        cols_to_use = [c for c in d.columns if c not in merged.columns or c == 'SEQN']
        merged = merged.merge(d[cols_to_use], on='SEQN', how='left')

    # Adult participant filter (Age >= 20)
    adults = merged[merged['RIDAGEYR'] >= 20].copy()

    age = adults['RIDAGEYR'].values
    sex = np.where(adults['RIAGENDR'] == 1.0, 'Male', 'Female')

    race_map = {
        1.0: 'Mexican American',
        2.0: 'Other Hispanic',
        3.0: 'Non-Hispanic White',
        4.0: 'Non-Hispanic Black',
        6.0: 'Non-Hispanic Asian',
        7.0: 'Other Race'
    }
    race = adults['RIDRETH3'].map(race_map).fillna('Other Race').values

    # Smoking status
    smk_100 = adults.get('SMQ020', pd.Series(index=adults.index, dtype=float))
    smk_now = adults.get('SMQ040', pd.Series(index=adults.index, dtype=float))
    smoking = np.where(
        smk_now.isin([1.0, 2.0]), 'Current',
        np.where(smk_100 == 1.0, 'Former', 'Never')
    )

    # Blood Pressure readings (mean of 3 serial physician measurements)
    sbp_cols = [c for c in ['BPXSY1', 'BPXSY2', 'BPXSY3'] if c in adults.columns]
    dbp_cols = [c for c in ['BPXDI1', 'BPXDI2', 'BPXDI3'] if c in adults.columns]
    sbp = adults[sbp_cols].mean(axis=1).values
    dbp = adults[dbp_cols].mean(axis=1).values

    # Anthropometrics
    bmi = adults.get('BMXBMI', pd.Series(index=adults.index, dtype=float)).values
    waist = adults.get('BMXWAIST', pd.Series(index=adults.index, dtype=float)).values
    height = adults.get('BMXHT', pd.Series(index=adults.index, dtype=float)).values

    # Laboratory Panel
    tot_chol = adults.get('LBXTC', pd.Series(index=adults.index, dtype=float)).values
    hdl_val = adults.get('LBDHDD', pd.Series(index=adults.index, dtype=float)).values
    trig_val = adults.get('LBXTR', pd.Series(index=adults.index, dtype=float)).values
    ldl_val = adults.get('LBDLDL', pd.Series(index=adults.index, dtype=float)).values
    glucose = adults.get('LBXGLU', pd.Series(index=adults.index, dtype=float)).values
    hba1c = adults.get('LBXGH', pd.Series(index=adults.index, dtype=float)).values
    serum_creatinine = adults.get('LBXSCR', pd.Series(index=adults.index, dtype=float)).values
    bun = adults.get('LBXSBU', pd.Series(index=adults.index, dtype=float)).values

    # Fallback for LDL if missing but Total, HDL, Trig are present (Friedewald equation)
    friedewald_ldl = tot_chol - hdl_val - (trig_val / 5.0)
    ldl_combined = np.where(np.isnan(ldl_val), friedewald_ldl, ldl_val)

    # Cardiovascular Composite Endpoint (Ever diagnosed with CHF, CHD, Angina, Heart Attack, or Stroke)
    cvd_cols = [c for c in ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F'] if c in adults.columns]
    has_cvd = (adults[cvd_cols] == 1.0).any(axis=1).astype(int)

    df_out = pd.DataFrame({
        "Age": np.round(age, 1),
        "Sex": sex,
        "Race": race,
        "Smoking": smoking,
        "BMI": np.round(bmi, 2),
        "Waist_Circumference": np.round(waist, 1),
        "Standing_Height": np.round(height, 1),
        "Systolic_BP": np.round(sbp, 1),
        "Diastolic_BP": np.round(dbp, 1),
        "Fasting_Glucose": np.round(glucose, 1),
        "HbA1c": np.round(hba1c, 2),
        "Total_Cholesterol": np.round(tot_chol, 1),
        "HDL_Cholesterol": np.round(hdl_val, 1),
        "LDL_Cholesterol": np.round(ldl_combined, 1),
        "Triglycerides": np.round(trig_val, 1),
        "Serum_Creatinine": np.round(serum_creatinine, 2),
        "BUN": np.round(bun, 1),
        "CVD_Diagnosis": has_cvd.values
    })
    return df_out


def build_real_nhanes_cohort():
    print("\n" + "=" * 80)
    print("  INGESTING AUTHENTIC CDC NHANES DATASETS (Cycles 2015-2016 & 2017-2018)")
    print("=" * 80)
    df_j = extract_real_nhanes_cycle(2017, "J")
    df_i = extract_real_nhanes_cycle(2015, "I")
    combined = pd.concat([df_j, df_i], ignore_index=True)

    nhanes_path = os.path.join(DATA_DIR, "nhanes_cardiovascular.csv")
    combined.to_csv(nhanes_path, index=False)
    print(f"[+] Real CDC NHANES dataset successfully created: {nhanes_path}")
    print(f"[+] Total Real Adults: {len(combined)} | True CVD Cases: {combined['CVD_Diagnosis'].sum()} ({combined['CVD_Diagnosis'].mean() * 100:.2f}%)")
    return combined


def build_real_framingham_cohort():
    print("\n" + "=" * 80)
    print("  INGESTING AUTHENTIC FRAMINGHAM HEART STUDY COHORT")
    print("=" * 80)
    framingham_url = "https://raw.githubusercontent.com/sta210-sp20/datasets/master/framingham.csv"
    print(f"[*] Fetching authentic Framingham data from: {framingham_url}...")
    df_raw = pd.read_csv(framingham_url)
    print(f"[*] Raw Framingham rows: {len(df_raw)}")

    # Harmonize features to align with clinical benchmark naming
    df_harm = pd.DataFrame({
        "Age": df_raw["age"].astype(float),
        "Sex": np.where(df_raw["male"] == 1, "Male", "Female"),
        "Smoking": np.where(df_raw["currentSmoker"] == 1, "Current", "Never"),
        "BMI": df_raw["BMI"],
        "Systolic_BP": df_raw["sysBP"],
        "Diastolic_BP": df_raw["diaBP"],
        "Total_Cholesterol": df_raw["totChol"],
        "Fasting_Glucose": df_raw["glucose"],
        "HeartRate": df_raw["heartRate"],
        "CVD_Diagnosis": df_raw["TenYearCHD"].astype(int)
    })

    framingham_path = os.path.join(DATA_DIR, "external_framingham_cohort.csv")
    df_harm.to_csv(framingham_path, index=False)
    print(f"[+] Real Framingham dataset successfully created: {framingham_path}")
    print(f"[+] Total Real Patients: {len(df_harm)} | True 10-Year CVD Cases: {df_harm['CVD_Diagnosis'].sum()} ({df_harm['CVD_Diagnosis'].mean() * 100:.2f}%)")
    return df_harm


if __name__ == "__main__":
    build_real_nhanes_cohort()
    build_real_framingham_cohort()
