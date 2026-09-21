# 📋 Clinical Dataset Provenance & Data Governance

This document establishes the institutional origin, ethical approvals, variable dictionaries, transformation pipelines, and ground-truth clinical endpoint definitions for all datasets utilized in the benchmark.

---

## 1. Flagship Primary Cohort: CDC NHANES Continuous Survey

### 1.1 Institutional Authority & Governance
* **Primary Custodian:** National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC), U.S. Department of Health and Human Services.
* **Institutional URL:** [https://wwwn.cdc.gov/nchs/nhanes/](https://wwwn.cdc.gov/nchs/nhanes/)
* **Survey Design:** Continuous, stratified, multistage probability sampling designed to assess the health and nutritional status of the civilian, non-institutionalized U.S. population.
* **Ethical Review & Consent:** Protocols reviewed and approved by the NCHS Research Ethics Review Board (ERB Protocol #2011-17 and #2018-01). All participants provided documented informed consent prior to clinical examination and laboratory biospecimen collection.
* **Study Cycles Ingested:** 
  * Continuous Cycle 2015–2016 (Data Suffix `_I`)
  * Continuous Cycle 2017–2018 (Data Suffix `_J`)

### 1.2 Ingestion & Merging Architecture
Data are programmatically fetched in official SAS transport format (`.XPT`) directly from CDC servers via [`scripts/build_real_datasets.py`](../scripts/build_real_datasets.py) and merged across unique participant sequence identifiers (`SEQN`):

| NHANES Examination Component | SAS File Name (2015-16 / 2017-18) | Clinical Features Extracted |
| :--- | :--- | :--- |
| **Demographics** | `DEMO_I.xpt` / `DEMO_J.xpt` | Chronological Age (`RIDAGEYR`), Biological Sex (`RIAGENDR`), Race/Ethnicity (`RIDRETH3`) |
| **Blood Pressure Exam** | `BPX_I.xpt` / `BPX_J.xpt` | Mean of 3 consecutive physician oscillometric SBP (`BPXSY1-3`) and DBP (`BPXDI1-3`) readings |
| **Body Measures (Anthropometrics)** | `BMX_I.xpt` / `BMX_J.xpt` | Body Mass Index (`BMXBMI`), Waist Circumference (`BMXWAIST`), Standing Height (`BMXHT`) |
| **Lipid Profile (Total Cholesterol)** | `TCHOL_I.xpt` / `TCHOL_J.xpt` | Serum Total Cholesterol (`LBXTC`) |
| **Lipid Profile (HDL Cholesterol)** | `HDL_I.xpt` / `HDL_J.xpt` | Serum High-Density Lipoprotein (`LBDHDD`) |
| **Lipid Profile (Triglycerides & LDL)** | `TRIGLY_I.xpt` / `TRIGLY_J.xpt` | Serum Triglycerides (`LBXTR`), Low-Density Lipoprotein (`LBDLDL`, Friedewald fallback) |
| **Standard Biochemistry (Renal Panel)** | `BIOPRO_I.xpt` / `BIOPRO_J.xpt` | Serum Creatinine (`LBXSCR`), Blood Urea Nitrogen (`LBXSBU`) |
| **Fasting Glycemia Panel** | `GLU_I.xpt` / `GLU_J.xpt` | Fasting Plasma Glucose (`LBXGLU`) |
| **Glycated Hemoglobin** | `GHB_I.xpt` / `GHB_J.xpt` | Glycated Hemoglobin ($\text{HbA}_{1c}$, `LBXGH`) |
| **Smoking & Tobacco Questionnaire** | `SMQ_I.xpt` / `SMQ_J.xpt` | Smoked $\ge 100$ cigarettes in lifetime (`SMQ020`), Current smoking status (`SMQ040`) |
| **Medical Conditions Questionnaire** | `MCQ_I.xpt` / `MCQ_J.xpt` | Self-reported, physician-diagnosed cardiovascular conditions (`MCQ160B-F`) |

### 1.3 Target Variable Definition: Composite CVD
* **Criteria:** Adults ($\text{Age} \ge 20$) diagnosed by a physician with any of the following five conditions:
  1. `MCQ160B`: Congestive Heart Failure (CHF)
  2. `MCQ160C`: Coronary Heart Disease (CHD)
  3. `MCQ160D`: Angina Pectoris
  4. `MCQ160E`: Heart Attack / Myocardial Infarction (MI)
  5. `MCQ160F`: Stroke
* **Sample Size & Prevalence:** **$N = 11,288$ authentic adult participants**, with **$1,326$ confirmed CVD cases ($11.75\%$ prevalence)**.

---

## 2. Independent External Cohort: Framingham Heart Study

### 2.1 Study Origin & Governance
* **Primary Study:** Framingham Heart Study (FHS), National Heart, Lung, and Blood Institute (NHLBI) & Boston University.
* **Secondary Release:** Curated longitudinal cohort distributed via Duke University Department of Statistical Science repository (`sta210-sp20/datasets/master/framingham.csv`).
* **Study Type:** Prospective, longitudinal community-based epidemiological cohort with 10-year clinical follow-up.
* **Cohort Size:** **$N = 4,240$ real patients**, with **$644$ incident coronary heart disease events ($15.19\%$ event rate)**.

### 2.2 Variables & Domain Harmonization
| Harmonized Benchmark Feature | Framingham Source Column | Clinical Description |
| :--- | :--- | :--- |
| `Age` | `age` | Participant age in years |
| `Sex` | `male` | Binary indicator ($1=\text{Male}, 0=\text{Female}$) |
| `Smoking` | `currentSmoker` | Binary smoking indicator ($1=\text{Current}, 0=\text{Never}$) |
| `BMI` | `BMI` | Body Mass Index ($\text{kg/m}^2$) |
| `Systolic_BP` | `sysBP` | Systolic blood pressure ($\text{mmHg}$) |
| `Diastolic_BP` | `diaBP` | Diastolic blood pressure ($\text{mmHg}$) |
| `Total_Cholesterol` | `totChol` | Serum total cholesterol ($\text{mg/dL}$) |
| `Fasting_Glucose` | `glucose` | Casual/fasting serum glucose ($\text{mg/dL}$) |
| `HeartRate` | `heartRate` | Resting heart rate ($\text{beats/min}$) |
| `CVD_Diagnosis` (Target) | `TenYearCHD` | 10-year risk of incident coronary heart disease event |

### 2.3 Methodological Note on Cross-Cohort Evaluation
* **Endpoint Mismatch:** The NHANES primary model predicts cross-sectional self-reported composite cardiovascular disease (prevalence $11.75\%$), whereas the Framingham cohort evaluates prospective 10-year incident coronary heart disease (prevalence $15.19\%$).
* **Biomarker Imputation:** Extended biomarkers present in NHANES but unmeasured in this Framingham release (e.g., eGFR, HbA1c, waist circumference) are passed as missing and imputed via the median imputer fitted strictly on the NHANES training partition, simulating prospective real-world deployment of an AI risk tool in a community clinic lacking advanced metabolic assays.

---

## 3. Comparative Baseline Datasets

| Dataset | Provenance / Host | Target Condition | Sample Size ($N$) | Features |
| :--- | :--- | :--- | :---: | :---: |
| **Heart Failure Clinical Records** | UCI ML Repository / Chicco & Jurman (2020) | In-hospital / follow-up mortality | 918 | 11 |
| **Stroke Prediction Dataset** | Kaggle Healthcare Dataset / Fedesoriano (2021) | Acute cerebrovascular stroke | 5,110 | 10 |
| **Breast Cancer Wisconsin (Diagnostic)** | UCI ML Repository / Wolberg, Street, & Mangasarian | Fine Needle Aspirate Malignancy | 569 | 30 |
| **Pima Indians Diabetes** | National Institute of Diabetes & Digestive Diseases | Diabetes onset within 5 years | 768 | 8 |
| **Chronic Kidney Disease (CKD)** | UCI ML Repository / Apollo Hospitals | Chronic Kidney Disease progression | 400 | 24 |

---

## 4. Scientific Policy on In-Silico Generation
To guarantee scientific integrity and eliminate any risk of artificial results:
1. **Zero Synthetic Benchmarking:** In-silico data generation (`generate_synthetic_demo_cohort`) is strictly quarantined for offline CI smoke tests where remote CDC servers cannot be reached.
2. **Explicit Failure Mode:** If authentic CDC files or Framingham cohorts are missing on disk, the loader immediately raises an explicit `FileNotFoundError` rather than silently generating synthetic samples.
