# Appendix: TRIPOD+AI Statement Compliance Checklist
## Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (AI Extension - 2024)

| Section / Item | Item No. | Checklist Description | Reported in Study | Page / Section |
| :--- | :---: | :--- | :---: | :--- |
| **Title** | 1 | Identify the study as developing/validating a clinical AI diagnostic model. | YES | Title Page |
| **Abstract** | 2 | Structured summary of objectives, study design, cohort, performance, and clinical safety. | YES | Abstract |
| **Introduction** | 3a | Medical context, existing diagnostic gaps, and rationale for trustworthy machine learning. | YES | Section 1 |
| | 3b | Study objectives and targeted clinical outcome definitions. | YES | Section 1.2 |
| **Methods - Source Data** | 4a | Describe study design, source of data (CDC NHANES Continuous Cohort). | YES | Section 2.1 |
| | 4b | Specify eligibility criteria, dates of data accrual, and demographic distribution. | YES | Section 2.1 |
| **Methods - Participants** | 5 | State participant inclusion/exclusion flowchart and handle missing covariates. | YES | Section 2.2 |
| **Methods - Outcome** | 6a | Define binary diagnostic endpoint (CVD composite criteria) blindly ascertained. | YES | Section 2.3 |
| **Methods - Predictors** | 7a | Document all candidate predictors, units, and clinical feature engineering indices. | YES | Section 2.4 |
| | 7b | Detail handling of continuous predictors and physiological derived ratios. | YES | Section 2.4 |
| **Methods - Sample Size** | 8 | Report cohort sample size ($N \ge 10,000$) and event rate (positive cases). | YES | Section 2.5 |
| **Methods - Missing Data** | 9 | Detail missing data mechanisms and zero-leakage KNNImputer specification. | YES | Section 2.6 |
| **Methods - AI Modeling** | 10a | Fully describe 10 ML architectures, ensembling mechanisms, and Optuna tuning. | YES | Section 3.1 |
| | 10b | Detail data splitting protocol (5x5 Nested Stratified Cross-Validation). | YES | Section 3.2 |
| | 10c | Specify inductive conformal prediction ($1 - lpha = 0.95$) coverage guarantees. | YES | Section 3.3 |
| **Methods - Evaluation** | 11 | Define discrimination, Brier score, ECE, sensor perturbation, and Friedman tests. | YES | Section 3.4 |
| **Results - Participants** | 12 | Table 1: Baseline clinical, demographic, and hemodynamic characteristics. | YES | Table 1 |
| **Results - Performance** | 13a | Model discrimination (ROC-AUC, PR-AUC, Sensitivity, Specificity, MCC). | YES | Table 2 |
| | 13b | Calibration curves and Expected Calibration Error (ECE) quantification. | YES | Figure 3 |
| **Results - Stress-Testing**| 14 | Degradation under 0% to 20% simulated laboratory sensor drift. | YES | Figure 2 |
| **Results - Explainability**| 15 | Global and local SHAP feature attributions cross-referenced with AHA/ACC guidelines. | YES | Figure 4 |
| **Discussion** | 16 | Interpretation of findings in light of clinical guidelines and existing literature. | YES | Section 4.1 |
| **Limitations** | 17 | Discuss cross-sectional nature, external validity, and compute constraints. | YES | Section 4.3 |
| **Data & Code Availability**| 18 | Fully reproducible GitHub repository with CI/CD and open NHANES cohort scripts. | YES | Back Matter |
