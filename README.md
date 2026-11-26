# Cardiovascular risk models on CDC NHANES, with external validation on Framingham

[![CI](https://github.com/Yasir-Alazmi/cardiovascular-risk-external-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/cardiovascular-risk-external-validation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project compares 11 models (10 classifiers and a stacking ensemble) for predicting cardiovascular disease from routine clinical measurements, then tests the best one on a different cohort without retraining.

![Logistic regression ranks first on the NHANES test set (ROC-AUC 81.04) but drops to 64.70 on Framingham](results/nhanes_cardiovascular/figure0_summary.png)

*Drawn by [`scripts/plot_summary.py`](scripts/plot_summary.py) from the committed CSVs only.*

**Key findings**
- Plain logistic regression has the best ROC-AUC (81.04 [78.37–83.54]); it is not significantly better than the runner-up, Extra Trees (paired bootstrap p = 0.72).
- Transported unchanged to Framingham (n = 4,240), its ROC-AUC falls to 64.70 and specificity to 41.66: the model does not transfer well across cohort designs.
- One global threshold gives 57.5% sensitivity for women vs 85.3% for men; subgroup calibration would be needed before any use.

## What is checked

The models are checked beyond a single accuracy number:

- 1,000-resample bootstrap confidence intervals, and a paired bootstrap test between the top two models;
- 5-fold stratified cross-validation with a Friedman test;
- robustness to simulated measurement noise;
- SHAP feature attributions for the top model;
- conformal prediction sets;
- a sex and age fairness check;
- external validation on a different cohort (Framingham) with no retraining.

**Data.** The main data is CDC NHANES 2015–2018: n = 10,739 examined adults, with 11.59% CVD prevalence (cleaning rules below). It is split 80/20, and the test set (n = 2,148) is used only for final evaluation. The decision threshold is fixed on the training data (Youden J) before the test set is touched. The Framingham Heart Study (n = 4,240) is used only for external validation.

Every number below is read from a committed file under [`results/nhanes_cardiovascular/`](results/nhanes_cardiovascular/). They were last reproduced on 2026-11-26 with `python scripts/reproduce_benchmarks.py` (seed 42), after the data cleaning described under [Datasets](#datasets). The tree-model figures can shift slightly between library builds and machines, even with the same seed. Logistic regression has ranked first in every run so far; the runner-up has changed between runs (CatBoost, Random Forest, now Extra Trees).

> Research project. Not validated for clinical use.

## Results

### Held-out test set (n = 2,148)

| Model | ROC-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic Regression | 81.04 [78.37 - 83.54] | 73.70 [68.28 - 79.15] | 74.96 [73.12 - 76.78] | 34.89 | 0.1668 | 0.2117 |
| Extra Trees | 80.85 [78.08 - 83.32] | 77.65 [72.26 - 82.85] | 67.48 [65.24 - 69.41] | 35.66 | 0.1232 | 0.1641 |
| Random Forest | 80.60 [77.74 - 83.31] | 78.49 [73.33 - 83.67] | 68.17 [66.00 - 70.21] | 35.96 | 0.1057 | 0.0953 |
| XGBoost | 80.31 [77.54 - 83.06] | 75.96 [70.71 - 81.61] | 68.91 [66.77 - 70.85] | 33.92 | 0.1448 | 0.1624 |
| LightGBM | 80.03 [77.15 - 82.58] | 82.47 [77.54 - 87.25] | 63.56 [61.36 - 65.72] | 33.46 | 0.0953 | 0.0434 |
| CatBoost | 79.84 [76.89 - 82.58] | 73.29 [67.47 - 78.97] | 72.24 [70.26 - 74.06] | 33.74 | 0.0985 | 0.0571 |
| Naive Bayes | 78.42 [75.40 - 81.08] | 62.74 [56.72 - 68.09] | 78.07 [76.25 - 79.76] | 29.78 | 0.1894 | 0.1911 |
| Stacking Ensemble (Super Learner) | 76.62 [73.00 - 79.91] | 74.41 [68.58 - 79.85] | 69.33 [67.25 - 71.27] | 33.70 | 0.1041 | 0.0594 |
| SVM (RBF) | 76.10 [72.92 - 79.17] | 71.64 [65.92 - 76.95] | 67.25 [65.12 - 69.41] | 30.19 | 0.1732 | 0.2180 |
| Neural Net (MLP) | 75.64 [72.26 - 78.62] | 67.64 [61.60 - 73.17] | 70.89 [68.90 - 72.96] | 28.83 | 0.1409 | 0.1326 |
| KNN | 67.96 [64.58 - 71.37] | 58.79 [52.26 - 64.69] | 68.14 [66.08 - 70.23] | 18.96 | 0.2020 | 0.2142 |

- **Plain logistic regression has the best discrimination.** A paired bootstrap test against the next model (Extra Trees) gives an AUC difference of 0.19 points [-0.86%, 1.22%], with p = 0.72. So the two models are not significantly different; the top six models lie within about 1.2 AUC points of each other. The stacking ensemble ranks eighth of eleven.
- **Discrimination and calibration disagree.** Logistic regression has the best AUC but poor calibration (ECE 0.21). LightGBM and CatBoost are much better calibrated (ECE 0.04–0.06). Which model is best depends on whether you need a ranking or a calibrated risk.

### 5-fold stratified cross-validation (n = 10,739)

| Model | ROC-AUC (mean ± sd) | Sensitivity | Specificity | PR-AUC | MCC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 83.61% ± 0.65% | 75.18% ± 2.85% | 76.76% ± 1.55% | 39.73% ± 2.48% | 0.366 | 0.1597 |
| Random Forest | 82.98% ± 0.57% | 45.30% ± 2.70% | 91.43% ± 0.72% | 37.57% ± 1.20% | 0.352 | 0.1013 |
| CatBoost | 82.91% ± 0.58% | 36.71% ± 2.30% | 93.53% ± 0.40% | 38.42% ± 1.74% | 0.323 | 0.093 |
| Extra Trees | 82.74% ± 0.38% | 52.13% ± 2.98% | 87.46% ± 0.94% | 37.23% ± 1.20% | 0.336 | 0.1196 |
| LightGBM | 82.33% ± 0.45% | 30.12% ± 2.25% | 95.12% ± 0.41% | 37.94% ± 1.14% | 0.301 | 0.0898 |
| XGBoost | 82.24% ± 0.61% | 67.31% ± 2.51% | 80.97% ± 1.24% | 37.38% ± 2.21% | 0.359 | 0.1396 |
| Naive Bayes | 80.75% ± 0.60% | 63.94% ± 3.16% | 80.23% ± 0.87% | 34.28% ± 1.80% | 0.327 | 0.1825 |
| SVM (RBF) | 79.88% ± 1.15% | 69.80% ± 4.10% | 75.76% ± 1.66% | 35.91% ± 2.32% | 0.32 | 0.1653 |
| Stacking Ensemble (Super Learner) | 78.57% ± 1.04% | 29.80% ± 1.87% | 94.65% ± 0.32% | 34.42% ± 1.07% | 0.285 | 0.0988 |
| Neural Net (MLP) | 77.69% ± 1.79% | 50.68% ± 3.42% | 84.55% ± 0.51% | 31.85% ± 1.99% | 0.284 | 0.1426 |
| KNN | 70.00% ± 1.39% | 55.58% ± 1.42% | 75.71% ± 0.96% | 21.72% ± 0.89% | 0.223 | 0.1932 |

A Friedman test across the folds rejects the hypothesis that all models perform the same (χ² = 46.18, p = 1.3 × 10⁻⁶; saved in `friedman_test.csv`). Logistic regression has the best mean rank (1.2 of 11). Its cross-validated AUC (83.61) is higher than its single held-out test AUC (81.04); the test split is one draw of 2,148 patients, and the 95% CI of the test AUC is wide.

### Robustness to measurement noise (logistic regression)

This is simulated noise, not real sensor or calibration drift. Gaussian noise with mean 0 and SD equal to the stated percentage of each numeric feature's test-set SD is added to the test set only. The test classifies at the default 0.5 probability cut-off, not the locked 0.48 threshold. That is why its baseline sensitivity (71.49) differs from the main table (73.70).

| Gaussian noise on numeric features (SD as % of feature SD) | ROC-AUC [95% CI] | Sensitivity | AUC retained |
|---|---|---|---|
| +0% Gaussian Noise | 80.91 [78.37%, 83.54%] | 71.49 | 100.0% |
| +5% Gaussian Noise | 80.82 [78.25%, 83.48%] | 71.89 | 99.9% |
| +10% Gaussian Noise | 80.65 [78.06%, 83.38%] | 70.28 | 99.7% |
| +15% Gaussian Noise | 80.35 [77.75%, 83.06%] | 70.28 | 99.3% |
| +20% Gaussian Noise | 79.97 [77.34%, 82.80%] | 70.28 | 98.8% |

### Feature attributions (SHAP, mean absolute value)

Linear SHAP on the logistic regression model (the top model), in log-odds units, on 150 test patients with the training partition as background. The top five features: Non_HDL_Cholesterol (1.48), Age (1.07), Total_Cholesterol (0.83), HDL_Cholesterol (0.46), Smoking_Never (0.31). Non-HDL cholesterol is total cholesterol minus HDL, so these three lipid features are collinear and their large attributions partly offset each other; read them as a group. These are statistical attributions, not causal effects.

### Conformal prediction

At a 95% target, empirical coverage on the test set is 93.85% [92.83%, 94.83%], slightly below the target: the bootstrap CI does not include 95%. The mean prediction-set size is 1.47. 47.4% of patients get an ambiguous two-label set and would be referred for review. The coverage guarantee is marginal; it does not hold separately within each subgroup.

### Fairness check

| Groups | Sensitivity | Equal-opportunity gap [95% CI] |
|---|---|---|
| Female (N=1113) vs Male (N=1035) | 57.5% [48.2%, 67.0%] vs 85.3% [79.5%, 91.0%] | 27.77% [16.12%, 38.85%] |
| Younger (<50) (N=1029) vs Older Adults (>=50) (N=1119) | 9.1% [0.0%, 20.6%] vs 83.3% [78.2%, 88.0%] | 74.24% [61.99%, 83.87%] |

A single global threshold misses far more disease in women and in people under 50. Each subgroup would need its own calibration before any real use.

### External validation on Framingham (n = 4,240, no retraining)

| ROC-AUC [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Brier | ECE |
|---|---|---|---|---|
| 64.70 [62.52 - 66.83] | 78.27 [74.96 - 81.36] | 41.66 [40.14 - 43.25] | 0.3370 | 0.4147 |

AUC falls from 81.0 to 64.7, and the locked threshold gives low specificity (41.66%). The two cohorts differ in design: NHANES records self-reported diagnoses in a cross-sectional survey, while Framingham records 10-year incident CHD. Several NHANES features are also missing in Framingham and were imputed with training medians. So this result mostly shows how poorly the model transfers to a different cohort, which is what the check is meant to measure.

## Datasets

| Dataset | Source | Design | n | Outcome |
|---|---|---|---|---|
| CDC NHANES 2015–2018 | U.S. National Center for Health Statistics | cross-sectional survey | 10,739 | self-reported physician-diagnosed CVD (`MCQ160B-F`) |
| Framingham Heart Study | NHLBI / Boston University (via Duke) | longitudinal cohort | 4,240 | 10-year incident CHD (`TenYearCHD`) |

**NHANES cleaning** (in [`scripts/build_real_datasets.py`](scripts/build_real_datasets.py)): adults aged 20+ who were examined (`RIDSTATR == 2`); a blood-pressure reading of 0 mmHg is treated as missing; LDL is left missing when triglycerides exceed 400 mg/dL (Friedewald is invalid there) and is never negative; exact duplicate rows are dropped. This took the cohort from 11,288 to 10,739 rows. The earlier version had 124 duplicate rows (interview-only participants with no measurements), 7 negative LDL values and 28 zero diastolic pressures; all numbers on this page come from the cleaned data.

Provenance and variable definitions are in [`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md). The [`legacy/`](legacy/) folder keeps early standalone scripts on classic public datasets (UCI heart failure, breast cancer, Pima diabetes, kidney disease, Kaggle stroke). They impute and encode before the train/test split, so they leak test information; they are not part of the results above (see [`legacy/README.md`](legacy/README.md)).

## Limitations

- NHANES outcomes are self-reported diagnoses, so recall bias and under-reporting are likely.
- Framingham lacks some NHANES biomarkers. They were imputed, which limits external performance.
- SHAP attributions are not causal.
- Conformal coverage holds on average, not within each subgroup.
- The main tables use fixed, regularised hyperparameters. Optuna tuning is optional, in [`src/tuning.py`](src/tuning.py).
- The evaluation is offline only. There is no prospective or clinical validation.

A TRIPOD+AI reporting checklist is mapped in [`results/nhanes_cardiovascular/tripod_ai_checklist.md`](results/nhanes_cardiovascular/tripod_ai_checklist.md).

## Reproduce

```bash
pip install -r requirements.txt        # needs Python 3.11+ (for scipy 1.17)
python scripts/reproduce_benchmarks.py # full run, seed 42 (took about 20 minutes on a laptop CPU)
python -m pytest tests/ -v             # tests
```

Other commands:
- `make data` downloads the datasets again from CDC and Duke.
- `make smoke` runs a fast check.
- `python run_benchmark.py --dataset nhanes_cardiovascular --nested-cv` runs nested CV with Optuna.
- `python scripts/plot_summary.py` redraws the summary figure from the committed CSVs.
- `python scripts/robustness_check.py --dataset breast_cancer --fast` runs a standalone CV, multi-seed and noise check on one dataset.

## Author

Yasir Alazmi. AI / ML engineer. I made the design and the evaluation choices and reviewed all of the code. Parts of the code were written with AI coding assistants, then reviewed and tested.

Code: MIT License. The MIT License covers the code only, not the datasets. Dataset terms, as far as they could be verified:

- **NHANES:** NCHS releases NHANES as public-use files "with some restrictions": statistical analysis and reporting only, no attempt to identify any person, and no linkage to identifiable data ([NCHS Data User Agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html)).
- **Framingham extract:** taken from a Duke course repository ([sta210-sp20/datasets](https://github.com/sta210-sp20/datasets)) that has no licence file and states no terms. **Its redistribution terms are unclear.** The copy in `datasets/` is kept only for reproducibility; `make data` downloads it again from the source.
- **Benchmark datasets** (UCI, Kaggle): their terms have not been checked here. See each host before reuse.

Details are in [`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md#5-licences-and-redistribution).
