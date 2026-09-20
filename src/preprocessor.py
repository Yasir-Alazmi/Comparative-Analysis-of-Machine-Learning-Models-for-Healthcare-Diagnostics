"""
Leakage-Free Clinical Preprocessing & Dataset Registry Module.
Features:
1. Zero-Leakage Pipeline Architecture (ColumnTransformer + ImbPipeline)
2. Robust Scaling & Missingness Imputation strictly inside folds
3. Proper Nominal One-Hot Encoding (no arbitrary LabelEncoder ordering)
4. Borderline-SMOTE oversampling on training folds only
5. Flagship CDC NHANES Cardiovascular Benchmark Registration
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

from src.features import engineer_clinical_features
from src.data_loader import load_nhanes_cardiovascular, load_external_validation_cohort

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def build_leakage_free_pipeline(
    classifier: Any,
    num_cols: List[str],
    cat_cols: List[str],
    use_smote: bool = False,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Constructs a scientifically sealed pipeline ensuring strictly zero data leakage.
    Imputers, scalers, encoders, and resamplers are fitted solely on training partitions.
    """
    num_steps = []
    if len(num_cols) > 0:
        num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
        num_steps.append(("scaler", RobustScaler()))
    num_pipe = ImbPipeline(num_steps) if len(num_steps) > 0 else "drop"

    cat_steps = []
    if len(cat_cols) > 0:
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    cat_pipe = ImbPipeline(cat_steps) if len(cat_steps) > 0 else "drop"

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    pipeline_steps = [("preprocessor", preprocessor)]

    if use_smote:
        # Borderline-SMOTE targets boundary decision regions for maximum clinical sensitivity
        pipeline_steps.append(("resampler", BorderlineSMOTE(random_state=random_state)))

    pipeline_steps.append(("clf", classifier))

    return ImbPipeline(pipeline_steps)


# =====================================================================
# Dataset Loaders (Returning Uncontaminated Features & Target)
# =====================================================================

def load_breast_cancer(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Breast Cancer Wisconsin (Diagnostic) FNA Cytology dataset."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "breast_cancer.csv")
    df = pd.read_csv(data_path)
    df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(inplace=True)
    y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int)
    X = df.drop(columns=["diagnosis"])
    return X, y


def load_chronic_kidney(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Chronic Kidney Disease (CKD) dataset."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "kidney_disease.csv")
    df = pd.read_csv(data_path)
    df.drop(columns=["id"], errors="ignore", inplace=True)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan, "nan": np.nan, "\t?": np.nan})

    target = df["classification"].str.lower().str.strip().apply(
        lambda x: 0 if "notckd" in str(x) else 1
    )
    X = df.drop(columns=["classification"])
    num_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, target


def load_heart_failure(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Heart Failure Prediction dataset with clinical feature engineering."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "heart_failure.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    X = engineer_clinical_features(X)
    return X, y


def load_pima_diabetes(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Pima Indians Diabetes Database with biological zero-handling."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "pima_diabetes.csv")
    df = pd.read_csv(data_path)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X = engineer_clinical_features(X)
    return X, y


def load_stroke_prediction(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Stroke Prediction Dataset with clinical feature engineering."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "stroke_prediction.csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id"], errors="ignore")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X = engineer_clinical_features(X)
    return X, y


# =====================================================================
# Master Benchmark Registry
# =====================================================================

DATASET_REGISTRY = {
    "nhanes_cardiovascular": {
        "name": "CDC NHANES Cardiovascular Cohort",
        "loader": load_nhanes_cardiovascular,
        "default_file": "nhanes_cardiovascular.csv",
        "use_smote": True,
        "sort_metric": "ROC-AUC",
        "target_name": "CVD (Cardiovascular Disease)"
    },
    "heart_failure": {
        "name": "Heart Failure Prediction",
        "loader": load_heart_failure,
        "default_file": "heart_failure.csv",
        "use_smote": False,
        "sort_metric": "ROC-AUC",
        "target_name": "Heart Disease"
    },
    "stroke_prediction": {
        "name": "Stroke Prediction Dataset",
        "loader": load_stroke_prediction,
        "default_file": "stroke_prediction.csv",
        "use_smote": True,
        "sort_metric": "ROC-AUC",
        "target_name": "Acute Stroke"
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "loader": load_breast_cancer,
        "default_file": "breast_cancer.csv",
        "use_smote": False,
        "sort_metric": "Accuracy",
        "target_name": "Malignancy"
    },
    "pima_diabetes": {
        "name": "Pima Indians Diabetes Database",
        "loader": load_pima_diabetes,
        "default_file": "pima_diabetes.csv",
        "use_smote": False,
        "sort_metric": "ROC-AUC",
        "target_name": "Diabetes Onset"
    },
    "chronic_kidney": {
        "name": "Chronic Kidney Disease (CKD)",
        "loader": load_chronic_kidney,
        "default_file": "kidney_disease.csv",
        "use_smote": False,
        "sort_metric": "ROC-AUC",
        "target_name": "CKD Progression"
    }
}
