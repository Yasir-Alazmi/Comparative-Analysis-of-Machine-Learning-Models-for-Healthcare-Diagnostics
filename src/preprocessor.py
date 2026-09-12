"""
Dataset loading and clinical feature preprocessing routines for 5 medical benchmarks:
1. Breast Cancer Wisconsin (Diagnostic)
2. Chronic Kidney Disease (CKD)
3. Heart Failure Prediction
4. Pima Indians Diabetes Database
5. Stroke Prediction Dataset (with SMOTE oversampling for extreme class imbalance)
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def load_breast_cancer(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Breast Cancer Wisconsin (Diagnostic) dataset."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "breast_cancer.csv")
    df = pd.read_csv(data_path)
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df.dropna(axis=1, how="all", inplace=True)
    df.dropna(inplace=True)
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def load_chronic_kidney(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Chronic Kidney Disease dataset with clinical imputation."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "kidney_disease.csv")
    df = pd.read_csv(data_path)
    df.drop(columns=["id"], errors="ignore", inplace=True)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan, "nan": np.nan, "	?": np.nan})

    target = df["classification"].str.lower().str.strip().apply(
        lambda x: 0 if "notckd" in str(x) else 1
    )

    num_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    cat_cols = ["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

    present_num = [c for c in num_cols if c in df.columns]
    present_cat = [c for c in cat_cols if c in df.columns]

    for col in present_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    for col in present_cat:
        df[col] = df[col].astype(str).str.strip().str.lower()
        mode_val = df[col].replace("nan", np.nan).dropna().mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "unknown"
        df[col] = df[col].replace("nan", fill_val).fillna(fill_val)
        df[col] = le.fit_transform(df[col])

    X = df[present_num + present_cat].copy()
    return X, target


def load_heart_failure(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Heart Failure Prediction dataset."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "heart_failure.csv")
    df = pd.read_csv(data_path)
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    return X, y


def load_pima_diabetes(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Pima Indians Diabetes Database with biological zero-imputation."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "pima_diabetes.csv")
    df = pd.read_csv(data_path)
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def load_stroke_prediction(data_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess Stroke Prediction Dataset."""
    if data_path is None:
        data_path = os.path.join(DEFAULT_DATA_DIR, "stroke_prediction.csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id"], errors="ignore")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y


DATASET_REGISTRY = {
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "loader": load_breast_cancer,
        "default_file": "breast_cancer.csv",
        "use_smote": False,
        "sort_metric": "Accuracy"
    },
    "chronic_kidney": {
        "name": "Chronic Kidney Disease (CKD)",
        "loader": load_chronic_kidney,
        "default_file": "kidney_disease.csv",
        "use_smote": False,
        "sort_metric": "Accuracy"
    },
    "heart_failure": {
        "name": "Heart Failure Prediction",
        "loader": load_heart_failure,
        "default_file": "heart_failure.csv",
        "use_smote": False,
        "sort_metric": "Accuracy"
    },
    "pima_diabetes": {
        "name": "Pima Indians Diabetes Database",
        "loader": load_pima_diabetes,
        "default_file": "pima_diabetes.csv",
        "use_smote": False,
        "sort_metric": "Accuracy"
    },
    "stroke_prediction": {
        "name": "Stroke Prediction Dataset",
        "loader": load_stroke_prediction,
        "default_file": "stroke_prediction.csv",
        "use_smote": True,
        "sort_metric": "F1-Score"
    }
}
