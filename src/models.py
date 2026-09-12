"""
Machine learning model definitions and pipelines for healthcare benchmarking.
Evaluates 10 diverse algorithms spanning linear, instance-based, probabilistic,
support vector, and ensemble architectures.
"""

from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def build_models(random_state: int = 42, fast_mode: bool = False) -> Dict[str, Pipeline]:
    """
    Builds the 10 machine learning models wrapped in scikit-learn Pipelines.
    Feature scaling is automatically applied to distance- and gradient-based models,
    while tree-based models retain original feature scales.

    Args:
        random_state: Random seed for reproducibility.
        fast_mode: If True, uses fewer estimators for quick smoke-testing/CI.

    Returns:
        Dictionary mapping model names to Scikit-Learn Pipeline instances.
    """
    n_est = 20 if fast_mode else 200

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=n_est, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_est, random_state=random_state),
        "XGBoost": XGBClassifier(
            n_estimators=n_est,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "SVM": SVC(probability=True, kernel="rbf", random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=n_est, random_state=random_state),
        "Bagging": BaggingClassifier(n_estimators=n_est, random_state=random_state),
    }

    # Tree-based algorithms do not strictly require feature standardization
    no_scale = {
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "AdaBoost",
        "Bagging",
    }

    pipelines = {}
    for name, model in models.items():
        if name in no_scale:
            pipelines[name] = Pipeline([("clf", model)])
        else:
            pipelines[name] = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    return pipelines
