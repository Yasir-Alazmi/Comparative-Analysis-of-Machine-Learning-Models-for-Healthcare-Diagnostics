"""
Machine Learning Model Architectures & Stacking Ensemble Engine.
Evaluates 10 foundational and SOTA clinical algorithms:
1. CatBoost (Categorical Gradient Boosting)
2. LightGBM (Leaf-wise Gradient Tree Boosting)
3. XGBoost (Extreme Gradient Boosting)
4. Random Forest (Bootstrap Ensemble)
5. Extra Trees (Extremely Randomized Trees)
6. Logistic Regression (ElasticNet Regularized Clinical Baseline)
7. Support Vector Classifier (RBF Kernel with Platt Scaling)
8. Multi-Layer Perceptron (Tabular Neural Network)
9. K-Nearest Neighbors (Manifold Instance Search)
10. Gaussian Naive Bayes (Probabilistic Maximum a Posteriori)
Plus:
- Stacking Ensemble (Super Learner) combining predictions via Calibrated Meta-Learner.
"""

from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessor import build_leakage_free_pipeline


def build_models(random_state: int = 42, fast_mode: bool = False, include_stacking: bool = True) -> Dict[str, Any]:
    """
    Constructs the model dictionary containing 10 diverse estimators and the Super Learner ensemble.
    """
    n_est = 25 if fast_mode else 250

    base_models = {
        "CatBoost": CatBoostClassifier(
            iterations=n_est,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=random_state
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=n_est,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_est,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=4.0,
            eval_metric="logloss",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_est,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=n_est,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            C=1.0,
            class_weight="balanced",
            random_state=random_state
        ),
        "SVM (RBF)": SVC(
            C=1.5,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300 if fast_mode else 600,
            early_stopping=True,
            random_state=random_state
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ),
        "Naive Bayes": GaussianNB(),
    }

    if include_stacking:
        # Construct Level-1 Super Learner Stacking Ensemble using top diverse estimators
        stack_estimators = [
            ("catboost", base_models["CatBoost"]),
            ("lightgbm", base_models["LightGBM"]),
            ("xgboost", base_models["XGBoost"]),
            ("rf", base_models["Random Forest"]),
            ("lr", base_models["Logistic Regression"]),
        ]
        meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        stacking_clf = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=meta_learner,
            cv=3 if fast_mode else 5,
            stack_method="predict_proba",
            n_jobs=-1
        )
        base_models["Stacking Ensemble (Super Learner)"] = stacking_clf

    return base_models


def build_clinical_pipelines(
    models: Dict[str, Any],
    num_cols: List[str],
    cat_cols: List[str],
    use_smote: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Wraps each raw algorithm inside a strictly isolated Zero-Leakage Pipeline.
    """
    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = build_leakage_free_pipeline(
            classifier=clf,
            num_cols=num_cols,
            cat_cols=cat_cols,
            use_smote=use_smote,
            random_state=random_state
        )
    return pipelines
