import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#1D3557", "#A8DADC", "#F77F00", "#43AA8B"
]
BG_COLOR     = "#0D1117"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#E6EDF3"
ACCENT_COLOR = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   GRID_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "axes.titlecolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": GRID_COLOR,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

CAT_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
NUM_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc"]


def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)
    df.drop(columns=["id"], errors="ignore", inplace=True)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "": np.nan, "nan": np.nan})

    target = df["classification"].str.lower().str.strip().apply(
        lambda x: 0 if "notckd" in str(x) else 1
    )

    present_num = [c for c in NUM_COLS if c in df.columns]
    present_cat = [c for c in CAT_COLS if c in df.columns]

    for col in present_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

    le = LabelEncoder()
    for col in present_cat:
        df[col] = df[col].astype(str).str.strip().str.lower()
        mode_val = df[col].replace("nan", np.nan).dropna().mode()
        if len(mode_val):
            df[col] = df[col].replace("nan", mode_val[0])
            df[col].fillna(mode_val[0], inplace=True)
        df[col] = le.fit_transform(df[col])

    X = df[present_num + present_cat].copy()
    return X, target


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, eval_metric="logloss",
                                             random_state=42, verbosity=0),
        "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "AdaBoost":            AdaBoostClassifier(n_estimators=200, random_state=42),
        "Bagging":             BaggingClassifier(n_estimators=200, random_state=42),
    }

    no_scale = {"Decision Tree", "Random Forest", "Gradient Boosting",
                "XGBoost", "AdaBoost", "Bagging"}

    pipelines = {}
    for name, model in models.items():
        if name in no_scale:
            pipelines[name] = Pipeline([("clf", model)])
        else:
            pipelines[name] = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    return pipelines


def evaluate_models(pipelines, X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results  = {}
    roc_data = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0)    * 100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0)        * 100, 2),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba)  * 100, 2),
        }

    results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
    return results_df, roc_data


def plot_bar_chart(results_df, palette, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)

    bars = ax.barh(
        results_df.index,
        results_df["Accuracy"],
        color=palette[:len(results_df)],
        edgecolor="none",
        height=0.6,
    )

    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", ha="left",
            fontsize=10, color=TEXT_COLOR, fontweight="bold"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy Comparison Across Algorithms", fontsize=14,
                 fontweight="bold", pad=15, color=ACCENT_COLOR)
    ax.grid(axis="x", visible=True)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def plot_heatmap(results_df, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_COLOR)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        results_df.astype(float),
        annot=True, fmt=".1f", cmap=cmap,
        linewidths=0.5, linecolor=GRID_COLOR,
        ax=ax, cbar_kws={"shrink": 0.8},
        vmin=70, vmax=100,
    )

    ax.set_title("Performance Metrics Heatmap (%)", fontsize=14,
                 fontweight="bold", pad=15, color=ACCENT_COLOR)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def plot_radar_chart(results_df, palette, save_path=None):
    metrics     = list(results_df.columns)
    num_metrics = len(metrics)
    angles      = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles     += angles[:1]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Radar Charts — Algorithm Performance Profile",
                 fontsize=15, fontweight="bold", color=ACCENT_COLOR, y=1.01)

    axes_flat = axes.flatten()
    for idx, (name, row) in enumerate(results_df.iterrows()):
        ax    = axes_flat[idx]
        color = palette[idx % len(palette)]
        vals  = row.tolist() + [row.tolist()[0]]

        ax.set_facecolor(BG_COLOR)
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], metrics,
                          fontsize=7, color=TEXT_COLOR)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6, color=GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5)
        ax.spines["polar"].set_color(GRID_COLOR)
        ax.set_title(name, fontsize=9, fontweight="bold", color=color, pad=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def plot_roc_curves(roc_data, palette, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG_COLOR)

    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID_COLOR,
            linewidth=1.5, label="Random Classifier")

    sorted_roc = sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True)
    for i, (name, (fpr, tpr, auc_val)) in enumerate(sorted_roc):
        ax.plot(fpr, tpr, color=palette[i % len(palette)], linewidth=2,
                label=f"{name}  (AUC = {auc_val:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Algorithms", fontsize=14,
                 fontweight="bold", pad=15, color=ACCENT_COLOR)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.4)
    ax.grid(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.show()


def print_summary_table(results_df):
    best_overall = results_df["Accuracy"].idxmax()

    print("\n" + "=" * 80)
    print(f"{'PERFORMANCE METRICS SUMMARY':^80}")
    print("=" * 80)
    header = f"{'Algorithm':<22}" + "".join(f"{m:>12}" for m in results_df.columns)
    print(header)
    print("-" * 80)

    for alg, row in results_df.iterrows():
        line = f"{str(alg):<22}" + "".join(f"{v:>11.2f}%" for v in row)
        print(line)

    print("=" * 80)
    print(f"\n  Best Overall Algorithm : {best_overall}")
    print(f"  Accuracy               : {results_df.loc[best_overall, 'Accuracy']:.2f}%")
    print(f"  Precision              : {results_df.loc[best_overall, 'Precision']:.2f}%")
    print(f"  Recall                 : {results_df.loc[best_overall, 'Recall']:.2f}%")
    print(f"  F1-Score               : {results_df.loc[best_overall, 'F1-Score']:.2f}%")
    print(f"  ROC-AUC                : {results_df.loc[best_overall, 'ROC-AUC']:.2f}%")
    print("=" * 80 + "\n")


def main():
    DATA_PATH = "kidney_disease.csv"

    X, y      = load_and_preprocess(DATA_PATH)
    pipelines = build_models()
    results_df, roc_data = evaluate_models(pipelines, X, y)

    print_summary_table(results_df)

    plot_bar_chart(results_df,  PALETTE, save_path="bar_chart.png")
    plot_heatmap(results_df,             save_path="heatmap.png")
    plot_radar_chart(results_df, PALETTE, save_path="radar_chart.png")
    plot_roc_curves(roc_data,   PALETTE, save_path="roc_curves.png")


if __name__ == "__main__":
    main()
