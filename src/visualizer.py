"""
Dark-themed medical research visualizations:
- Bar Charts: Algorithm accuracy comparisons with value badges
- Heatmaps: Full multi-metric performance grid
- Radar Charts: Multi-dimensional model strengths and trade-offs
- ROC Curves: True Positive vs False Positive trade-offs with AUC annotations
"""

import os
from math import pi
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-blocking, headless-friendly rendering
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A0572", "#1D3557", "#A8DADC", "#F77F00", "#43AA8B"
]
BG_COLOR = "#0D1117"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#E6EDF3"
ACCENT_COLOR = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "legend.facecolor": "#161B22",
    "legend.edgecolor": GRID_COLOR,
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


def plot_bar_chart(results_df: pd.DataFrame, save_path: str, title: str = "Accuracy Comparison"):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(results_df))]
    bars = ax.barh(results_df.index, results_df["Accuracy"], color=colors, edgecolor=GRID_COLOR, height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%", va="center", ha="left", color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)", color=TEXT_COLOR, labelpad=10)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=GRID_COLOR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def plot_heatmap(results_df: pd.DataFrame, save_path: str, title: str = "Performance Metrics Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        results_df, annot=True, fmt=".2f", cmap="magma",
        linewidths=0.5, linecolor=GRID_COLOR,
        cbar_kws={"label": "Score (%)"},
        annot_kws={"size": 10, "weight": "bold", "color": "#FFFFFF"},
        ax=ax
    )
    ax.set_title(title, color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def plot_radar_chart(results_df: pd.DataFrame, save_path: str, title: str = "Radar Chart"):
    categories = list(results_df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor(BG_COLOR)

    plt.xticks(angles[:-1], categories, color=TEXT_COLOR, size=10, weight="bold")
    ax.tick_params(axis="x", pad=15)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#8B949E", size=8)
    plt.ylim(0, 105)
    ax.grid(color=GRID_COLOR, linestyle="--", alpha=0.6)

    for i, (alg, row) in enumerate(results_df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, linewidth=1.8, linestyle="solid", label=alg, color=c)
        ax.fill(angles, values, color=c, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), facecolor="#161B22", edgecolor=GRID_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()


def plot_roc_curves(roc_data: dict, save_path: str, title: str = "ROC Curves Comparison"):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC = {roc_auc*100:.1f}%)")

    ax.plot([0, 1], [0, 1], color="#484F58", lw=1.2, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("True Positive Rate", color=TEXT_COLOR, labelpad=8)
    ax.set_title(title, color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", facecolor="#161B22", edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, color=GRID_COLOR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
