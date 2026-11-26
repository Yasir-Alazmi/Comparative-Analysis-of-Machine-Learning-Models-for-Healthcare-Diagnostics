#!/usr/bin/env python3
"""
Draw the README summary figure from the committed result CSVs only.

Panel A: held-out test ROC-AUC with 95% bootstrap CI for every model.
Panel B: logistic regression, internal NHANES test set vs external Framingham.

No model is trained and no number is typed in by hand: every value is parsed
from the "ROC-AUC [95% CI]" columns of

    results/nhanes_cardiovascular/benchmark_metrics.csv
    results/nhanes_cardiovascular/external_validation_metrics.csv

Usage:
    python scripts/plot_summary.py
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "nhanes_cardiovascular")
BENCHMARK_CSV = os.path.join(RESULTS_DIR, "benchmark_metrics.csv")
EXTERNAL_CSV = os.path.join(RESULTS_DIR, "external_validation_metrics.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "figure0_summary.png")

CI_COLUMN = "ROC-AUC [95% CI]"
HIGHLIGHT_MODEL = "Logistic Regression"

HIGHLIGHT = "#257de0"
MUTED = "#8a8a8a"
TEXT = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e6e6e6"

_CI_PATTERN = re.compile(r"^\s*([\d.]+)\s*\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s*$")


def parse_ci(value: str):
    """Parse '83.30 [80.80 - 85.77]' into (83.30, 80.80, 85.77)."""
    match = _CI_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Unexpected CI format: {value!r}")
    return tuple(float(g) for g in match.groups())


def load_internal():
    df = pd.read_csv(BENCHMARK_CSV, index_col=0)
    if len(df) != 11:
        raise ValueError(f"Expected 11 models in {BENCHMARK_CSV}, found {len(df)}")
    rows = [(name, *parse_ci(df.loc[name, CI_COLUMN])) for name in df.index]
    return sorted(rows, key=lambda r: r[1])  # ascending, so the best model is drawn on top


def load_external():
    df = pd.read_csv(EXTERNAL_CSV)
    if len(df) != 1:
        raise ValueError(f"Expected 1 row in {EXTERNAL_CSV}, found {len(df)}")
    row = df.iloc[0]
    return int(row["N_Patients"]), parse_ci(row[CI_COLUMN])


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dot_whisker(ax, y, point, lo, hi, color, label_value=True):
    ax.hlines(y, lo, hi, color=color, linewidth=2)
    ax.plot(point, y, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    if label_value:
        ax.text(hi + 0.4, y, f"{point:.2f}", va="center", ha="left", fontsize=8.5, color=TEXT)


def main():
    internal = load_internal()
    n_external, (ext_auc, ext_lo, ext_hi) = load_external()
    lr = next(r for r in internal if r[0] == HIGHLIGHT_MODEL)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=200, gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor("white")

    # Panel A: all models on the internal held-out test set
    for i, (name, auc, lo, hi) in enumerate(internal):
        color = HIGHLIGHT if name == HIGHLIGHT_MODEL else MUTED
        _dot_whisker(ax_a, i, auc, lo, hi, color)
    ax_a.set_yticks(range(len(internal)))
    ax_a.set_yticklabels([r[0] for r in internal], color=TEXT)
    for tick, row in zip(ax_a.get_yticklabels(), internal):
        if row[0] == HIGHLIGHT_MODEL:
            tick.set_fontweight("bold")
    lo_all = min(r[2] for r in internal)
    hi_all = max(r[3] for r in internal)
    ax_a.set_xlim(lo_all - 1.5, hi_all + 3)
    ax_a.set_xlabel("ROC-AUC (%), 95% bootstrap CI", color=TEXT_SECONDARY, fontsize=9)
    ax_a.set_title("A. Held-out NHANES test set, all 11 models", loc="left", fontsize=11,
                   color=TEXT, fontweight="bold")
    _style(ax_a)

    # Panel B: logistic regression, internal vs external
    rows_b = [
        (f"Framingham, external\n(n = {n_external:,}, no retraining)", ext_auc, ext_lo, ext_hi),
        ("NHANES test set, internal", lr[1], lr[2], lr[3]),
    ]
    for i, (name, auc, lo, hi) in enumerate(rows_b):
        _dot_whisker(ax_b, i, auc, lo, hi, HIGHLIGHT)
    ax_b.set_yticks(range(len(rows_b)))
    ax_b.set_yticklabels([r[0] for r in rows_b], color=TEXT)
    ax_b.set_ylim(-0.6, len(rows_b) - 0.4)
    ax_b.axvline(50, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax_b.text(50.4, -0.5, "chance (50)", fontsize=8, color=TEXT_SECONDARY, va="bottom")
    ax_b.set_xlim(48, max(lr[3], ext_hi) + 4)
    ax_b.set_xlabel("ROC-AUC (%), 95% bootstrap CI", color=TEXT_SECONDARY, fontsize=9)
    ax_b.set_title("B. Logistic regression: internal vs external", loc="left", fontsize=11,
                   color=TEXT, fontweight="bold")
    _style(ax_b)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[+] Saved {os.path.relpath(OUT_PNG, REPO_ROOT)}")


if __name__ == "__main__":
    main()
