"""
src/eda.py
──────────
Exploratory Data Analysis helpers — generate and save plots used in the notebook
and in the automated pipeline run.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def class_distribution(df: pd.DataFrame, target: str = "label",
                        output_dir: str = "outputs"):
    """Bar chart of class counts."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(["Legitimate", "Phishing"], counts.values, color=colors, edgecolor="white",
           width=0.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "class_distribution.png")


def correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs"):
    """Correlation matrix heatmap for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "correlation_heatmap.png")


def feature_distributions(df: pd.DataFrame, target: str = "label",
                           output_dir: str = "outputs"):
    """KDE plots comparing feature distributions for each class."""
    features = [c for c in df.columns if c != target]
    n = len(features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        for cls, color, lbl in zip([0, 1], ["#2ecc71", "#e74c3c"],
                                    ["Legitimate", "Phishing"]):
            subset = df.loc[df[target] == cls, feat]
            try:
                sns.kdeplot(subset, ax=axes[i], color=color, label=lbl, fill=True,
                            alpha=0.3)
            except Exception:
                axes[i].hist(subset, color=color, alpha=0.4, bins=15, label=lbl)
        axes[i].set_title(feat, fontsize=9)
        axes[i].legend(fontsize=7)
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "feature_distributions.png")


def _save(fig, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"EDA plot saved → {path}")
