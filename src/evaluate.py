"""
src/evaluate.py
───────────────
Comprehensive model evaluation:
- Classification report
- Confusion matrix (saved as PNG)
- ROC-AUC curve (saved as PNG)
- Feature importance bar chart (saved as PNG)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)

from src.logger import get_logger

log = get_logger(__name__)

# ── Consistent visual style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"phishing": "#e74c3c", "legit": "#2ecc71", "accent": "#2980b9"}


def evaluate_model(model, X_test: np.ndarray, y_test, output_dir: str = "outputs"):
    """
    Run full evaluation suite and save all plots.

    Args:
        model      : Fitted sklearn estimator.
        X_test     : Scaled test features.
        y_test     : True test labels.
        output_dir : Directory to save plots.

    Returns:
        dict of key metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    roc   = roc_auc_score(y_test, y_proba)

    log.info("─" * 50)
    log.info("MODEL EVALUATION RESULTS")
    log.info("─" * 50)
    log.info(f"Accuracy  : {acc:.4f}")
    log.info(f"F1 Score  : {f1:.4f}")
    log.info(f"ROC-AUC   : {roc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred,
                                          target_names=["Legitimate", "Phishing"]))

    _plot_confusion_matrix(y_test, y_pred, output_dir)
    _plot_roc_curve(y_test, y_proba, roc, output_dir)

    return {"accuracy": acc, "f1": f1, "roc_auc": roc}


def _plot_confusion_matrix(y_test, y_pred, output_dir: str):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved → {path}")


def _plot_roc_curve(y_test, y_proba, roc_score: float, output_dir: str):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=COLORS["accent"], lw=2,
            label=f"ROC Curve (AUC = {roc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS["accent"])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Phishing Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved → {path}")


def plot_feature_importance(model, feature_names: list, output_dir: str = "outputs"):
    """
    Bar chart of top-15 feature importances (RandomForest / GradientBoosting only).
    Skipped gracefully for models without feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        log.warning("Model does not expose feature_importances_ — skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]   # Top 15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        importances[indices[::-1]],
        color=COLORS["accent"],
        edgecolor="white",
    )
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Top-15 Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Feature importance chart saved → {path}")
