"""
src/train.py
────────────
Model selection, training, cross-validation, and persistence.
Supports: RandomForest, LogisticRegression, GradientBoosting
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.logger import get_logger

log = get_logger(__name__)


def get_model(model_type: str, params: dict):
    """
    Factory function — returns an sklearn estimator by name.

    Args:
        model_type : One of 'RandomForest', 'LogisticRegression', 'GradientBoosting'
        params     : Dict of hyperparameters for the chosen model.

    Returns:
        Unfitted sklearn estimator.
    """
    models = {
        "RandomForest":       RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GradientBoosting":   GradientBoostingClassifier,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from: {list(models.keys())}"
        )

    log.info(f"Initialising model: {model_type}")
    log.debug(f"Hyperparameters: {params}")
    return models[model_type](**params)


def train_model(model, X_train: np.ndarray, y_train, cv_folds: int = 5):
    """
    Fit the model and evaluate with stratified k-fold cross-validation.

    Args:
        model   : Unfitted sklearn estimator.
        X_train : Scaled training features.
        y_train : Training labels.
        cv_folds: Number of CV folds.

    Returns:
        Fitted model.
    """
    log.info(f"Starting training on {len(y_train)} samples...")

    # Cross-validation before final fit
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds, scoring="f1", n_jobs=-1
    )
    log.info(
        f"Cross-Val F1 ({cv_folds}-fold): "
        f"mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}"
    )

    # Final fit on entire training set
    model.fit(X_train, y_train)
    log.info("Model training complete.")
    return model


def save_model(model, path: str):
    """Persist trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    log.info(f"Model saved to: {path}")


def load_model(path: str):
    """Load a previously saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at '{path}'")
    model = joblib.load(path)
    log.info(f"Model loaded from: {path}")
    return model
