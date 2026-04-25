"""
src/predict.py
──────────────
Inference engine — loads saved model + scaler and makes predictions
on new feature vectors supplied via CLI or other scripts.
"""

import numpy as np
import joblib
import os
import yaml

from src.logger import get_logger

log = get_logger(__name__)


def load_artifacts(model_path: str, scaler_path: str):
    """Load the persisted model and scaler from disk."""
    for p in (model_path, scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Artifact not found: '{p}'")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    log.info("Model and scaler loaded successfully.")
    return model, scaler


def predict_single(feature_dict: dict, model, scaler, feature_cols: list,
                   threshold: float = 0.5) -> dict:
    """
    Predict whether a single URL feature set is phishing.

    Args:
        feature_dict : {feature_name: value} — must include all feature_cols.
        model        : Loaded sklearn estimator.
        scaler       : Fitted StandardScaler.
        feature_cols : Ordered list of feature names used during training.
        threshold    : Probability cut-off for the 'Phishing' class.

    Returns:
        dict with keys: prediction, probability, label
    """
    # Build ordered feature vector
    try:
        vector = np.array([[feature_dict[col] for col in feature_cols]], dtype=float)
    except KeyError as e:
        raise ValueError(f"Missing feature in input: {e}")

    vector_scaled = scaler.transform(vector)
    proba         = model.predict_proba(vector_scaled)[0][1]
    label         = int(proba >= threshold)

    result = {
        "probability": round(float(proba), 4),
        "prediction":  label,
        "label":       "⚠️  PHISHING" if label == 1 else "✅ LEGITIMATE",
    }

    log.info(
        f"Prediction → {result['label']}  "
        f"(confidence: {result['probability']:.2%})"
    )
    return result


def predict_batch(df_features, model, scaler, feature_cols: list,
                  threshold: float = 0.5):
    """
    Predict for a DataFrame of feature rows.

    Returns:
        DataFrame with added columns: proba_phishing, prediction, label
    """
    import pandas as pd

    X = df_features[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    probas    = model.predict_proba(X_scaled)[:, 1]
    preds     = (probas >= threshold).astype(int)

    df_out = df_features.copy()
    df_out["proba_phishing"] = probas.round(4)
    df_out["prediction"]     = preds
    df_out["label"]          = df_out["prediction"].map({0: "Legitimate", 1: "Phishing"})
    log.info(f"Batch prediction complete for {len(df_out)} samples.")
    return df_out
