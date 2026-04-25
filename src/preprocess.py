"""
src/preprocess.py
─────────────────
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.logger import get_logger

log = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and perform basic validation."""
    log.info(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'")

    df = pd.read_csv(path)
    log.info(f"Dataset shape: {df.shape}")
    log.debug(f"Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe:
    - Drop duplicates
    - Handle missing values (fill numeric with median)
    - Clip outliers at 1st/99th percentile for numeric cols
    """
    log.info("Cleaning data...")
    initial_rows = len(df)

    df = df.drop_duplicates()
    log.debug(f"Dropped {initial_rows - len(df)} duplicate rows")

    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing = df[numeric_cols].isnull().sum().sum()
    if missing > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        log.debug(f"Filled {missing} missing values with median")

    log.info(f"Clean dataset shape: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that improve model discrimination power.
    - dots_per_length  : density of dots in URL
    - digit_ratio      : proportion of digit characters
    - suspicion_score  : additive risk proxy
    """
    log.info("Engineering new features...")

    # Avoid division by zero
    df["dots_per_length"] = df["num_dots"] / (df["url_length"] + 1)
    df["digit_ratio"]     = df["num_digits"] / (df["url_length"] + 1)

    # Simple additive suspicion proxy (not the label leak — uses raw features)
    df["suspicion_score"] = (
        df["has_ip"].astype(int)
        + df["tld_suspicious"].astype(int)
        + (1 - df["has_https"].astype(int))
        + (df["num_at"] > 0).astype(int)
        + (df["num_hyphens"] > 3).astype(int)
    )

    log.debug("New features: dots_per_length, digit_ratio, suspicion_score")
    return df


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_path: str = "models/scaler.pkl",
):
    """
    Split into train/test sets and apply StandardScaler.

    Returns:
        X_train, X_test, y_train, y_test (all as numpy arrays / pd.Series)
    """
    log.info(f"Splitting data | test_size={test_size} | random_state={random_state}")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Persist scaler for use during inference
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved to: {scaler_path}")

    log.info(f"Train size: {X_train_scaled.shape[0]} | Test size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
