"""
train_model.py
--------------
Trains two ML models for the Smart Grid AI System:

1. Load Forecaster        -> RandomForestRegressor
2. Anomaly / Fault Detector -> IsolationForest

Both are serialized into a single `model.pkl` file via joblib.

Run:
    python train_model.py
"""

from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# Make local package importable when run as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_fetch import get_energy_data, generate_mock_data
from utils.preprocessing import build_features, FEATURE_COLUMNS


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame) -> dict:
    """Train load forecaster + anomaly detector on the given DataFrame."""
    if df.empty:
        raise ValueError("Training data is empty.")

    X = df[FEATURE_COLUMNS].values
    y = df["value"].values

    # --- Load forecasting model ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)
    preds = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    print(f"[Forecaster] MAE={mae:.2f}  MSE={mse:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    # Baseline linear model for comparison (kept for transparency)
    linear = LinearRegression().fit(X_train, y_train)
    lin_r2 = r2_score(y_test, linear.predict(X_test))
    print(f"[Linear baseline] R2={lin_r2:.3f}")

    # --- Anomaly detection model ---
    anomaly_features = df[["value"] + FEATURE_COLUMNS].values
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )
    iso_forest.fit(anomaly_features)
    print("[AnomalyDetector] fitted on", len(anomaly_features), "samples")

    # Store reference mean/std for denormalization & scoring thresholds
    stats = {
        "value_mean": float(df["value"].mean()),
        "value_std": float(df["value"].std() or 1.0),
        "value_max": float(df["value"].max()),
        "value_min": float(df["value"].min()),
    }

    # Evaluate Anomaly Detector using statistical heuristics as pseudo ground-truth
    pred_labels = iso_forest.predict(anomaly_features)
    pred_anomalies = (pred_labels == -1).astype(int)
    
    true_anomalies = ((df["value"] > stats["value_mean"] + 2 * stats["value_std"]) | 
                      (df["value"] < stats["value_mean"] - 2 * stats["value_std"])).astype(int)
    
    precision = precision_score(true_anomalies, pred_anomalies, zero_division=0)
    recall = recall_score(true_anomalies, pred_anomalies, zero_division=0)
    f1 = f1_score(true_anomalies, pred_anomalies, zero_division=0)
    print(f"[AnomalyDetector] Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    bundle = {
        "regressor": regressor,
        "linear_baseline": linear,
        "anomaly": iso_forest,
        "feature_cols": FEATURE_COLUMNS,
        "metrics": {
            "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "linear_r2": lin_r2,
            "anomaly_precision": float(precision), "anomaly_recall": float(recall), "anomaly_f1": float(f1)
        },
        "stats": stats,
    }
    return bundle


def save_bundle(bundle: dict, path: str = MODEL_PATH) -> None:
    joblib.dump(bundle, path)
    print(f"Saved model bundle to {path}")


def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction helpers (also used at runtime from app.py)
# ---------------------------------------------------------------------------

def predict_load(bundle: dict, feature_frame: pd.DataFrame) -> np.ndarray:
    X = feature_frame[bundle["feature_cols"]].values
    return bundle["regressor"].predict(X)


def detect_anomalies(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Return array of -1 (anomaly) / 1 (normal) per row."""
    cols = ["value"] + bundle["feature_cols"]
    X = df[cols].values
    return bundle["anomaly"].predict(X)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching training data...")
    try:
        result = get_energy_data(hours=24 * 30)  # 30 days if possible
        raw = result["data"]
        print(f"  source={result['source']} rows={len(raw)}")
    except Exception as exc:
        print(f"  live fetch failed ({exc}) – using mock data")
        raw = generate_mock_data(hours=24 * 30)

    # If the API returned too few rows (common on demo keys), augment with mock
    if len(raw) < 200:
        print("  augmenting with synthetic data for stable training...")
        raw = raw + generate_mock_data(hours=24 * 30)

    df = build_features(raw)
    bundle = train_models(df)
    save_bundle(bundle)
    print("Training complete.")


if __name__ == "__main__":
    main()
