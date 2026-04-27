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
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_fetch import get_energy_data, generate_mock_data
from utils.preprocessing import (
    build_features,
    FEATURE_COLUMNS,
    ANOMALY_FEATURE_COLUMNS,
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


# ---------------------------------------------------------------------------
# Anomaly feature construction (training-time AND inference-time)
# ---------------------------------------------------------------------------

def _build_anomaly_matrix(df: pd.DataFrame, regressor: RandomForestRegressor) -> np.ndarray:
    """
    Assemble the input matrix for the Isolation Forest.

    Combines three strong signals:
      * the raw `value`
      * the engineered ANOMALY_FEATURE_COLUMNS (lag, rolling stats, residuals)
      * the residual from the load forecaster
        (signed and scaled by rolling std).  Real anomalies show up as
        large residuals against the forecast - this is the single biggest
        lift for precision/recall.
    """
    base = df[["value"] + ANOMALY_FEATURE_COLUMNS].to_numpy(dtype=float)

    y_hat = regressor.predict(df[FEATURE_COLUMNS].values)
    resid = df["value"].values - y_hat
    scale = df["roll_std_24"].replace(0, np.nan).fillna(df["value"].std() or 1.0).values
    scaled_resid = resid / scale
    abs_scaled_resid = np.abs(scaled_resid)

    extras = np.column_stack([resid, scaled_resid, abs_scaled_resid])
    return np.hstack([base, extras])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(df: pd.DataFrame) -> dict:
    """Train load forecaster + anomaly detector on the given DataFrame."""
    if df.empty:
        raise ValueError("Training data is empty.")

    X = df[FEATURE_COLUMNS].values
    y = df["value"].values

    # --- Load forecasting model --------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        bootstrap=True,
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

    linear = LinearRegression().fit(X_train, y_train)
    lin_r2 = r2_score(y_test, linear.predict(X_test))
    print(f"[Linear baseline] R2={lin_r2:.3f}")

    stats = {
        "value_mean": float(df["value"].mean()),
        "value_std": float(df["value"].std() or 1.0),
        "value_max": float(df["value"].max()),
        "value_min": float(df["value"].min()),
    }

    # --- Anomaly detection model -------------------------------------------
    anomaly_features = _build_anomaly_matrix(df, regressor)

    true_anomalies = (
        (df["value"] > stats["value_mean"] + 2 * stats["value_std"])
        | (df["value"] < stats["value_mean"] - 2 * stats["value_std"])
    ).astype(int).values

    contamination = float(np.clip(true_anomalies.mean(), 0.01, 0.15))
    if contamination <= 0:
        contamination = 0.05

    iso_forest = IsolationForest(
        n_estimators=400,
        max_samples=min(512, len(anomaly_features)),
        contamination=contamination,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(anomaly_features)
    print(
        f"[AnomalyDetector] fitted on {len(anomaly_features)} samples "
        f"(contamination={contamination:.3f})"
    )

    pred_labels = iso_forest.predict(anomaly_features)
    pred_anomalies = (pred_labels == -1).astype(int)

    precision = precision_score(true_anomalies, pred_anomalies, zero_division=0)
    recall = recall_score(true_anomalies, pred_anomalies, zero_division=0)
    f1 = f1_score(true_anomalies, pred_anomalies, zero_division=0)
    print(f"[AnomalyDetector] Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    bundle = {
        "regressor": regressor,
        "linear_baseline": linear,
        "anomaly": iso_forest,
        "feature_cols": FEATURE_COLUMNS,
        "anomaly_feature_cols": ANOMALY_FEATURE_COLUMNS,
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "linear_r2": float(lin_r2),
            "anomaly_precision": float(precision),
            "anomaly_recall": float(recall),
            "anomaly_f1": float(f1),
            "anomaly_contamination": float(contamination),
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
    regressor = bundle["regressor"]
    anomaly_cols = bundle.get("anomaly_feature_cols", bundle["feature_cols"])
    base_cols = ["value"] + anomaly_cols

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        cols = ["value"] + bundle["feature_cols"]
        return bundle["anomaly"].predict(df[cols].values)

    base = df[base_cols].to_numpy(dtype=float)
    y_hat = regressor.predict(df[bundle["feature_cols"]].values)
    resid = df["value"].values - y_hat
    if "roll_std_24" in df.columns:
        scale = df["roll_std_24"].replace(0, np.nan).fillna(
            df["value"].std() or 1.0
        ).values
    else:
        scale = np.full_like(resid, df["value"].std() or 1.0, dtype=float)
    scaled_resid = resid / scale
    extras = np.column_stack([resid, scaled_resid, np.abs(scaled_resid)])
    X = np.hstack([base, extras])
    return bundle["anomaly"].predict(X)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching training data...")
    try:
        result = get_energy_data(hours=24 * 60)
        raw = result["data"]
        print(f"  source={result['source']} rows={len(raw)}")
    except Exception as exc:
        print(f"  live fetch failed ({exc}) - using mock data")
        raw = generate_mock_data(hours=24 * 60)

    if len(raw) < 24 * 30:
        print("  augmenting with synthetic data for stable training...")
        raw = raw + generate_mock_data(hours=24 * 60)

    df = build_features(raw)
    bundle = train_models(df)
    save_bundle(bundle)
    print("Training complete.")


if __name__ == "__main__":
    main()
