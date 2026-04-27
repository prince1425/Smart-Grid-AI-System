"""
preprocessing.py
----------------
Data cleaning and feature engineering helpers used by both the ML
training pipeline and the live Flask inference endpoints.

Two feature schemas are exported:

* FEATURE_COLUMNS         -> deterministic, timestamp-derivable features
                             used by the load forecaster (works for both
                             historical training and future-horizon
                             inference where only the timestamp is known).

* ANOMALY_FEATURE_COLUMNS -> richer schema (lag values, rolling mean/std,
                             rolling z-score, residual proxies) used by
                             the Isolation Forest anomaly detector. Always
                             computed from a contiguous historical series.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """Convert the list-of-dicts payload from data_fetch into a DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps using forward-fill then interpolation, drop residual NaNs."""
    if df.empty:
        return df
    df = df.copy()
    df["value"] = (
        df["value"]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .interpolate(method="linear")
    )
    df = df.dropna(subset=["value"])
    return df


# ---------------------------------------------------------------------------
# Time-only features (safe for forecasting future timestamps)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich with calendar / cyclical features for ML models."""
    if df.empty:
        return df
    df = df.copy()
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_business_hour"] = (
        (df["hour"] >= 8) & (df["hour"] <= 18) & (df["is_weekend"] == 0)
    ).astype(int)
    df["is_morning_peak"] = ((df["hour"] >= 6) & (df["hour"] <= 9)).astype(int)
    df["is_evening_peak"] = ((df["hour"] >= 17) & (df["hour"] <= 21)).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


# ---------------------------------------------------------------------------
# Lag / rolling features (require contiguous history; used for anomaly model)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-window statistics on the `value` column.
    Captures recent dynamics so Isolation Forest can spot points that
    deviate from the local pattern (not just globally extreme values).
    """
    if df.empty or "value" not in df.columns:
        return df
    df = df.copy()
    v = df["value"]

    df["lag_1"] = v.shift(1)
    df["lag_24"] = v.shift(24)
    df["lag_168"] = v.shift(168)
    df["delta_1"] = v.diff(1)

    rolling24 = v.shift(1).rolling(window=24, min_periods=4)
    df["roll_mean_24"] = rolling24.mean()
    df["roll_std_24"] = rolling24.std()
    df["roll_min_24"] = rolling24.min()
    df["roll_max_24"] = rolling24.max()

    rolling168 = v.shift(1).rolling(window=168, min_periods=24)
    df["roll_mean_168"] = rolling168.mean()
    df["roll_std_168"] = rolling168.std()

    safe_std = df["roll_std_24"].replace(0, np.nan)
    df["roll_z_24"] = (v - df["roll_mean_24"]) / safe_std

    df["seasonal_resid_24"] = v - df["lag_24"]
    df["seasonal_resid_168"] = v - df["lag_168"]

    df = df.replace([np.inf, -np.inf], np.nan)

    fill_zero = [
        "lag_1", "lag_24", "lag_168", "delta_1",
        "roll_z_24", "seasonal_resid_24", "seasonal_resid_168",
    ]
    fill_with_value = [
        "roll_mean_24", "roll_min_24", "roll_max_24", "roll_mean_168",
    ]
    fill_one = ["roll_std_24", "roll_std_168"]

    for c in fill_zero:
        df[c] = df[c].fillna(0.0)
    for c in fill_with_value:
        df[c] = df[c].fillna(v)
    for c in fill_one:
        df[c] = df[c].fillna(1.0)

    for c in ["lag_1", "lag_24", "lag_168"]:
        mask = df[c] == 0.0
        df.loc[mask, c] = v[mask].values
    return df


def normalize(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """Z-score normalize a numeric column; add a `<col>_norm` field."""
    if df.empty or col not in df.columns:
        return df
    df = df.copy()
    mean = df[col].mean()
    std = df[col].std() or 1.0
    df[f"{col}_norm"] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "is_weekend", "is_business_hour",
    "is_morning_peak", "is_evening_peak",
    "month",
]

ANOMALY_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "lag_1", "lag_24", "lag_168",
    "delta_1",
    "roll_mean_24", "roll_std_24", "roll_min_24", "roll_max_24",
    "roll_mean_168", "roll_std_168",
    "roll_z_24",
    "seasonal_resid_24", "seasonal_resid_168",
]


# ---------------------------------------------------------------------------
# Public pipelines
# ---------------------------------------------------------------------------

def build_features(df) -> pd.DataFrame:
    """Full preprocessing pipeline -> ready-to-train DataFrame."""
    df = to_dataframe(df) if isinstance(df, list) else df
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = normalize(df)
    return df


def future_feature_frame(last_ts: dt.datetime, horizon_hours: int = 24) -> pd.DataFrame:
    """Build a feature frame for timestamps we want to forecast."""
    rows = []
    for i in range(1, horizon_hours + 1):
        ts = last_ts + dt.timedelta(hours=i)
        rows.append({"timestamp": ts, "value": 0.0})
    df = pd.DataFrame(rows)
    df = add_time_features(df)
    return df


if __name__ == "__main__":
    from data_fetch import get_energy_data  # type: ignore

    result = get_energy_data()
    df = build_features(result["data"])
    print(df.head())
    print("shape:", df.shape)
