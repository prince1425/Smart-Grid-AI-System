"""
preprocessing.py
----------------
Data cleaning and feature engineering helpers used by both the ML
training pipeline and the live Flask inference endpoints.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Dict

import numpy as np
import pandas as pd


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich with calendar / cyclical features for ML models."""
    if df.empty:
        return df
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encodings so the model sees 23:00 as close to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
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


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend", "month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline → ready-to-train DataFrame."""
    df = to_dataframe(df) if isinstance(df, list) else df
    df = handle_missing(df)
    df = add_time_features(df)
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
