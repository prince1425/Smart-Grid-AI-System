"""
app.py
------
Flask backend for the Smart Grid AI System.

Endpoints
---------
GET  /                  -> Dashboard (index.html)
GET  /get-energy-data   -> Live electricity demand data (EIA or mock)
GET  /predict-load      -> Next-24h load forecast
GET  /detect-fault      -> Isolation-Forest anomaly detection on recent data
GET  /historical-data   -> Past N hours of demand data
GET  /optimize          -> Energy optimization suggestions
GET  /export-csv        -> Download current data as CSV
GET  /health            -> Simple health check

Run:
    python app.py
"""

from __future__ import annotations

import os
import io
import csv
import json
import traceback
import datetime as dt

import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from utils.data_fetch import get_energy_data, generate_mock_data
from utils.preprocessing import build_features, future_feature_frame
from train_model import (
    load_bundle, predict_load, detect_anomalies, train_models, save_bundle,
    MODEL_PATH,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load (or train) model bundle at startup -----------------------------------
MODEL = None


def _ensure_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_bundle(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No model.pkl found – training on mock data...")
        raw = generate_mock_data(hours=24 * 30)
        df = build_features(raw)
        MODEL = train_models(df)
        save_bundle(MODEL)
    return MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def json_error(msg: str, code: int = 500):
    return jsonify({"ok": False, "error": msg}), code


def classify_status(value: float, stats: dict) -> str:
    """Map a live reading to a simple grid status."""
    threshold_high = stats["value_mean"] + 1.5 * stats["value_std"]
    threshold_low = max(0, stats["value_mean"] - 1.5 * stats["value_std"])
    if value >= threshold_high:
        return "Overload"
    if value <= threshold_low:
        return "Under-utilized"
    return "Normal"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "time": dt.datetime.utcnow().isoformat()})


@app.route("/get-energy-data")
def get_energy():
    """Return the most recent electricity-demand data."""
    try:
        region = request.args.get("region", "PACW")
        hours = int(request.args.get("hours", 48))
        result = get_energy_data(region=region, hours=hours)
        bundle = _ensure_model()

        data = result["data"]
        if data:
            latest = data[-1]
            latest["status"] = classify_status(latest["value"], bundle["stats"])

        return jsonify({
            "ok": True,
            "source": result["source"],
            "region": result["region"],
            "count": len(data),
            "data": data,
            "note": result.get("note"),
        })
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


@app.route("/predict-load")
def predict():
    """Forecast electricity load for the next N hours."""
    try:
        horizon = int(request.args.get("horizon", 24))
        bundle = _ensure_model()

        result = get_energy_data(hours=24)
        df = build_features(result["data"])
        if df.empty:
            return json_error("No data available for prediction", 400)

        last_ts = df["timestamp"].iloc[-1].to_pydatetime()
        future_df = future_feature_frame(last_ts, horizon_hours=horizon)
        preds = predict_load(bundle, future_df)

        forecast = [
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:00:00"),
                "predicted_value": round(float(v), 2),
            }
            for ts, v in zip(future_df["timestamp"], preds)
        ]

        return jsonify({
            "ok": True,
            "horizon_hours": horizon,
            "base_timestamp": last_ts.strftime("%Y-%m-%dT%H:00:00"),
            "forecast": forecast,
            "metrics": bundle.get("metrics", {}),
        })
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


@app.route("/detect-fault")
def detect_fault():
    """Run Isolation Forest on recent data and flag anomalies."""
    try:
        hours = int(request.args.get("hours", 48))
        bundle = _ensure_model()

        result = get_energy_data(hours=hours)
        df = build_features(result["data"])
        if df.empty:
            return json_error("No data available for fault detection", 400)

        labels = detect_anomalies(bundle, df)
        df_out = df[["timestamp", "value"]].copy()
        df_out["anomaly"] = (labels == -1).astype(int)

        anomalies = df_out[df_out["anomaly"] == 1]
        events = [
            {
                "timestamp": row.timestamp.strftime("%Y-%m-%dT%H:00:00"),
                "value": round(float(row.value), 2),
                "severity": (
                    "High" if abs(row.value - bundle["stats"]["value_mean"])
                    > 2 * bundle["stats"]["value_std"] else "Medium"
                ),
            }
            for row in anomalies.itertuples(index=False)
        ]

        return jsonify({
            "ok": True,
            "total_points": int(len(df_out)),
            "anomaly_count": int(df_out["anomaly"].sum()),
            "anomalies": events,
            "series": [
                {
                    "timestamp": row.timestamp.strftime("%Y-%m-%dT%H:00:00"),
                    "value": round(float(row.value), 2),
                    "anomaly": int(row.anomaly),
                }
                for row in df_out.itertuples(index=False)
            ],
            "metrics": bundle.get("metrics", {}),
        })
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


@app.route("/historical-data")
def historical():
    """Return historical demand data (up to 7 days by default)."""
    try:
        hours = int(request.args.get("hours", 24 * 7))
        result = get_energy_data(hours=hours)
        return jsonify({
            "ok": True,
            "source": result["source"],
            "count": len(result["data"]),
            "data": result["data"],
        })
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


@app.route("/optimize")
def optimize():
    """Very simple rule-based optimization suggestions."""
    try:
        bundle = _ensure_model()
        result = get_energy_data(hours=24)
        df = build_features(result["data"])
        if df.empty:
            return json_error("No data available for optimization", 400)

        mean = bundle["stats"]["value_mean"]
        std = bundle["stats"]["value_std"]
        latest = df["value"].iloc[-1]

        suggestions = []
        if latest > mean + std:
            suggestions.append(
                "Peak demand detected – shift deferrable loads (EV charging, "
                "water heating, dishwashers) to off-peak hours."
            )
            suggestions.append(
                "Engage demand response program: signal large industrial "
                "consumers to reduce non-critical loads."
            )
        elif latest < mean - std:
            suggestions.append(
                "Low-demand window – ideal time to charge batteries, run "
                "pumped-hydro recharge, or schedule maintenance."
            )
        else:
            suggestions.append(
                "Grid operating within normal band – maintain current "
                "dispatch schedule."
            )

        # Next 6-hour quick forecast to support the recommendation
        future_df = future_feature_frame(
            df["timestamp"].iloc[-1].to_pydatetime(), horizon_hours=6
        )
        next6 = predict_load(bundle, future_df)
        if next6.max() > mean + std:
            suggestions.append(
                f"Forecast shows a peak of {next6.max():.0f} MWh in the next "
                f"6 hours – pre-position spinning reserves."
            )

        return jsonify({
            "ok": True,
            "current_value": round(float(latest), 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "suggestions": suggestions,
            "next_6h_peak": round(float(next6.max()), 2),
            "next_6h_min": round(float(next6.min()), 2),
        })
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


@app.route("/export-csv")
def export_csv():
    """Export the latest data as a downloadable CSV."""
    try:
        hours = int(request.args.get("hours", 48))
        result = get_energy_data(hours=hours)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "value_MWh", "region", "source"])
        for row in result["data"]:
            writer.writerow([
                row["timestamp"],
                row["value"],
                row.get("region", ""),
                result["source"],
            ])

        mem = io.BytesIO(buf.getvalue().encode("utf-8"))
        mem.seek(0)
        filename = f"smart_grid_{dt.datetime.utcnow():%Y%m%d_%H%M}.csv"
        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as exc:
        traceback.print_exc()
        return json_error(str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ensure_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
