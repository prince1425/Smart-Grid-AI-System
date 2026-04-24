"""
data_fetch.py
-------------
Handles fetching real-time electricity data from the U.S. EIA (Energy
Information Administration) API. Falls back to synthetic mock data if the
API is unreachable or an API key is not provided.

EIA API docs: https://www.eia.gov/opendata/
Endpoint used: /v2/electricity/rto/region-data/data/
"""

import os
import math
import random
import datetime as dt
from typing import List, Dict, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DEFAULT_REGION = "PACW"   # Pacific West (change to any valid EIA region code)
DEFAULT_HOURS = 48        # Hours of history to fetch


def get_api_key() -> Optional[str]:
    """Read EIA API key from environment. Return None if missing."""
    return os.environ.get("EIA_API_KEY")


# ---------------------------------------------------------------------------
# Live data fetch
# ---------------------------------------------------------------------------

def fetch_eia_data(region: str = DEFAULT_REGION,
                   hours: int = DEFAULT_HOURS,
                   api_key: Optional[str] = None) -> List[Dict]:
    """
    Fetch the most recent `hours` of demand data for a given region.

    Returns a list of dicts: [{"timestamp": "...", "value": float, "region": ...}, ...]
    Raises Exception on failure so caller can fall back to mock data.
    """
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError("EIA_API_KEY not provided")

    params = {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "facets[type][]": "D",          # "D" = Demand
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": hours,
    }

    resp = requests.get(EIA_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("response", {}).get("data", [])
    if not rows:
        raise RuntimeError("EIA returned no data rows")

    cleaned = []
    for r in rows:
        cleaned.append({
            "timestamp": r.get("period"),
            "value": float(r.get("value") or 0),
            "region": r.get("respondent", region),
            "unit": r.get("value-units", "MWh"),
        })

    # Sort oldest -> newest for charting
    cleaned.sort(key=lambda x: x["timestamp"])
    return cleaned


# ---------------------------------------------------------------------------
# Mock fallback generator (realistic daily demand curve)
# ---------------------------------------------------------------------------

def generate_mock_data(hours: int = DEFAULT_HOURS,
                       inject_anomaly: bool = True) -> List[Dict]:
    """
    Generate synthetic electricity demand data that mimics a real diurnal
    load curve (peaks in the morning and evening, troughs overnight).
    """
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    data = []

    # Base daily peak in MWh
    base_peak = 3500

    for i in range(hours):
        ts = now - dt.timedelta(hours=(hours - 1 - i))
        hour_of_day = ts.hour

        # Two-peak sinusoidal daily pattern
        morning = math.sin((hour_of_day - 6) / 24 * 2 * math.pi) * 0.4
        evening = math.sin((hour_of_day - 18) / 24 * 2 * math.pi) * 0.6
        daily_factor = 0.7 + max(morning, 0) + max(evening, 0)

        # Weekend dip
        weekend_factor = 0.9 if ts.weekday() >= 5 else 1.0

        # Random noise
        noise = random.uniform(-80, 80)

        value = base_peak * daily_factor * weekend_factor + noise

        # Sprinkle anomalies for demonstration
        if inject_anomaly and random.random() < 0.03:
            value *= random.choice([0.4, 1.8])

        data.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:00:00"),
            "value": round(value, 2),
            "region": "MOCK",
            "unit": "MWh",
        })
    return data


# ---------------------------------------------------------------------------
# Public helper used by the Flask layer
# ---------------------------------------------------------------------------

def get_energy_data(region: str = DEFAULT_REGION,
                    hours: int = DEFAULT_HOURS) -> Dict:
    """
    Primary entry point. Tries live EIA data first, falls back to mock.
    Returns a dict with source info + the list of readings.
    """
    try:
        rows = fetch_eia_data(region=region, hours=hours)
        return {"source": "EIA", "region": region, "data": rows}
    except Exception as exc:
        rows = generate_mock_data(hours=hours)
        return {
            "source": "MOCK",
            "region": region,
            "data": rows,
            "note": f"Using mock data ({exc})",
        }


if __name__ == "__main__":
    # Quick smoke test
    result = get_energy_data()
    print(f"Source: {result['source']} | rows: {len(result['data'])}")
    for row in result["data"][-5:]:
        print(row)
