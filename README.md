# Smart Grid AI System

A production-style, end-to-end **Smart Grid AI** application built with **Flask + scikit-learn + Chart.js**. It ingests real electricity demand data from the U.S. **EIA API**, forecasts load with a Random Forest regressor, flags grid faults with an Isolation Forest, and presents everything on a modern dark-mode dashboard.

---

## Overview

The system simulates the core responsibilities of a smart grid control center:

- Pulls live hourly demand from the EIA (`/v2/electricity/rto/region-data/data/`)
- Falls back to a realistic synthetic load curve when the API is unavailable
- Trains two ML models at startup (or on demand) and serializes them to `model.pkl`
- Exposes a clean REST API consumed by a single-page dashboard
- Auto-refreshes every 10 seconds when simulation mode is enabled

---

## Features

- Real-time monitoring of grid demand with live status badge (Normal / Overload / Under-utilized)
- Machine-learning load forecasting (Random Forest, with Linear Regression baseline)
- Fault / anomaly detection via Isolation Forest
- Demand-response simulation & energy-optimization suggestions
- Alerts feed for overloads and anomalies
- Export live data to CSV
- Voice assistant that reads out current usage
- Responsive dark-mode UI with card layout and Chart.js graphs
- Auto-refresh toggle (every 10s)

---

## Tech Stack

| Layer         | Technology |
| ------------- | ---------- |
| Backend       | Flask, Flask-CORS |
| ML            | scikit-learn (RandomForestRegressor, IsolationForest, LinearRegression) |
| Data          | U.S. EIA Open Data API, pandas, numpy |
| Frontend      | HTML5, CSS3 (custom dark theme), vanilla JS, Chart.js |
| Packaging     | joblib for model persistence |

---

## Project Structure

```
Smart-Grid-AI-System/
├── .gitignore
├── app.py                  # Flask backend & REST API
├── train_model.py          # Train/save ML models
├── requirements.txt
├── README.md
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
└── utils/
    ├── __init__.py
    ├── data_fetch.py       # EIA API + mock generator
    └── preprocessing.py    # Cleaning + feature engineering
```

> [!NOTE]
> `model.pkl` is generated after running `train_model.py`.


---

## API Reference

All endpoints return JSON unless noted otherwise.

| Method | Endpoint           | Description |
| ------ | ------------------ | ----------- |
| GET    | `/`                | Dashboard UI |
| GET    | `/get-energy-data` | Latest demand data (params: `region`, `hours`) |
| GET    | `/predict-load`    | Next-24h forecast (param: `horizon`) |
| GET    | `/detect-fault`    | Anomaly detection (param: `hours`) |
| GET    | `/historical-data` | Past N hours of demand |
| GET    | `/optimize`        | Demand-response suggestions |
| GET    | `/export-csv`      | Download CSV of recent data |
| GET    | `/health`          | Service heartbeat |

Example:

```bash
curl "http://localhost:5000/predict-load?horizon=24"
```

---

## How to Run

### 1. Clone and install

```bash
cd Smart-Grid-AI-System
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Set your EIA API key

Get a free key at <https://www.eia.gov/opendata/register.php> and then:

```bash
export EIA_API_KEY="your-key-here"        # Windows: set EIA_API_KEY=your-key-here
```

Without a key, the app automatically uses synthetic mock data that mimics a realistic diurnal load curve — everything still works.

### 3. Train the models (first time only)

```bash
python train_model.py
```

This creates `model.pkl`. If you skip this step, `app.py` will train a fresh model on startup.

### 4. Start the server

```bash
python app.py
```

Visit <http://localhost:5000>.

---

## Dashboard Controls

- **Refresh** - pull latest data and rerun inference
- **Start Simulation** - auto-refresh every 10 seconds
- **Speak Usage** - browser voice synthesis reads the current reading
- **Export CSV** - download the latest 48 hours of data

---

## Screenshots

_Screenshots placeholder — (Note: The `docs/` folder is not included in the repository by default. You can create it to store your own dashboard captures)._

- `dashboard.png` — main dark-mode dashboard (example)
- `forecast.png` — forecast overlay on the demand chart (example)
- `anomalies.png` — anomaly markers on the fault chart (example)

## Troubleshooting

### `InconsistentVersionWarning` (scikit-learn)
If you see a warning about unpickling a model from a different version of scikit-learn, simply retrain the model locally to match your current environment:
```bash
python train_model.py
```

### Charts not appearing / blank
If the charts are blank or improperly sized, ensure you are using a modern browser. The dashboard includes a `.chart-container` fix to prevent infinite resizing loops. If layout issues persist, clear your browser cache to ensure `style.css` and `script.js` are fully updated.

---

## Notes

- The EIA `region-data` endpoint requires a valid API key; set `EIA_API_KEY` in your environment. The free tier is ample for hobby use.
- `contamination=0.05` on Isolation Forest roughly matches historical anomaly rates; tune it to match your region's data.
- For production, swap Flask's dev server for Gunicorn + Nginx, and replace in-memory model caching with a shared model store.
