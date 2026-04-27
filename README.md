# ⚡ Smart Grid AI System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, end-to-end **Smart Grid AI** application designed for real-time monitoring, load forecasting, and fault detection. The system leverages the **U.S. EIA API** for live electricity demand data and employs advanced machine learning models to ensure grid stability and efficiency.

---

## 🚀 Key Features

- **Live Monitoring**: Real-time dashboard with status indicators (Normal, Overload, Under-utilized).
- **ML Forecasting**: Predictive load modeling using **Random Forest Regressors** with cyclical feature engineering.
- **Anomaly Detection**: Automated fault detection via **Isolation Forest**, identifying grid irregularities in real-time.
- **Intelligent Optimization**: Rule-based demand-response suggestions for load shifting and peak shaving.
- **Voice Integration**: Built-in voice assistant for hands-free status reports.
- **Data Portability**: Instant CSV export for historical analysis.
- **Modern UI**: High-performance, dark-mode dashboard powered by **Chart.js**.

---

## 🏗️ Architecture

The system is built on a modular architecture:

- **Data Layer**: Handles communication with the U.S. Energy Information Administration (EIA) API, with a robust fallback to synthetic data generation.
- **Preprocessing Engine**: Implements complex feature engineering, including sinusoidal transformations for temporal data and rolling-window statistics.
- **Model Service**: Manages training and inference for an ensemble of models (Random Forest + Isolation Forest).
- **Web API**: A Flask-based RESTful interface serving processed data and model predictions.
- **Frontend**: A responsive single-page application (SPA) providing visual insights.

---

## 📂 Project Structure

```bash
Smart-Grid-AI-System/
├── app.py                  # Core Flask server & API Layer
├── train_model.py          # ML Training Pipeline
├── model.pkl               # Serialized Model Bundle (generated)
├── requirements.txt        # System Dependencies
├── README.md               # Documentation
├── static/
│   ├── script.js           # Dashboard Logic & Charting
│   └── style.css           # Premium Dark-Mode Styling
├── templates/
│   └── index.html          # Main UI Structure
└── utils/
    ├── data_fetch.py       # API Integration & Mock Logic
    └── preprocessing.py    # Feature Engineering & Normalization
```

---

## 🛠️ Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd Smart-Grid-AI-System

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration (Optional)
The system works out-of-the-box with mock data. To use live data, obtain an [EIA API Key](https://www.eia.gov/opendata/register.php) and set it:
```bash
export EIA_API_KEY="your_api_key_here"  # Windows: set EIA_API_KEY=your_api_key_here
```

### 3. Initialize Models
```bash
python train_model.py
```

### 4. Launch Dashboard
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 📊 Machine Learning Insights

### Load Forecaster
- **Algorithm**: Random Forest Regressor
- **Features**: Hour (sin/cos), Day of Week, Month, Weekend flag, Peak Hour flags.
- **Performance**: Consistently outperforms linear baselines by capturing non-linear daily patterns.

### Anomaly Detector
- **Algorithm**: Isolation Forest
- **Logic**: Analyzes residuals between actual demand and predicted load. High residuals are flagged as potential grid faults or sensor errors.
- **Contamination**: Dynamically adjusted based on historical variance.

---

## 🔮 Future Roadmap

- [ ] **Multi-Region Support**: Real-time switching between different ISO/RTO regions.
- [ ] **Weather Integration**: Correlate load forecasts with real-time temperature and humidity data.
- [ ] **Deep Learning Migration**: Implementation of LSTM/GRU networks for improved time-series forecasting.
- [ ] **Containerization**: Full Docker support for seamless cloud deployment.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ for Smart Energy Management.*
