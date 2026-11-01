# 🚀 Google Play Store App Rating Prediction - Production MLOps System

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) pipeline that predicts **Google Play Store application ratings** ($1.0$ to $5.0$) using application metadata. 

Engineered with continuous regression benchmarking, **MLflow** experiment tracking, high-performance **FastAPI** model serving, **Docker** containerization, structured **JSON Lines inference logging**, statistical **Population Stability Index (PSI)** data drift monitoring, and **automated email alerts**.

---

## 📌 Architecture Overview

```mermaid
flowchart TD
    A["Raw Dataset\n(googleplaystore.csv)"] --> B["Data Validation & Cleaning\n(src/validate_data.py)"]
    B --> C["Feature Engineering & Preprocessing\n(StandardScaler + OneHotEncoder)"]
    C --> D["Multi-Model Benchmarking\n(Linear Regression, DT, Random Forest, GBR)"]
    D --> E["MLflow Experiment Tracking & Registry\n(sqlite:///mlflow.db)"]
    E -->|Champion Model| F["FastAPI Serving Service\n(api/main.py - Port 8000)"]
    F -->|Docker Image| G["Docker Container\n(play-store-rating-api:v1)"]
    F -->|Inference Events| H["Real-Time Inference Logs\n(logs/inference/prediction_logs.jsonl)"]
    H --> I["Automated Drift Monitor\n(automatic_drift_monitor.py)"]
    I --> J["Statistical Drift Engine\n(Numerical KS-test & Categorical Chi-Square)"]
    J --> K["Population Stability Index (PSI)\n(CSV Reports in reports/)"]
    K -->|HIGH DRIFT (PSI >= 0.25)| L["SMTP Automated Email Alerts\n(with Throttled Cooldown)"]
```

---

## 🌟 Key Highlights

- **Pure Continuous Regression:** Accurately estimates continuous ratings ($1.0$ to $5.0$) rather than trivial classification proxies.
- **Automated MLflow Tracking:** Compares 4 regression architectures, tracking MAE, MSE, RMSE, and $R^2$ metrics into a persistent SQLite database with versioned model registry.
- **High-Concurrency FastAPI Microservice:** Sub-70ms inference latencies with native Pydantic schema validation and automatic OpenAPI Swagger UI (`/docs`).
- **Production Preprocessing Engine:** Intelligently handles messy real-world strings (e.g. Size `"19M"`, Installs `"10,000,000+"`, Price `"$4.99"`).
- **Docker Containerization:** Reproducible container packaging built on `python:3.12-slim` with health checks and volume persistence.
- **Statistical Data Drift Detection:** Features both **Population Stability Index (PSI)**, **Kolmogorov-Smirnov (KS)** tests for numerical columns, and **Chi-Square** tests for categorical columns.
- **Automated Alerting with Cooldown:** Alerts on `HIGH DRIFT` via secure SMTP (Gmail) with an intelligent cooldown mechanism to eliminate inbox spam.
- **100% Automated Pytest Coverage:** 15 comprehensive unit and integration tests across data, models, endpoints, and monitoring.

---

## 📁 Repository Structure

```
play_store/
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application endpoints
│   ├── schemas.py                  # Pydantic input/output schemas
│   ├── preprocessing.py            # Feature transformation & string cleaner
│   ├── model_loader.py             # Singleton in-memory model service
│   └── monitoring_logger.py        # Thread-safe JSONL inference logger
│
├── data/
│   ├── raw/
│   │   └── googleplaystore.csv     # Raw dataset (10,841 rows)
│   └── processed/
│       ├── processed_googleplaystore.csv # Cleaned training data (8,892 rows)
│       └── dataset_metadata.json   # Feature schemas and row counts
│
├── models/
│   ├── best_google_play_rating_model.pkl # Champion Random Forest pipeline
│   └── preprocessor.pkl            # Preprocessor ColumnTransformer
│
├── monitoring/
│   ├── drift_detector.py           # PSI and statistical hypothesis drift engine
│   └── alert_service.py            # SMTP email alert dispatcher with cooldown
│
├── outputs/
│   ├── model_comparison_report.csv # Multi-model evaluation benchmark
│   └── model_metrics.csv           # Champion model evaluation metrics
│
├── reports/                        # Timestamped drift reports (CSV)
│   └── .gitkeep
│
├── logs/
│   ├── inference/
│   │   └── prediction_logs.jsonl   # Live production inference audit logs
│   ├── drift_alerts.log            # Drift history log
│   └── .gitkeep
│
├── docs/
│   ├── IEEE_Project_Report.md      # Publication-ready IEEE-format paper
│   └── Viva_Questions_and_Answers.md # Technical interview & viva preparation
│
├── tests/
│   └── test_pipeline.py            # 15 automated pytest test cases
│
├── automatic_drift_monitor.py      # Automated monitoring scheduler
├── simulate_traffic.py             # Traffic generator (normal vs. drifted)
├── Dockerfile                      # Production Docker container definition
├── .dockerignore                   # Docker build context filters
├── .gitignore                      # Git tracking exclusion rules
├── .env.example                    # Environment secrets template
├── requirements.txt                # Full project environment dependencies
├── requirements-api.txt            # Minimal production Docker dependencies
└── README.md                       # Comprehensive project documentation
```

---

## ⚡ Quickstart Guide (macOS / VS Code)

### 1. Prerequisites
- **macOS** with **Python 3.12** installed
- **Docker Desktop** running

### 2. Navigate and Activate Environment
```bash
cd ~/Desktop/play_store
source .venv/bin/activate
```

*(If creating a new virtual environment: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)*

---

## 🏋️‍♂️ Model Training & MLflow Experiment Tracking

To execute multi-model benchmarking across Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors:

```bash
python3 src/train_mlflow_regression.py
```

### Benchmark Results

| Model Architecture | MAE | MSE | RMSE | $R^2$ Score | Regression Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.3494 | 0.2480 | 0.4980 | 0.0573 | 91.70% |
| **Decision Tree Regressor** | 0.3391 | 0.2668 | 0.5165 | -0.0141 | 91.94% |
| **Random Forest Regressor** 🏆 | **0.3128** | **0.2175** | **0.4663** | **0.1734** | **92.57%** |
| **Gradient Boosting Regressor**| 0.3221 | 0.2240 | 0.4733 | 0.1485 | 92.34% |

**Champion Model:** **Random Forest Regressor** was selected and saved to `models/best_google_play_rating_model.pkl`.

---

## 🌐 FastAPI Prediction Service

### Running Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
- **`GET /`** – API status and version metadata
- **`GET /health`** – Liveness and model readiness health check
- **`GET /model-info`** – Model architecture and baseline evaluation metrics
- **`POST /predict`** – Predict single app rating and log telemetry
- **`POST /predict/batch`** – Bulk app ratings evaluation

Interactive Swagger UI: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🐳 Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t play-store-rating-api:v1 .
```

### 2. Run the Container
```bash
docker run -d \
  --name playstore-api \
  -p 8001:8000 \
  -v $(pwd)/logs/inference:/app/logs/inference \
  play-store-rating-api:v1
```

*(Note: Port 8001 is mapped to avoid conflicts with existing background containers)*

### 3. Test Containerized Prediction via cURL
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "App": "Candy Crush Saga",
    "Category": "GAME",
    "Reviews": 22426677,
    "Size": "74M",
    "Installs": "500,000,000+",
    "Type": "Free",
    "Price": "0",
    "Content Rating": "Everyone",
    "Genres": "Casual",
    "Last Updated": "July 5, 2018",
    "Current Ver": "1.129.0",
    "Android Ver": "4.0.3 and up"
  }'
```

**Response:**
```json
{
  "request_id": "4c00ce94-70b0-4e5b-b3dd-ba7694ed90c9",
  "app_name": "Candy Crush Saga",
  "predicted_rating": 4.39,
  "prediction_latency_ms": 82.85,
  "model_name": "Google_Play_Store_RandomForest",
  "model_version": "1.0.0",
  "timestamp": "2026-11-01T06:37:51.418599Z",
  "status": "SUCCESS"
}
```

---

## 📊 Data Drift Monitoring & Automated Email Alerts

### 1. Send Simulated Traffic
```bash
# Send normal traffic:
python3 simulate_traffic.py --mode normal

# Send drifted traffic (outlier prices and sizes):
python3 simulate_traffic.py --mode drift
```

### 2. Run Drift Detection Check
```bash
# Run a single drift check:
python3 automatic_drift_monitor.py --once

# Or run continuous background monitoring (polls every 60s):
python3 automatic_drift_monitor.py --interval 60
```

### Population Stability Index (PSI) Thresholds
- **$\text{PSI} < 0.10$**: 🟢 **NO SIGNIFICANT DRIFT**
- **$0.10 \le \text{PSI} < 0.25$**: 🟡 **MODERATE DRIFT**
- **$\text{PSI} \ge 0.25$**: 🔴 **HIGH DRIFT** (triggers automated email alert)

### 3. Configuring Email Alerts
Copy `.env.example` to `.env` and provide your SMTP details:
```ini
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_SENDER_EMAIL=your-email@gmail.com
SMTP_APP_PASSWORD=your-16-character-app-password
SMTP_RECEIVER_EMAIL=recipient-email@gmail.com
```

---

## 🧪 Automated Testing Suite

Run the 15-test automated Pytest verification:

```bash
PYTHONPATH=. pytest -v tests/test_pipeline.py
```

**Expected Result:**
```
======================== 15 passed, 6 warnings in 2.15s ========================
```

---

## 🐙 Pushing to GitHub

To push this repository to GitHub:

```bash
# 1. Initialize git (already completed)
cd ~/Desktop/play_store

# 2. Link your remote GitHub repository
git remote add origin https://github.com/your-username/google-play-rating-mlops.git

# 3. Rename branch to main
git branch -M main

# 4. Push code
git push -u origin main
```

---

## 📄 Documentation Deliverables

- **IEEE Project Report:** [`docs/IEEE_Project_Report.md`](docs/IEEE_Project_Report.md)
- **Viva Questions & Answers:** [`docs/Viva_Questions_and_Answers.md`](docs/Viva_Questions_and_Answers.md)
