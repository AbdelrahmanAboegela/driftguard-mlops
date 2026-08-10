# 🛡️ DriftGuard — Drift-Aware Fraud Detection & Autonomous Retraining

[![CI/CD Pipeline](https://github.com/AbdelrahmanAboegela/driftguard-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelrahmanAboegela/driftguard-mlops/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)

**DriftGuard** is an end-to-end, production-grade MLOps system that detects **covariate** and **concept drift** in real-time transaction streams and autonomously triggers **champion-challenger retraining, evaluation, and zero-downtime hot-reloading**.

---

## 📐 System Architecture

```
                    ┌─────────────────────────┐
   Transaction ───► │  FastAPI (/predict)     │───► Prediction + Prometheus Metrics
   Stream           └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Async WAL Logger        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Statistical Drift Engine│ (PSI & KS-Test, threshold > 0.25)
                    └────────────┬────────────┘
                                 │ (Drift Alert Triggered)
                                 ▼
                    ┌─────────────────────────┐
                    │ Retraining Pipeline     │ (Enriched Historical + Recent Stream)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ MLflow Model Registry   │ (Champion vs Challenger Benchmark)
                    └────────────┬────────────┘
                                 │ (Zero-Downtime Hot-Reload)
                                 ▼
                    ┌─────────────────────────┐
                    │ FastAPI Model Reload    │ (/reload-model)
                    └─────────────────────────┘
```

---

## 🌟 Key Features

1. **High-Throughput Serving**: Sub-10ms p95 latency FastAPI inference engine supporting single-record and high-density batch scoring.
2. **Real-Time Observability**: Built-in Prometheus metrics (`driftguard_http_requests_total`, `driftguard_prediction_latency_seconds`, `driftguard_fraud_score_distribution`, `driftguard_feature_psi`).
3. **Statistical Drift Detection**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test monitoring across raw and engineered dimensions.
4. **Autonomous Champion-Challenger Workflow**: Automatically retrains XGBoost upon critical drift detection and promotes challengers only if they outperform the incumbent champion on validation holdouts.
5. **Zero-Downtime Hot-Reloading**: In-memory model swapping via thread-safe `ModelManager` without dropping inflight requests.
6. **Full-Stack Containerization**: One-command launch with Docker Compose for FastAPI, MLflow Tracking Server, Prometheus, and Grafana.

---

## 📊 End-to-End Simulation Benchmark

Under high-velocity simulated covariate shift (macroeconomic inflation) and adversarial concept drift (micro-transaction card testing):

| Phase | Model Version | PR-AUC | F1 Score | Precision | Recall | Missed Fraud (FNR %) | Drift Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Pre-Drift (Clean Traffic)** | `v1` | **0.8501** | **0.8713** | 0.9778 | 0.7857 | 21.4% | No Drift |
| **2. Post-Drift (Stale Baseline)** | `v1` | 0.0404 | 0.0417 | 0.0303 | 0.0667 | **93.3%** | 🚨 **CRITICAL DRIFT** (PSI > 0.25) |
| **3. Post-Retrain (Promoted Challenger)** | `v2` | **0.4141** | **0.5055** | 0.5000 | 0.5111 | **48.8%** | 🛡️ **Recovered & Adapted** |

- ⚡ **Retraining Latency**: ~37 seconds autonomous execution
- 🛡️ **Missed Fraud Reduction**: 44.5% absolute FNR drop recovery

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/AbdelrahmanAboegela/driftguard-mlops.git
cd driftguard-mlops
pip install -r requirements.txt
```

### 2. Download Data & Train Champion Model
```bash
# Split temporal dataset (70% Train, 15% Holdout, 15% Stream)
python data/split_data.py

# Train baseline XGBoost champion and register to MLflow
python training/train.py
```

### 3. Launch Local Serving API
```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Production Simulation
```bash
python scripts/simulate_production.py
```

---

## 🐳 Running with Docker Compose

Launch the complete microservices stack (FastAPI, MLflow, Prometheus, Grafana) with a single command:

```bash
docker-compose up -d --build
```

- **Inference API**: [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) *(admin / admin)*

---

## 🧪 Testing Suite

Execute comprehensive unit, integration, and drift tests:

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```
driftguard-mlops/
├── data/
│   └── split_data.py          # Temporal train/validation/stream data splitter
├── training/
│   ├── feature_engineering.py # RobustScaler + cyclical time transformations
│   ├── train.py               # Baseline training & MLflow registration
│   └── evaluate.py            # PR-AUC, F1, FNR, threshold optimization
├── serving/
│   ├── app.py                 # FastAPI service with Prometheus middleware
│   ├── model_loader.py        # Thread-safe MLflow registry hot-reloader
│   ├── logger.py              # Asynchronous WAL transaction logger
│   └── schemas.py             # Pydantic v2 schemas & request validation
├── monitoring/
│   ├── drift_detector.py      # PSI & KS-test statistical evaluation engine
│   ├── inject_drift.py        # Synthetic covariate & concept drift generator
│   └── metrics_exporter.py    # Background periodic drift evaluator
├── orchestration/
│   ├── retrain_pipeline.py    # Champion-challenger validation & promotion
│   ├── rollback.py            # Disaster recovery rollback utility
│   └── dags/                  # Apache Airflow autonomous DAG definitions
├── load_testing/
│   └── locustfile.py          # High-concurrency traffic simulation
├── scripts/
│   ├── simulate_production.py # E2E drift injection and auto-retrain demo
│   └── run_load_test.py       # Performance benchmarking runner
├── grafana/                   # Pre-configured Grafana monitoring dashboards
├── prometheus/                # Prometheus scraping configuration
├── tests/                     # Comprehensive test suite (11 unit/integration tests)
├── Dockerfile                 # Multi-stage production container build
├── docker-compose.yml         # Containerized microservices definition
└── Makefile                   # Common development workflows
```

---

## 📜 License
MIT License.
