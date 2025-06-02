# End-to-End MLOps Pipeline for Insurance Prediction 🚗💨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## 🎯 Project Overview

This project implements a complete end-to-end Machine Learning Operations (MLOps) pipeline for predicting vehicle insurance purchase likelihood. It demonstrates best practices in data versioning, experiment tracking, model serving, CI/CD, containerization, monitoring, and creating an interactive user interface. The goal is to build a robust, reproducible, and automated system for deploying and managing an ML model in a simulated production environment.

This was developed as part of the 2nd-year AI Engineering curriculum at ENSIAS, Mohammed V University.

---

## 🚀 Live Demos & Resources

* **FastAPI Backend API Endpoint:** [https://insurance-prediction-production.up.railway.app/](https://insurance-prediction-production.up.railway.app/)
* **API Documentation (Swagger UI):** [https://insurance-prediction-production.up.railway.app/docs](https://insurance-prediction-production.up.railway.app/docs)
* **Streamlit User Interface & Dashboard:** [https://insurance-prediction-mlops.streamlit.app/](https://insurance-prediction-mlops.streamlit.app/)
* **API Uptime Status Page:** [https://stats.uptimerobot.com/EDZuPKfmwD](https://stats.uptimerobot.com/EDZuPKfmwD)
* **GitHub Repository:** [https://github.com/hamzaben404/insurance-prediction](https://github.com/hamzaben404/insurance-prediction)
* **Project Report:** [Link to your PDF report if you host it, e.g., in the repo or a Google Drive link]

---

## ✨ Key Features Implemented

* **Automated ML Pipeline:**
    * Reproducible data processing & feature engineering (scripted, DVC versioned).
    * Systematic model training (XGBoost as the chosen model) and evaluation.
    * Experiment tracking with MLflow (parameters, metrics, artifacts).
* **Production-Ready API Service:**
    * FastAPI backend serving single and batch predictions.
    * Input/output validation using Pydantic models.
    * Health checks and monitoring endpoints (`/health`, `/simple_status`, `/monitor/metrics`, `/monitor/health/detailed`).
* **Interactive User Interface & Dashboard:**
    * Streamlit application for easy model interaction (prediction form).
    * Dashboard section showcasing data insights, model performance, and MLOps tool status.
* **MLOps Best Practices:**
    * **CI/CD:** Automated workflows with GitHub Actions for code quality checks (Black, isort, Flake8), security scanning (Bandit, Safety), testing (Pytest), Docker image building, and deployment to Railway.
    * **Containerization:** Dockerized application for consistent environments.
    * **Version Control:** Git for code, DVC for data and large model artifacts.
    * **Testing:** Comprehensive test suite (unit tests).
* **Deployment & Foundational Monitoring:**
    * Automated deployment of the FastAPI backend to Railway.app.
    * Deployment of the Streamlit UI to Streamlit Community Cloud.
    * Uptime monitoring with UptimeRobot.
    * Error tracking integration with Sentry (setup initiated).

---

## 🏗️ System Architecture

The project implements a full MLOps lifecycle. A detailed architecture diagram can be found in the project report or [here](./figures/mlops_architecture_with_ui.png) *(assuming you place your diagram in a `figures` folder and name it `mlops_architecture_with_ui.png`)*.

**Core Flow:**
Data (Versioned with DVC) -> Preprocessing & Feature Engineering -> Model Training (Tracked with MLflow) -> FastAPI Service (Dockerized) -> CI/CD Pipeline (GitHub Actions) -> Deployed API (Railway) & Deployed UI (Streamlit Cloud) -> Monitoring (UptimeRobot, Sentry)

---

## 🛠️ Technology Stack

* **Programming Language:** Python (3.10+)
* **Data Handling & ML:** Pandas, NumPy, Scikit-learn, XGBoost
* **API Development:** FastAPI, Pydantic, Uvicorn
* **MLOps & Experimentation:** MLflow, DVC
* **User Interface:** Streamlit, Requests
* **Containerization:** Docker
* **CI/CD & Version Control:** Git, GitHub, GitHub Actions
* **Deployment:** Railway.app (for API), Streamlit Community Cloud (for UI)
* **Code Quality & Testing:** Pytest, Flake8, Black, isort, Bandit, Safety
* **Monitoring:** UptimeRobot, Sentry, Python `logging`

---

## 📁 Project Structure Overview

```

insurance-prediction/
├── .github/workflows/      \# GitHub Actions CI/CD workflows
├── .streamlit/             \# Streamlit app configuration (e.g., config.toml for themes)
├── assets/                 \# Images and static files for Streamlit UI/report
├── config/                 \# Configuration files (e.g., for different environments)
├── data/                   \# Datasets (raw, processed, DVC tracked)
│   ├── raw/
│   └── processed/
├── docs/                   \# Project documentation (like this README, report, diagrams)
├── figures/                \# Saved plots and diagrams for the report/README
├── mlruns/                 \# MLflow experiment tracking data (usually gitignored)
├── models/                 \# Trained model artifacts (e.g., production model, comparison models)
├── notebooks/              \# Jupyter notebooks for EDA and experimentation
├── reports/                \# Generated reports (e.g., data quality HTML reports)
├── scripts/                \# Utility scripts
├── src/                    \# Source code
│   ├── api/                \# FastAPI application (main.py, routers, services, models)
│   ├── data/               \# Data processing scripts (load, preprocess, split)
│   ├── features/           \# Feature engineering scripts
│   ├── models/             \# Model training, evaluation, tuning scripts
│   └── utils/              \# Utility functions (e.g., logging)
├── tests/                  \# Pytest test suite (unit, integration, etc.)
├── app\_ui.py               \# Streamlit UI application code
├── Dockerfile              \# Docker configuration for the FastAPI app
├── dvc.yaml                \# DVC pipeline definition
├── requirements.txt        \# Python dependencies
├── Makefile                \# Optional: for automating common tasks
└── README.md               \# This file

````

---

## 🚀 Getting Started Locally

### Prerequisites
* Python (3.10 recommended)
* Git
* Docker (optional, for building/running the API container locally)
* DVC (`pip install dvc`)

### 1. Clone the Repository
```bash
git clone [https://github.com/hamzaben404/insurance-prediction.git](https://github.com/hamzaben404/insurance-prediction.git)
cd insurance-prediction
````

### 2\. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Data Setup with DVC

*(This step depends on whether you have a DVC remote configured. If not, ensure the data is present or your DVC pipeline can generate it).*
To pull data if a DVC remote is configured:

```bash
dvc pull
```

To reproduce the data pipeline if you have the raw data and DVC stages defined:

```bash
dvc repro
```

### 5\. Environment Variables (for API)

Create a `.env` file in the project root for local API execution (FastAPI backend). You can copy from `.env.example` if you create one.
Example `.env` content:

```env
# Example: No specific external service keys needed for basic local run
# If Sentry is to be tested locally:
# SENTRY_DSN="your_local_or_dev_sentry_dsn"
APP_VERSION="0.1.0-local"
RAILWAY_ENVIRONMENT="local"
LOG_LEVEL="DEBUG"
```

### 6\. Running Applications Locally

  * **FastAPI Backend API:**

    ```bash
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

    Access at `http://localhost:8000` and docs at `http://localhost:8000/docs`.

  * **Streamlit UI:**
    *(Ensure the `API_URL` in `app_ui.py` points to your local FastAPI instance, e.g., `http://localhost:8000/predictions/predict`, or set the `PREDICTION_API_URL` environment variable before running)*

    ```bash
    # Example: set environment variable for local API
    # export PREDICTION_API_URL="http://localhost:8000/predictions/predict" # Linux/macOS
    # $env:PREDICTION_API_URL="http://localhost:8000/predictions/predict" # PowerShell
    # set PREDICTION_API_URL="http://localhost:8000/predictions/predict" # Windows CMD

    streamlit run app_ui.py
    ```

    Access at the local URL Streamlit provides (usually `http://localhost:8501`).

  * **MLflow UI (to view experiments):**
    Navigate to your project root in the terminal and run:

    ```bash
    mlflow ui
    ```

    Access at `http://localhost:5000` (or the port MLflow indicates).

### 7\. Running Tests

```bash
pytest
```

To include coverage:

```bash
pytest --cov=src
```

-----

## 🔄 CI/CD Pipeline

This project uses GitHub Actions for CI/CD. Workflows are defined in `.github/workflows/`.
The main pipeline (`main.yml` or `code-quality.yml` & `test.yml`) typically includes:

1.  **Code Quality Checks:** Black, isort, Flake8.
2.  **Security Scans:** Bandit, Safety.
3.  **Unit Tests:** Pytest execution.
4.  **Docker Build:** Builds the FastAPI application Docker image.
5.  **Deployment:** Automatically deploys the validated image to Railway.app on pushes to the `main` branch.

-----

## 🔮 Future Work

Key areas for future development include:

  * Full verification and utilization of Sentry for error analysis.
  * Implementation of data drift detection mechanisms.
  * Development of model performance monitoring in production.
  * Creation of an automated model retraining pipeline with MLflow Model Registry.
  * Integration of advanced data validation tools like Great Expectations.
  * Further enhancements to the Streamlit UI/Dashboard.

-----

## 👨‍💻 Author

  * **Benatmane Hamza**
      * GitHub: [@hamzaben404](https://www.google.com/search?q=https://github.com/hamzaben404)

-----

## 📜 License
