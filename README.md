# RetailAI Churn Intelligence

![CI](https://github.com/meganemmladjyn-cmyk/retailai-churn-intelligence/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-tuned-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

**Customer churn prediction system for SME e-commerce — FastAPI + LightGBM + Streamlit, deployed on Render.**

Live demo: [YOUR_STREAMLIT_URL](YOUR_STREAMLIT_URL)

---

## Project Overview

RetailAI Churn Intelligence predicts which customers are likely to churn so that a CRM system can act on them automatically before they leave. The end user is not a human analyst — it is an automated CRM pipeline that consumes a REST API and triggers targeted re-engagement campaigns.

**Dataset:** 49,920 synthetic customers generated with Faker, calibrated to realistic SME e-commerce distributions (order frequency, support ticket volume, recency segments, revenue tiers).

**Model output:** For each customer, the API returns:
- `churn_probability` — a float between 0 and 1
- `risk_level` — `LOW`, `MEDIUM`, or `HIGH` based on probability thresholds
- SHAP feature contributions explaining the individual prediction

**Key design decision — recall over precision:** In a CRM use case, missing a churner (false negative) costs more than sending one unnecessary re-engagement email (false positive). The model and Optuna objective were tuned to maximize recall while keeping F1 acceptable. LightGBM Tuned achieves **recall 0.911** on the test set.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  data/generate.py  →  src/data/cleaner.py  →  src/features/    │
│  (Faker, 49,920 rows)   (nulls, types,         engineering.py   │
│                          UTF-8 fix)            (ratios, bins,   │
│                                                 recency)        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                         │
│                                                                 │
│  LogisticRegression  →  XGBoost  →  LightGBM + Optuna (50 tr.) │
│                                                                 │
│  All runs tracked in MLflow (local SQLite + mlartifacts/)       │
│  Best model serialized to models/production_model.pkl           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌──────────────────┐    ┌──────────────────────┐
        │    FastAPI       │    │      Streamlit        │
        │  (async, SHAP)   │    │  (4 pages, real-time  │
        │  PostgreSQL 16   │    │   predictor, KPIs)    │
        │  Alembic async   │    │                       │
        └────────┬─────────┘    └──────────┬───────────┘
                 │                         │
                 └───────────┬─────────────┘
                             ▼
                     ┌──────────────┐
                     │    Render    │
                     │   (Docker)   │
                     │  GitHub CI   │
                     └──────────────┘
```

---

## Results

All metrics measured on a held-out test set (20% stratified split). CV scores are 5-fold on the training set.

| Model               | CV ROC-AUC | Test ROC-AUC | F1     | Recall    | Train time |
|---------------------|------------|--------------|--------|-----------|------------|
| LogisticRegression  | 0.9714     | 0.9702       | 0.8227 | 0.7643    | 0.09 s     |
| XGBoost             | 0.9694     | 0.9687       | 0.8127 | 0.8932    | 0.74 s     |
| LightGBM Tuned      | 0.9710     | 0.9700       | 0.8140 | **0.911** | 0.49 s     |

**Best model:** LightGBM Tuned (`models/production_model.pkl`)

Best Optuna hyperparameters:
```json
{
  "n_estimators": 257,
  "max_depth": 3,
  "learning_rate": 0.02650,
  "num_leaves": 46,
  "scale_pos_weight": 3.139
}
```

---

## Tech Stack

| Layer               | Technology                                           |
|---------------------|------------------------------------------------------|
| Language            | Python 3.12                                          |
| ML                  | LightGBM, XGBoost, scikit-learn, SHAP                |
| Hyperparameter      | Optuna (50 trials, TPE sampler)                      |
| Experiment tracking | MLflow (local SQLite backend)                        |
| Data generation     | Faker                                                |
| API                 | FastAPI (async), Pydantic v2, Uvicorn                |
| Database            | PostgreSQL 16, SQLAlchemy (async), Alembic           |
| Frontend            | Streamlit                                            |
| Containerization    | Docker, Docker Compose                               |
| CI/CD               | GitHub Actions, Render                               |
| Package manager     | uv                                                   |
| Linter              | Ruff                                                 |

---

## Setup & Installation

**Prerequisites:** Python 3.12+, `uv`, Docker (optional for full stack)

```bash
# 1. Clone the repository
git clone https://github.com/meganemmladjyn-cmyk/retailai-churn-intelligence.git
cd retailai-churn-intelligence

# 2. Create and activate virtual environment
uv venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and model paths
```

`.env.example` reference:
```env
DATABASE_URL=postgresql+asyncpg://retailai:password@localhost:5432/retailai_churn
DEBUG=false
MODEL_PATH=models/production_model.pkl
SHAP_EXPLAINER_PATH=ml/artifacts/shap_explainer.joblib
```

---

## How to Run

### Full ML pipeline (data → model)

```bash
# Generate synthetic data (49,920 customers)
python data/generate.py

# Clean and validate
python src/data/cleaner.py

# Feature engineering
python src/features/run_features.py

# Train all models + Optuna tuning + MLflow tracking
python src/models/train.py
```

### FastAPI server

```bash
# Apply database migrations
alembic upgrade head

# Start API (development)
uvicorn app.main:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs
```

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

### Docker (full stack: API + PostgreSQL)

```bash
docker compose up --build
```

### Docker (Streamlit only)

```bash
docker compose -f docker-compose.streamlit.yml up --build
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Specific modules
python -m pytest tests/test_ml.py tests/test_model.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## Feature Engineering

Features are constructed in `src/features/engineering.py` from raw customer records. The pipeline produces ratio features, behavioral bins, and recency segments that outperform raw counts.

### Top 5 features by SHAP importance

| Feature              | Description                                    | Business rationale                                                                          |
|----------------------|------------------------------------------------|---------------------------------------------------------------------------------------------|
| `support_ratio`      | Support tickets / total orders                 | High ratio signals friction — customers who contact support often churn disproportionately  |
| `total_orders`       | Lifetime order count                           | Low order count combined with recency is the strongest churn signal for SME e-commerce     |
| `recency_segment`    | Ordinal bin: days since last purchase (0–3)    | Customers inactive >90 days are 3× more likely to churn; binning outperforms raw day count |
| `support_tickets`    | Raw support ticket count                       | Absolute volume captures high-value customers with many orders but also many complaints     |
| `purchase_frequency` | Orders per active month                        | Frequency drop precedes churn by 30–60 days; captures declining engagement early           |

All features pass through a scikit-learn `Pipeline` with `StandardScaler` before training.

---

## Key Decisions & Lessons Learned

### Data leakage detected and fixed (ROC-AUC 1.0 → 0.97)

Early model runs produced ROC-AUC of 1.0. Investigation revealed that the synthetic label generation used `days_since_purchase` as a direct deterministic input to the churn flag, which then appeared as a feature. Fix: the label is now generated from a probabilistic logistic function with Gaussian noise, and leaky columns are dropped before the feature pipeline runs.

### LightGBM over XGBoost (+2% recall)

XGBoost achieved recall 0.893. LightGBM with `scale_pos_weight` tuned by Optuna reached 0.911. The gain comes from LightGBM's leaf-wise tree growth being more aggressive on minority class splits when the positive weight is correctly calibrated. LightGBM also trains 33% faster on this dataset (0.49 s vs 0.74 s).

### Optuna over GridSearch

GridSearch over a 4-parameter grid with 5-fold CV would require ~2,000 fits. Optuna's TPE sampler finds a better solution in 50 trials (~250 fits) by learning the parameter landscape. The final `learning_rate` (0.02650) and `num_leaves` (46) are values a coarse grid would never hit.

### Probabilistic labels over deterministic

Deterministic labels (`churned = 1 if inactive > 90 days`) produced perfectly separable data — the model learns a threshold rule, not a probability. Switching to probabilistic labels (sigmoid of a weighted feature sum + Gaussian noise) forces the model to learn a real distribution, producing calibrated `churn_probability` outputs usable for CRM prioritization by score, not just binary flag.

### Windows UTF-8 fix

Faker generates names with accented characters. On Windows, `open()` defaults to `cp1252`, which raises `UnicodeEncodeError` when writing CSV. Fix applied in `data/generate.py` and `src/data/cleaner.py`: all file I/O uses `encoding="utf-8"` explicitly. `PYTHONPATH=${{ github.workspace }}` is also set in CI to ensure consistent module resolution across platforms.

---

## File Structure

```
retailai-churn-intelligence/
│
├── app/                            # FastAPI application
│   ├── main.py                     # App factory, middleware, lifespan
│   ├── config.py                   # pydantic-settings BaseSettings
│   ├── database.py                 # Async SQLAlchemy engine + session
│   ├── streamlit_app.py            # Streamlit dashboard (4 pages)
│   ├── models/
│   │   └── customer.py             # SQLAlchemy ORM model
│   ├── routers/
│   │   ├── customers.py            # CRUD endpoints
│   │   ├── predictions.py          # POST /predict, batch predict
│   │   └── health.py               # GET /health
│   ├── schemas/
│   │   ├── customer.py             # Pydantic request/response schemas
│   │   └── prediction.py           # PredictionResponse with SHAP values
│   └── services/
│       ├── predictor.py            # Model loading + inference
│       └── explainer.py            # SHAP TreeExplainer wrapper
│
├── data/
│   ├── generate.py                 # Faker synthetic data (49,920 rows)
│   ├── raw/
│   │   └── customers.csv           # Raw generated data
│   └── processed/
│       ├── customers_clean.csv     # After cleaner.py
│       └── features.csv            # After engineering.py
│
├── src/
│   ├── data/
│   │   ├── cleaner.py              # Null handling, type coercion, UTF-8
│   │   ├── loader.py               # Typed CSV loader
│   │   └── quality.py              # Validation checks (schema, ranges)
│   ├── features/
│   │   ├── engineering.py          # Feature construction (ratios, bins)
│   │   └── run_features.py         # CLI entry point for feature pipeline
│   └── models/
│       ├── baseline.py             # LogisticRegression baseline
│       ├── compare.py              # Side-by-side model evaluation
│       ├── train.py                # LightGBM training + MLflow logging
│       ├── tuning.py               # Optuna objective + study
│       └── run_training.py         # CLI entry point for full training
│
├── ml/                             # Legacy pipeline (Days 3–4)
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── models/                         # Serialized model artifacts
│   ├── production_model.pkl        # Active production model
│   ├── lgbm_churn.pkl
│   ├── tuned_model.pkl
│   ├── xgboost.pkl
│   ├── baseline.pkl
│   └── best_params.json            # Optuna best hyperparameters
│
├── migrations/                     # Alembic async migrations
│   ├── env.py
│   └── versions/
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_features.ipynb           # Feature importance analysis
│   └── 03_modeling.ipynb           # Model comparison + SHAP plots
│
├── tests/
│   ├── conftest.py                 # Fixtures, async test client
│   ├── test_ml.py                  # ML pipeline integration tests
│   ├── test_data_quality.py        # Data validation tests
│   ├── test_features.py            # Feature engineering unit tests
│   ├── test_model.py               # Model inference + threshold tests
│   └── test_api.py                 # FastAPI endpoint tests
│
├── .github/workflows/
│   ├── ci.yml                      # Test + lint on push/PR
│   └── deploy.yml                  # Auto-deploy to Render on main
│
├── Dockerfile                      # FastAPI production image
├── Dockerfile.streamlit            # Streamlit production image
├── docker-compose.yml              # API + PostgreSQL stack
├── docker-compose.streamlit.yml    # Streamlit standalone
├── render.yaml                     # Render deployment config
├── alembic.ini                     # Alembic configuration
├── pyproject.toml                  # Project metadata + tool config
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Dev + test dependencies
├── setup.py                        # Editable install (pip install -e .)
└── .env.example                    # Environment variable template
```

---

## API Reference

### `POST /api/v1/predictions/single`

```json
// Request
{
  "customer_id": "cust_001",
  "total_orders": 3,
  "support_tickets": 5,
  "days_since_purchase": 95,
  "purchase_frequency": 0.4,
  "avg_order_value": 42.0
}

// Response
{
  "customer_id": "cust_001",
  "churn_probability": 0.847,
  "risk_level": "HIGH",
  "shap_values": {
    "support_ratio": 0.312,
    "recency_segment": 0.289,
    "total_orders": -0.154
  }
}
```

### `GET /health`

```json
{ "status": "ok", "model": "LightGBM Tuned", "version": "1.0.0" }
```

### All endpoints

| Method | Endpoint                        | Description                        |
|--------|---------------------------------|------------------------------------|
| GET    | `/health`                       | Health check + model version       |
| GET    | `/api/v1/customers`             | List customers (paginated)         |
| POST   | `/api/v1/customers`             | Create a customer record           |
| GET    | `/api/v1/customers/{id}`        | Get customer by ID                 |
| PATCH  | `/api/v1/customers/{id}`        | Update customer fields             |
| POST   | `/api/v1/predictions/single`    | Predict churn for one customer     |
| POST   | `/api/v1/predictions/batch`     | Predict churn for a list           |

---

## CI/CD

GitHub Actions runs two jobs on every push to `main` and on all pull requests:

- **test** — installs dependencies via `uv`, generates synthetic data, runs the full ML pipeline, then executes all pytest suites with `PYTHONPATH` set to the workspace root
- **lint** — runs `ruff check src/ app/` (zero tolerance for errors)

Render deploys automatically on merge to `main` via `deploy.yml` and `render.yaml`.

---

## License

MIT
