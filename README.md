# RetailAI Churn Intelligence System

![CI](https://github.com/meganemmladjyn-cmyk/retailai-churn-intelligence/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

> Customer churn prediction system for SME e-commerce — XGBoost · SHAP · FastAPI · PostgreSQL · Docker

---

## Overview

**RetailAI Churn Intelligence System** identifies e-commerce customers at risk of churning and
provides actionable retention recommendations powered by an explainable machine learning model.

A customer is considered **churned** if they have made no purchase in the last **90 days**.
The system scores each customer with a churn probability (0–1) and explains the top contributing
factors using SHAP values.

---

## Features

- Synthetic dataset of **50,000 customers** generated with Faker
- **XGBoost** classifier with `scale_pos_weight` for class imbalance
- **SHAP** explainability — top 5 factors per prediction
- **FastAPI** REST API with async SQLAlchemy + PostgreSQL
- **Alembic** async migrations
- **Docker Compose** one-command local setup
- **CI/CD** via GitHub Actions → Render

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/meganemmladjyn-cmyk/retailai-churn-intelligence.git
cd retailai-churn-intelligence
pip install -e .
pip install -r requirements.txt

# 2. Environment
cp .env.example .env

# 3. Generate synthetic data & train model
python data/generate.py
python ml/train.py

# 4. Start services
docker-compose up --build

# 5. Run migrations
docker-compose exec app alembic upgrade head
```

API docs available at `http://localhost:8000/docs`

---

## Project structure

```
retailai-churn-intelligence/
├── app/            # FastAPI application
├── ml/             # XGBoost training pipeline + SHAP
├── data/           # Synthetic data generation (Faker)
├── migrations/     # Alembic async migrations
├── notebooks/      # EDA, feature engineering, modeling
├── tests/          # pytest + httpx
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/customers` | List customers (paginated) |
| `POST` | `/api/v1/customers` | Create a customer |
| `GET` | `/api/v1/customers/{id}` | Get customer by ID |
| `PATCH` | `/api/v1/customers/{id}` | Update customer |
| `POST` | `/api/v1/predictions/single` | Predict churn for one customer |
| `POST` | `/api/v1/predictions/batch` | Predict churn for a list of customers |

---

## Exploratory Data Analysis

**Dataset:** 49,920 customers × 12 features after cleaning (duplicates and missing values removed).

**Class imbalance:** Churn rate of **15.3%** (85/15 split). XGBoost is trained with `scale_pos_weight` to compensate; primary evaluation metric is **AUC-ROC**.

**Feature correlations with churn:**

| Feature | Pearson r | Direction |
|---|---|---|
| `total_orders` | −0.46 | More orders → less churn |
| `total_spent` | −0.44 | Higher spend → less churn |
| `avg_order_value` | −0.16 | Weaker signal |

**Distributions:** `total_spent` and `avg_order_value` are right-skewed — log-transform applied; **RobustScaler** recommended for monetary features to limit outlier influence.

**Business insight:** Customers with **2+ support tickets AND ≤ 3 orders** represent the highest churn risk segment. Recommended action: deploy a targeted retention campaign within **30 days of last purchase** for this cohort.

---

## Model performance (target)

| Metric | Target |
|---|---|
| AUC-ROC | > 0.85 |
| Recall (churned) | Maximised |
| Churn rate (dataset) | ~25 % |
