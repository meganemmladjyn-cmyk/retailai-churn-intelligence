# CLAUDE.md — RetailAI Churn Intelligence System

## Project context

**RetailAI Churn Intelligence System** is a customer churn prediction system for SME e-commerce.
It generates 50,000 synthetic customers (Faker), trains an XGBoost + SHAP model, and exposes
predictions via a FastAPI REST API backed by PostgreSQL, deployed on Render via Docker.

Stack: Python 3.12 · FastAPI · SQLAlchemy (async) · PostgreSQL 16 · XGBoost · SHAP ·
Alembic · Docker · GitHub Actions · Render

Key paths:
- `app/` — FastAPI application (routers, models, schemas, services)
- `ml/` — feature definition, training pipeline, evaluation
- `data/` — synthetic data generation (Faker, 50k rows)
- `migrations/` — Alembic async migrations
- `tests/` — pytest + pytest-asyncio + httpx
- `notebooks/` — exploratory notebooks (EDA, features, modeling)

---

## Permanent rules

### Language
- **Always respond in French** — all explanations, questions, and answers.
- **Always write code in English** — variable names, function names, file names, comments,
  commit messages, and docstrings.

### Code style
- Use `pathlib.Path` for all file system operations — never `os.path`.
- Add type hints to every function signature (parameters + return type).
- Write docstrings in English on all public functions, classes, and modules.
- Use `pydantic-settings` `BaseSettings` for configuration — never hardcode secrets.
- Prefer `async`/`await` throughout the FastAPI layer.

### Development workflow
- Run `python data/generate.py` before `python ml/train.py`.
- Run `alembic upgrade head` before starting the API against a fresh database.
- `pip install -e .` resolves `app`, `ml`, `data`, `migrations` as top-level packages.

---

## Errors log

| Date | File | Error | Fix applied |
|------|------|-------|-------------|
| —    | —    | —     | —           |
