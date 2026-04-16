# CLAUDE.md — RetailAI Churn Intelligence System

## Project context

**RetailAI Churn Intelligence System** is a customer churn prediction 
system for SME e-commerce. It generates 50,000 synthetic customers 
(Faker), trains an XGBoost + SHAP model, and exposes predictions via 
a FastAPI REST API backed by PostgreSQL, deployed on Render via Docker.

Stack: Python 3.12 · FastAPI · SQLAlchemy (async) · PostgreSQL 16 · 
XGBoost · SHAP · Alembic · Docker · GitHub Actions · Render

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
- Always respond in French
- Always write code in English — variable names, function names, 
  file names, comments, commit messages, docstrings

### Code style
- Use pathlib.Path for all file system operations, never os.path
- Add type hints to every function signature (parameters + return)
- Write docstrings in English on all public functions and classes
- Use pydantic-settings BaseSettings for config, never hardcode secrets
- Prefer async/await throughout the FastAPI layer

### Development workflow
- Run python data/generate.py before python ml/train.py
- Run alembic upgrade head before starting API on fresh database
- pip install -e . resolves app, ml, data, migrations as packages

### Commit format
- Conventional Commits obligatoire
- Format : type(scope): description en anglais, max 72 caractères
- Types : feat, fix, docs, test, refactor, chore

### Environment
- OS : Windows 11, PowerShell
- Terminal Claude Code : PowerShell
- Node.js installé via LTS officiel nodejs.org
- Toujours fermer/rouvrir PowerShell après modification du PATH

---

## Errors log

| Jour | Contexte | Erreur | Fix appliqué |
|------|----------|--------|--------------|
| J0 | Claude Code launch | ERR_BAD_RESPONSE sur platform.claude.com depuis Guadeloupe | setx ANTHROPIC_BASE_URL "https://api.anthropic.com" avant claude |
| J0 | pip install -e . | Multiple top-level packages dans flat-layout | Correction setup.py avec find_packages(include=[...]) |
| J0 | git branch | Branch main vide après git init | Premier commit requis avant que la branch existe |
| J1 | Auth Claude Code | Conflit token claude.ai / API key à chaque session | setx ANTHROPIC_BASE_URL permanent + claude /logout |
| J1 | Claude Code mode | Auto mode proposé | Refusé — garder le contrôle manuel sur chaque action |