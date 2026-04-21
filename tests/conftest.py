import asyncio
from pathlib import Path

import joblib
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_CLEAN_CSV = _ROOT / "data" / "processed" / "customers_clean.csv"
_FEATURES_CSV = _ROOT / "data" / "features.csv"
_MODEL_PKL = _ROOT / "models" / "production_model.pkl"

_NON_FEATURE_COLS = {
    "customer_id", "signup_date", "last_purchase_date",
    "country", "gender", "churn",
}

# ---------------------------------------------------------------------------
# ML / data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_cleaned_df() -> pd.DataFrame:
    """Full cleaned customer dataset loaded from data/processed/customers_clean.csv."""
    return pd.read_csv(_CLEAN_CSV, parse_dates=["signup_date", "last_purchase_date"])


@pytest.fixture(scope="session")
def model_input_df() -> pd.DataFrame:
    """Feature matrix (first 20 rows) aligned with the production model's training columns."""
    df = pd.read_csv(_FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    return df[feature_cols].head(20)


@pytest.fixture(scope="session")
def production_model():
    """Production LightGBM model loaded from models/production_model.pkl."""
    return joblib.load(_MODEL_PKL)

TEST_DATABASE_URL = (
    "postgresql+asyncpg://retailai:password@localhost:5432/retailai_churn_test"
)

engine = create_async_engine(TEST_DATABASE_URL)
TestSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _drop():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    try:
        asyncio.run(_create())
    except Exception:
        pass
    yield
    try:
        asyncio.run(_drop())
    except Exception:
        pass


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()
