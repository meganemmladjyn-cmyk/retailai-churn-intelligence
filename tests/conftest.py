import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app

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
