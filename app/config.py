from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql+asyncpg://retailai:password@localhost:5432/retailai_churn"

    # API
    api_title: str = "RetailAI Churn Intelligence API"
    api_version: str = "0.1.0"
    debug: bool = False

    # ML artifacts
    model_path: str = "ml/artifacts/xgboost_churn.joblib"
    shap_explainer_path: str = "ml/artifacts/shap_explainer.joblib"

    # CORS
    allowed_origins: list[str] = ["*"]


settings = Settings()
