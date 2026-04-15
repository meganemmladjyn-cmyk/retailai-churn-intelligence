from pydantic import BaseModel


class PredictionRequest(BaseModel):
    customer_id: str


class ShapExplanation(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str  # "positive" | "negative"


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk: str  # "low" | "medium" | "high"
    top_factors: list[ShapExplanation]
    recommendation: str


class BatchPredictionRequest(BaseModel):
    customer_ids: list[str]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
