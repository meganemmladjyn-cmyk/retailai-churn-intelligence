from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.customer import Customer
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.explainer import ShapExplainer
from app.services.predictor import ChurnPredictor

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Lazy-loaded singletons — models are loaded on first prediction request
_predictor = ChurnPredictor()
_explainer = ShapExplainer()


@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    payload: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Customer).where(Customer.customer_id == payload.customer_id)
    )
    customer = result.scalar_one_or_none()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    prediction = _predictor.predict(customer)
    top_factors = _explainer.explain(customer)

    customer.churn_score = prediction["churn_probability"]
    await db.commit()

    return PredictionResponse(
        customer_id=customer.customer_id,
        top_factors=top_factors,
        **prediction,
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Customer).where(Customer.customer_id.in_(payload.customer_ids))
    )
    customers = result.scalars().all()
    if not customers:
        raise HTTPException(status_code=404, detail="No customers found")

    predictions = []
    for customer in customers:
        pred = _predictor.predict(customer)
        top_factors = _explainer.explain(customer)
        customer.churn_score = pred["churn_probability"]
        predictions.append(
            PredictionResponse(
                customer_id=customer.customer_id,
                top_factors=top_factors,
                **pred,
            )
        )

    await db.commit()
    return BatchPredictionResponse(predictions=predictions, total=len(predictions))
