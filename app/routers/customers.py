from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.customer import Customer
from app.schemas.customer import (
    CustomerCreate,
    CustomerListResponse,
    CustomerResponse,
    CustomerUpdate,
)

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("", response_model=CustomerListResponse)
async def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    churned_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    query = select(Customer)
    count_query = select(func.count()).select_from(Customer)

    if churned_only:
        query = query.where(Customer.is_churned == True)  # noqa: E712
        count_query = count_query.where(Customer.is_churned == True)  # noqa: E712

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(query.offset(offset).limit(page_size))
    customers = result.scalars().all()

    return CustomerListResponse(total=total, page=page, page_size=page_size, items=list(customers))


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Customer).where(Customer.customer_id == customer_id))
    customer = result.scalar_one_or_none()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.post("", response_model=CustomerResponse, status_code=201)
async def create_customer(payload: CustomerCreate, db: AsyncSession = Depends(get_db)):
    customer = Customer(**payload.model_dump())
    db.add(customer)
    await db.commit()
    await db.refresh(customer)
    return customer


@router.patch("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    payload: CustomerUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Customer).where(Customer.customer_id == customer_id))
    customer = result.scalar_one_or_none()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(customer, field, value)

    await db.commit()
    await db.refresh(customer)
    return customer
