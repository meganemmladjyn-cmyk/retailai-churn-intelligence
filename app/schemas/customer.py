from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CustomerBase(BaseModel):
    customer_id: str
    age: int
    gender: str
    city: str
    country: str = "France"
    registration_date: date
    last_purchase_date: Optional[date] = None
    total_orders: int = 0
    total_spent: float = 0.0
    avg_order_value: float = 0.0
    days_since_last_purchase: Optional[int] = None
    purchase_frequency: float = 0.0
    return_rate: float = 0.0
    email_open_rate: float = 0.0
    support_tickets_count: int = 0
    preferred_category: Optional[str] = None
    is_churned: bool = False


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(BaseModel):
    last_purchase_date: Optional[date] = None
    total_orders: Optional[int] = None
    total_spent: Optional[float] = None
    avg_order_value: Optional[float] = None
    days_since_last_purchase: Optional[int] = None
    purchase_frequency: Optional[float] = None
    return_rate: Optional[float] = None
    email_open_rate: Optional[float] = None
    support_tickets_count: Optional[int] = None
    is_churned: Optional[bool] = None
    churn_score: Optional[float] = None


class CustomerResponse(CustomerBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    churn_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class CustomerListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[CustomerResponse]
