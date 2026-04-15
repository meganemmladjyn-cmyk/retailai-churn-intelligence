from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)

    # Demographics
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(20))
    city: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(100), default="France")

    # Behavioral features
    registration_date: Mapped[date] = mapped_column(Date)
    last_purchase_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    total_orders: Mapped[int] = mapped_column(Integer, default=0)
    total_spent: Mapped[float] = mapped_column(Float, default=0.0)
    avg_order_value: Mapped[float] = mapped_column(Float, default=0.0)
    days_since_last_purchase: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    purchase_frequency: Mapped[float] = mapped_column(Float, default=0.0)  # orders/month
    return_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 0–1

    # Engagement features
    email_open_rate: Mapped[float] = mapped_column(Float, default=0.0)  # 0–1
    support_tickets_count: Mapped[int] = mapped_column(Integer, default=0)
    preferred_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Target & ML output
    is_churned: Mapped[bool] = mapped_column(Boolean, default=False)
    churn_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
