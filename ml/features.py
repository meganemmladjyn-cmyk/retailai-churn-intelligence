# Numerical features used by the XGBoost model (training + inference)
FEATURE_COLUMNS: list[str] = [
    "age",
    "total_orders",
    "total_spent",
    "avg_order_value",
    "days_since_last_purchase",
    "purchase_frequency",
    "return_rate",
    "email_open_rate",
    "support_tickets_count",
]

TARGET_COLUMN: str = "is_churned"
