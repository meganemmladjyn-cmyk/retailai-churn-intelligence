"""
Generate 50,000 synthetic RetailAI customers using Faker.

Usage:
    python data/generate.py
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent))

fake = Faker("fr_FR")
random.seed(42)
np.random.seed(42)

N_CUSTOMERS = 50_000
CHURN_RATE = 0.25
OUTPUT_PATH = Path("data/synthetic_customers.csv")

CATEGORIES = [
    "Electronics", "Fashion", "Home & Garden", "Sports", "Beauty",
    "Books", "Toys", "Food & Drinks", "Jewelry", "Automotive",
]

CITIES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Lille",
    "Toulouse", "Nantes", "Strasbourg", "Nice", "Rennes",
]


def generate_customer(idx: int) -> dict:
    is_churned = random.random() < CHURN_RATE

    registration_date = fake.date_between(start_date="-3y", end_date="-30d")
    months_active = max(1, (date.today() - registration_date).days / 30)

    if is_churned:
        days_since = random.randint(90, 730)
        total_orders = random.randint(1, 8)
        email_open_rate = round(random.uniform(0.0, 0.20), 3)
        support_tickets_count = random.randint(0, 8)
        return_rate = round(random.uniform(0.20, 0.60), 3)
    else:
        days_since = random.randint(1, 89)
        total_orders = random.randint(3, 50)
        email_open_rate = round(random.uniform(0.15, 0.80), 3)
        support_tickets_count = random.randint(0, 3)
        return_rate = round(random.uniform(0.0, 0.35), 3)

    avg_order_value = round(random.uniform(15.0, 450.0), 2)
    total_spent = round(total_orders * avg_order_value * random.uniform(0.8, 1.2), 2)
    purchase_frequency = round(total_orders / months_active, 3)

    return {
        "customer_id": f"RET-{idx:06d}",
        "age": random.randint(18, 75),
        "gender": random.choice(["M", "F", "Non-binary"]),
        "city": random.choice(CITIES),
        "country": "France",
        "registration_date": registration_date,
        "last_purchase_date": date.today() - timedelta(days=days_since),
        "total_orders": total_orders,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "days_since_last_purchase": days_since,
        "purchase_frequency": purchase_frequency,
        "return_rate": return_rate,
        "email_open_rate": email_open_rate,
        "support_tickets_count": support_tickets_count,
        "preferred_category": random.choice(CATEGORIES),
        "is_churned": int(is_churned),
    }


def generate_dataset() -> pd.DataFrame:
    print(f"Generating {N_CUSTOMERS:,} synthetic customers...")
    customers = [generate_customer(i + 1) for i in range(N_CUSTOMERS)]
    df = pd.DataFrame(customers)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to  : {OUTPUT_PATH}")
    print(f"Shape     : {df.shape}")
    print(f"Churn rate: {df['is_churned'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_dataset()
