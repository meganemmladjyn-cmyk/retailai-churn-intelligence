"""
Generate 50,000 synthetic retail customers using Faker.

Three personas drive realistic purchase behaviour; intentional data-quality
issues are injected afterwards for pipeline-testing purposes.

Usage:
    python data/generate.py

Output:
    data/raw/customers.csv
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

fake = Faker()

TODAY = date.today()
N_CUSTOMERS = 50_000
OUTPUT_PATH = Path("data/raw/customers.csv")

# Persona distribution
PERSONA_WEIGHTS: dict[str, float] = {
    "active": 0.60,
    "at_risk": 0.25,
    "churned": 0.15,
}

COUNTRIES: list[str] = [
    "France", "Germany", "United Kingdom", "Spain", "Italy",
    "Netherlands", "Belgium", "Switzerland", "Sweden", "Poland",
]

GENDERS: list[str] = ["M", "F", "Non-binary"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_persona() -> str:
    """Sample a persona name according to PERSONA_WEIGHTS."""
    r = random.random()
    cumulative = 0.0
    for name, weight in PERSONA_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return name
    return "churned"  # fallback


def _build_customer(idx: int) -> dict:
    """Build a single clean customer record driven by persona.

    Args:
        idx: 1-based integer used to build a unique customer_id.

    Returns:
        Dictionary with all customer fields.
    """
    persona = _sample_persona()

    # signup_date: random point between 4 years ago and 6 months ago
    signup_date: date = fake.date_between(start_date="-4y", end_date="-6m")

    if persona == "active":
        days_since = random.randint(1, 29)
        total_orders = random.randint(10, 80)
        total_spent = round(random.uniform(500.0, 8_000.0), 2)
    elif persona == "at_risk":
        days_since = random.randint(30, 89)
        total_orders = random.randint(3, 20)
        total_spent = round(random.uniform(100.0, 1_500.0), 2)
    else:  # churned
        days_since = random.randint(90, 730)
        total_orders = random.randint(1, 8)
        total_spent = round(random.uniform(20.0, 400.0), 2)

    last_purchase_date: date = TODAY - timedelta(days=days_since)
    avg_order_value: float = round(total_spent / total_orders, 2)

    return {
        "customer_id": f"CUST-{idx:06d}",
        "signup_date": signup_date,
        "last_purchase_date": last_purchase_date,
        "total_orders": total_orders,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "return_rate": round(random.uniform(0.0, 0.50), 4),
        "support_tickets": random.randint(0, 10),
        "country": random.choice(COUNTRIES),
        "age": random.randint(18, 80),
        "gender": random.choice(GENDERS),
        # Churn label: 1 if no purchase in the last 90 days
        "churn": 1 if days_since >= 90 else 0,
    }


# ---------------------------------------------------------------------------
# Public single-record generator (aligned with ml.features schema)
# ---------------------------------------------------------------------------


def generate_customer(idx: int) -> dict:
    """Generate a single customer record aligned with FEATURE_COLUMNS and TARGET_COLUMN.

    Args:
        idx: 1-based integer used to build a unique customer_id (format RET-XXXXXX).

    Returns:
        Dictionary with customer_id, all FEATURE_COLUMNS, and is_churned target.
    """
    persona = _sample_persona()

    signup_date: date = fake.date_between(start_date="-4y", end_date="-6m")

    if persona == "active":
        days_since = random.randint(1, 29)
        total_orders = random.randint(10, 80)
        total_spent = round(random.uniform(500.0, 8_000.0), 2)
    elif persona == "at_risk":
        days_since = random.randint(30, 89)
        total_orders = random.randint(3, 20)
        total_spent = round(random.uniform(100.0, 1_500.0), 2)
    else:
        days_since = random.randint(90, 730)
        total_orders = random.randint(1, 8)
        total_spent = round(random.uniform(20.0, 400.0), 2)

    months_active = max(1, (TODAY - signup_date).days // 30)
    avg_order_value: float = round(total_spent / total_orders, 2)
    purchase_frequency: float = round(total_orders / months_active, 4)

    return {
        "customer_id": f"RET-{idx:06d}",
        "age": random.randint(18, 75),
        "total_orders": total_orders,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "days_since_last_purchase": days_since,
        "purchase_frequency": purchase_frequency,
        "return_rate": round(random.uniform(0.0, 1.0), 4),
        "email_open_rate": round(random.uniform(0.0, 1.0), 4),
        "support_tickets_count": random.randint(0, 10),
        "is_churned": 1 if days_since >= 90 else 0,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_clean_dataset(n: int) -> pd.DataFrame:
    """Generate n clean customer records with no quality issues.

    Args:
        n: Number of rows to generate.

    Returns:
        DataFrame with n rows and all expected columns.
    """
    print(f"  Generating {n:,} base customer records...")
    records = [_build_customer(i + 1) for i in range(n)]
    return pd.DataFrame(records)


def inject_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Inject intentional data-quality issues into the dataset.

    Issues injected:
    - 5 % missing values on age, gender, country
    - 200 duplicate rows appended
    - 50 rows with negative total_spent
    - 30 rows with signup_date in the future
    - support_tickets dtype converted from int to float
    - 20 rows with return_rate > 1.0 (physically impossible)

    Args:
        df: Clean DataFrame produced by generate_clean_dataset.

    Returns:
        DataFrame with quality issues injected (length = len(df) + 200).
    """
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(SEED)

    # 1. 5 % missing values on age, gender, country
    missing_count = int(n * 0.05)
    for col in ("age", "gender", "country"):
        missing_idx = rng.choice(n, size=missing_count, replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. 200 duplicate rows (appended at the end)
    dup_source_idx = rng.choice(n, size=200, replace=False)
    duplicates = df.iloc[dup_source_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 3. 50 rows with negative total_spent
    neg_idx = rng.choice(n, size=50, replace=False)
    df.loc[neg_idx, "total_spent"] = -(df.loc[neg_idx, "total_spent"].abs())

    # 4. 30 rows with signup_date in the future
    future_idx = rng.choice(n, size=30, replace=False)
    for i in future_idx.tolist():
        df.loc[i, "signup_date"] = TODAY + timedelta(days=random.randint(1, 365))

    # 5. support_tickets stored as float instead of int
    df["support_tickets"] = df["support_tickets"].astype(float)

    # 6. 20 rows with return_rate > 1.0
    invalid_return_idx = rng.choice(n, size=20, replace=False)
    df.loc[invalid_return_idx, "return_rate"] = (
        rng.uniform(1.01, 3.0, size=20).round(4)
    )

    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to a CSV file, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_report(df: pd.DataFrame, path: Path, base_n: int) -> None:
    """Print a structured summary of the generated dataset.

    Args:
        df: Final DataFrame (including injected issues).
        path: Path where the CSV was saved.
        base_n: Number of rows before duplicates were appended.
    """
    churn_rate = df["churn"].mean()
    n_active = ((df["churn"] == 0) & (df["last_purchase_date"] >= TODAY - timedelta(days=29))).sum()
    n_at_risk = (
        (df["churn"] == 0)
        & (df["last_purchase_date"] < TODAY - timedelta(days=29))
        & (df["last_purchase_date"] >= TODAY - timedelta(days=89))
    ).sum()
    n_churned = (df["churn"] == 1).sum()

    missing_age = df["age"].isna().sum()
    missing_gender = df["gender"].isna().sum()
    missing_country = df["country"].isna().sum()
    dup_rows = len(df) - base_n
    negative_spend = (df["total_spent"] < 0).sum()
    future_signup = (pd.to_datetime(df["signup_date"]) > pd.Timestamp(TODAY)).sum()
    is_float_tickets = df["support_tickets"].dtype == float
    invalid_return = (df["return_rate"] > 1.0).sum()

    sep = "=" * 56
    print()
    print(sep)
    print("  DATASET SUMMARY")
    print(sep)
    print(f"  Output path          : {path}")
    print(f"  Shape                : {df.shape}")
    print(f"  Churn rate           : {churn_rate:.2%}")
    print()
    print(f"  Active customers     : {n_active:,}")
    print(f"  At-risk customers    : {n_at_risk:,}")
    print(f"  Churned customers    : {n_churned:,}")
    print()
    print("  INJECTED QUALITY ISSUES")
    print("-" * 56)
    print(f"  Missing age          : {missing_age:,} rows")
    print(f"  Missing gender       : {missing_gender:,} rows")
    print(f"  Missing country      : {missing_country:,} rows")
    print(f"  Duplicate rows added : {dup_rows:,}")
    print(f"  Negative total_spent : {negative_spend:,} rows")
    print(f"  Future signup_date   : {future_signup:,} rows")
    print(f"  support_tickets=float: {is_float_tickets}")
    print(f"  return_rate > 1.0    : {invalid_return:,} rows")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate, inject quality issues, save, and report the dataset."""
    print("RetailAI - Synthetic customer data generation")
    print(f"Target  : {N_CUSTOMERS:,} customers  ->  {OUTPUT_PATH}")
    print()

    df_clean = generate_clean_dataset(N_CUSTOMERS)
    print("  Injecting data-quality issues...")
    df_final = inject_quality_issues(df_clean)
    print(f"  Saving to {OUTPUT_PATH} ...")
    save_dataset(df_final, OUTPUT_PATH)
    print_report(df_final, OUTPUT_PATH, base_n=N_CUSTOMERS)


if __name__ == "__main__":
    main()
