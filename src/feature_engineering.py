"""
feature_engineering.py
Derives the engineered features used by the model from the cleaned dataset:
- balance_salary_ratio, zero_balance, has_multiple_products flags
- age_group and credit_score_category bins
Writes the feature-engineered dataset to data/processed/feature_data.csv.
"""
import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()

    # Balance relative to income
    fe["balance_salary_ratio"] = fe["balance"] / (fe["estimated_salary"] + 1)

    # Many customers carry a literal $0 balance (checking-only relationship)
    fe["zero_balance"] = (fe["balance"] == 0).astype(int)

    # Holding several products can signal either loyalty or overexposure
    fe["has_multiple_products"] = (fe["products_number"] > 1).astype(int)

    # Age bands - churn risk is highly non-linear with age
    fe["age_group"] = pd.cut(
        fe["age"], bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    ).astype(str)

    # Credit score bands
    fe["credit_score_category"] = pd.cut(
        fe["credit_score"], bins=[299, 579, 669, 739, 799, 900],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    ).astype(str)

    logger.info(f"Feature-engineered data shape: {fe.shape}")
    return fe


def apply_single_record(raw: dict) -> dict:
    """Reproduce the same engineered features for a single raw record
    (used by predict.py / the FastAPI backend at inference time)."""
    r = dict(raw)
    balance = float(r.get("balance", 0))
    salary = float(r.get("estimated_salary", 1))
    age = float(r.get("age", 0))
    credit_score = float(r.get("credit_score", 0))
    products = float(r.get("products_number", 0))

    r["balance_salary_ratio"] = balance / (salary + 1)
    r["zero_balance"] = int(balance == 0)
    r["has_multiple_products"] = int(products > 1)

    if age <= 30:
        r["age_group"] = "18-30"
    elif age <= 40:
        r["age_group"] = "31-40"
    elif age <= 50:
        r["age_group"] = "41-50"
    elif age <= 60:
        r["age_group"] = "51-60"
    else:
        r["age_group"] = "60+"

    if credit_score <= 579:
        r["credit_score_category"] = "Poor"
    elif credit_score <= 669:
        r["credit_score_category"] = "Fair"
    elif credit_score <= 739:
        r["credit_score_category"] = "Good"
    elif credit_score <= 799:
        r["credit_score_category"] = "Very Good"
    else:
        r["credit_score_category"] = "Excellent"

    return r


def run(save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    fe = engineer_features(df)
    if save:
        config.FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        fe.to_csv(config.FEATURE_DATA_PATH, index=False)
        logger.info(f"Saved feature-engineered data to {config.FEATURE_DATA_PATH}")
    return fe


if __name__ == "__main__":
    run()
