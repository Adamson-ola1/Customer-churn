"""
preprocess.py
Loads the raw Bank Customer Churn CSV and applies data cleaning:
- drops the non-predictive identifier column
- standardises categorical text
- checks for missing values / duplicates
Writes the cleaned dataset to data/processed/processed_data.csv.
"""
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw churn dataframe and return a copy ready for feature engineering."""
    df = df.copy()

    missing = df.isnull().sum()
    if missing.sum():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found.")

    dup_rows = df.duplicated().sum()
    dup_ids = df[config.ID_COL].duplicated().sum() if config.ID_COL in df.columns else 0
    logger.info(f"Duplicate rows: {dup_rows} | Duplicate customer_id: {dup_ids}")
    df = df.drop_duplicates()

    # customer_id carries no predictive signal - drop after logging for traceability
    if config.ID_COL in df.columns:
        df = df.drop(columns=[config.ID_COL])

    # standardise categorical text (strip whitespace, consistent casing)
    for col in config.RAW_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # basic sanity bounds - clip impossible values rather than dropping rows
    if "age" in df.columns:
        df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(300, 900)

    df = df.reset_index(drop=True)
    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def run(save: bool = True) -> pd.DataFrame:
    df = load_raw_data()
    df_clean = clean_data(df)
    if save:
        config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(config.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved processed data to {config.PROCESSED_DATA_PATH}")
    return df_clean


if __name__ == "__main__":
    run()
