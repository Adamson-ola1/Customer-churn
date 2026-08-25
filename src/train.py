"""
train.py
Splits the feature-engineered data, builds a preprocessing + model
Pipeline for each candidate model (Logistic Regression / XGBoost / MLP),
trains them, and saves every fitted pipeline to models/.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config
from src.utils import get_logger, save_object

logger = get_logger(__name__)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), config.NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), config.CATEGORICAL_FEATURES),
    ])


def get_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(**config.MODEL_PARAMS["logistic_regression"]),
        "xgboost": XGBClassifier(
            **config.MODEL_PARAMS["xgboost"], scale_pos_weight=scale_pos_weight,
        ),
        "mlp": MLPClassifier(**config.MODEL_PARAMS["mlp"]),
    }


def split_data(fe: pd.DataFrame):
    X = fe[config.FEATURES].copy()
    y = fe[config.TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_all(X_train, y_train) -> dict:
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos
    logger.info(f"scale_pos_weight (XGBoost): {scale_pos_weight:.2f}")

    models = get_models(scale_pos_weight)
    trained_pipelines = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        logger.info(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
    return trained_pipelines


def run():
    fe = pd.read_csv(config.FEATURE_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(fe)
    trained_pipelines = train_all(X_train, y_train)

    for name, pipe in trained_pipelines.items():
        save_object(pipe, config.MODELS_DIR / f"model_{name}.pkl")
    save_object(config.FEATURES, config.FEATURE_NAMES_PATH)

    # persist the split so evaluate.py can reuse it without re-splitting
    save_object((X_test, y_test), config.MODELS_DIR / "_test_split.pkl")

    logger.info(f"Saved {len(trained_pipelines)} trained pipelines to {config.MODELS_DIR}")
    return trained_pipelines, X_test, y_test


if __name__ == "__main__":
    run()
