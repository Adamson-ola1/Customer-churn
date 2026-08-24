"""
config.py
Central configuration: filesystem paths, feature lists, constants and
model hyperparameters shared by every script in `src/`.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"
FEATURE_DATA_PATH = DATA_DIR / "processed" / "feature_data.csv"

MODELS_DIR = ROOT_DIR / "models"
CHARTS_DIR = ROOT_DIR / "charts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

for d in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "external",
          MODELS_DIR, CHARTS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
TARGET = "churn"
ID_COL = "customer_id"

RAW_NUMERIC_COLS = ["credit_score", "age", "tenure", "balance",
                     "products_number", "estimated_salary"]
RAW_CATEGORICAL_COLS = ["country", "gender"]
RAW_BINARY_COLS = ["credit_card", "active_member"]

ENGINEERED_NUMERIC_COLS = [
    "credit_score", "age", "tenure", "balance", "products_number",
    "estimated_salary", "balance_salary_ratio", "zero_balance",
    "has_multiple_products", "credit_card", "active_member",
]
ENGINEERED_CATEGORICAL_COLS = [
    "country", "gender", "age_group", "credit_score_category",
]

FEATURES = ENGINEERED_NUMERIC_COLS + ENGINEERED_CATEGORICAL_COLS
NUMERIC_FEATURES = ENGINEERED_NUMERIC_COLS
CATEGORICAL_FEATURES = ENGINEERED_CATEGORICAL_COLS

# --------------------------------------------------------------------------
# Modeling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_PARAMS = {
    "logistic_regression": dict(
        penalty="elasticnet", l1_ratio=0.5, solver="saga",
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
    ),
    "xgboost": dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
    "mlp": dict(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
        random_state=RANDOM_STATE,
    ),
}

BEST_MODEL_PATH = MODELS_DIR / "trained_model.pkl"
SCALER_MODEL_PATH = MODELS_DIR / "scaler.pkl"  # kept for the requested layout;
# scaling actually lives inside the sklearn Pipeline stored in trained_model.pkl
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
ALL_MODELS_GLOB = "model_*.pkl"
