"""
predict.py
Loads the saved best model (models/trained_model.pkl) and exposes a
ChurnPredictor class that reproduces feature engineering for single
records or batches, used by the CLI, main.py and the FastAPI backend.
"""
import pandas as pd

import config
from src.feature_engineering import apply_single_record
from src.utils import get_logger, load_object

logger = get_logger(__name__)


class ChurnPredictor:
    """Loads the saved best model and reproduces the feature-engineering
    step for a single raw customer record (or a batch of records)."""

    def __init__(self, model_path=None):
        model_path = model_path or config.BEST_MODEL_PATH
        self.pipeline = load_object(model_path)
        self.feature_names = load_object(config.FEATURE_NAMES_PATH)
        logger.info(f"Loaded model from {model_path}")

    def predict_one(self, raw: dict) -> dict:
        engineered = apply_single_record(raw)
        row = pd.DataFrame([engineered])[self.feature_names]
        proba = float(self.pipeline.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)
        return {
            "churn_prediction": pred,
            "churn_probability": round(proba, 4),
            "risk_level": self._risk_level(proba),
        }

    def predict_batch(self, records: list) -> pd.DataFrame:
        engineered = [apply_single_record(r) for r in records]
        df = pd.DataFrame(engineered)[self.feature_names]
        proba = self.pipeline.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = pd.DataFrame(records)
        out["churn_prediction"] = pred
        out["churn_probability"] = proba.round(4)
        out["risk_level"] = [self._risk_level(p) for p in proba]
        return out

    @staticmethod
    def _risk_level(proba: float) -> str:
        if proba >= 0.7:
            return "High"
        if proba >= 0.4:
            return "Medium"
        return "Low"
    
    
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    sample = {
        "credit_score": 590,
        "country": "Germany",
        "gender": "Female",
        "age": 58,
        "tenure": 2,
        "balance": 150000,
        "products_number": 3,
        "credit_card": 1,
        "active_member": 0,
        "estimated_salary": 60000,
    }
    print(predictor.predict_one(sample))
