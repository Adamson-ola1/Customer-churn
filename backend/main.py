"""
backend/main.py
FastAPI service exposing the trained churn model:
- POST /predict        single-customer prediction
- POST /predict/batch   batch prediction from a list of customers
- GET  /model/info      metadata & metrics for the best model
- GET  /health          liveness check
Run with: uvicorn backend.main:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # allow `import config`, `src.*`

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from backend.schema import (
    BatchPredictionResponse, CustomerRecord, HealthResponse,
    ModelInfoResponse, PredictionResponse,
)
from src.predict import ChurnPredictor
from src.utils import get_logger, load_json

logger = get_logger("backend")

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Serves predictions from the best trained churn model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


@app.on_event("startup")
def on_startup():
    try:
        get_predictor()
        logger.info("Model loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Could not load model at startup: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = _predictor is not None
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found - run the training pipeline first.")
    metrics = load_json(config.METRICS_PATH)
    return ModelInfoResponse(
        best_model=metrics["best_model"],
        results=metrics["results"],
        features=list(config.FEATURES),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRecord):
    try:
        predictor = get_predictor()
        result = predictor.predict_one(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: list[CustomerRecord]):
    try:
        predictor = get_predictor()
        records = [c.model_dump() for c in customers]
        df = predictor.predict_batch(records)
        return BatchPredictionResponse(
            count=len(df),
            predictions=df.to_dict(orient="records"),
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {"message": "Bank Customer Churn Prediction API", "docs": "/docs"}
