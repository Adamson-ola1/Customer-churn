"""
backend/main.py

FastAPI service exposing the trained customer churn model.

Endpoints:
- GET  /                 Root API information
- GET  /health           API and model health check
- GET  /model/info       Model metadata and evaluation metrics
- POST /predict         Single-customer churn prediction
- POST /predict/batch   Batch churn prediction

Run:
    uvicorn backend.main:app --reload --port 8000
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

import config

from backend.schema import (
    BatchPredictionResponse,
    CustomerRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

from src.predict import ChurnPredictor
from src.utils import get_logger, load_json


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = get_logger("backend")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Production-ready API for predicting customer churn "
        "using the best trained machine learning model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global model instance
# ---------------------------------------------------------------------------

_predictor: ChurnPredictor | None = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """
    Load the trained churn prediction model into memory.

    This function is intentionally separated from the startup event
    so it can also be reused for testing or future model management.
    """
    global _predictor

    if _predictor is not None:
        logger.info("Model already loaded.")
        return _predictor

    logger.info("Loading trained churn model...")

    model_start = time.time()

    _predictor = ChurnPredictor()

    model_elapsed = time.time() - model_start

    logger.info(
        f"Model loaded successfully in {model_elapsed:.2f}s"
    )

    return _predictor


def get_predictor() -> ChurnPredictor:
    """
    Return the loaded model.

    If the model was not loaded during startup, load it lazily.
    """
    global _predictor

    if _predictor is None:
        load_models()

    return _predictor


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    """
    Load the model when FastAPI starts and display startup duration.
    """

    startup_start = time.time()

    print("\n" + "=" * 70)
    print("  Bank Customer Churn Prediction API")
    print("=" * 70)

    print("  Initializing application...")

    try:
        load_models()

        elapsed = time.time() - startup_start

        print("-" * 70)
        print("  Model Status : LOADED")
        print(f"  API ready in : {elapsed:.2f}s")
        print("  API          : http://127.0.0.1:8000")
        print("  Docs         : http://127.0.0.1:8000/docs")
        print("  ReDoc        : http://127.0.0.1:8000/redoc")
        print("=" * 70 + "\n")

    except Exception as e:

        elapsed = time.time() - startup_start

        print("-" * 70)
        print("  Model Status : FAILED")
        print(f"  Startup time : {elapsed:.2f}s")
        print(f"  Error        : {e}")
        print("=" * 70 + "\n")

        logger.exception("Could not load model at startup.")


# ---------------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["Root"],
)
def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "status": "ok",
        "message": "Bank Customer Churn Prediction API is running.",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "prediction": "/predict",
        "batch_prediction": "/predict/batch",
    }


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health():
    """
    Check API and machine-learning model health.
    """

    model_loaded = _predictor is not None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info():
    """
    Return metadata, metrics and features for the selected model.
    """

    if not config.METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "metrics.json not found. "
                "Run the training and evaluation pipeline first."
            ),
        )

    metrics = load_json(config.METRICS_PATH)

    return ModelInfoResponse(
        best_model=metrics["best_model"],
        results=metrics["results"],
        features=list(config.FEATURES),
    )


# ---------------------------------------------------------------------------
# SINGLE PREDICTION
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(customer: CustomerRecord):
    """
    Predict churn probability for a single customer.
    """

    try:
        predictor = get_predictor()

        result = predictor.predict_one(
            customer.model_dump()
        )

        return PredictionResponse(**result)

    except Exception as e:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ---------------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------------

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Batch Prediction"],
)
def predict_batch(customers: list[CustomerRecord]):
    """
    Predict churn for multiple customers in one request.
    """

    try:
        predictor = get_predictor()

        records = [
            customer.model_dump()
            for customer in customers
        ]

        df = predictor.predict_batch(records)

        return BatchPredictionResponse(
            count=len(df),
            predictions=df.to_dict(orient="records"),
        )

    except Exception as e:

        logger.exception("Batch prediction failed.")

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )