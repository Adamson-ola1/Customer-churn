"""
utils.py
Small, dependency-light helper functions reused across the pipeline
(logging, saving/loading artifacts, figure saving).
"""
import json
import logging
import sys
from pathlib import Path

import joblib


def get_logger(name: str = "churn_pipeline") -> logging.Logger:
    """Return a configured stdout logger (idempotent - no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_object(obj, path: Path) -> None:
    """Persist any Python object (model, scaler, encoder, list...) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path):
    """Load an object previously stored with `save_object`."""
    return joblib.load(Path(path))


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(Path(path)) as f:
        return json.load(f)


def savefig(fig, directory: Path, filename: str) -> Path:
    """Save a matplotlib figure to `directory/filename` at 300 dpi."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return out_path
