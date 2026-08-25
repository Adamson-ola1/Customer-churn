"""
evaluate.py
Evaluates every trained pipeline on the held-out test set, saves a
confusion matrix + ROC curve PNG per model to charts/, a bar-chart
model comparison, and writes outputs/metrics.json. Selects and saves
the best model (by ROC AUC) as models/trained_model.pkl.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

import config
from src.utils import get_logger, load_object, save_json, save_object, savefig

logger = get_logger(__name__)


def evaluate_model(name: str, pipe, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                            display_labels=["Retained", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    savefig(fig, config.CHARTS_DIR, f"confusion_matrix_{name}.png")
    savefig(fig, config.PLOTS_DIR, f"confusion_matrix_{name}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
    ax.set_title(f"ROC Curve — {name}")
    savefig(fig, config.CHARTS_DIR, f"roc_curve_{name}.png")
    savefig(fig, config.PLOTS_DIR, f"roc_curve_{name}.png")
    plt.close(fig)

    return metrics


def plot_model_comparison(results_df: pd.DataFrame):
    comparison = results_df.set_index("model")
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison[["accuracy", "precision", "recall", "f1_score", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set_title("Churn Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    savefig(fig, config.CHARTS_DIR, "model_comparison.png")
    savefig(fig, config.PLOTS_DIR, "model_comparison.png")
    plt.close(fig)


def run():
    X_test, y_test = load_object(config.MODELS_DIR / "_test_split.pkl")

    model_files = sorted(config.MODELS_DIR.glob(config.ALL_MODELS_GLOB))
    if not model_files:
        raise FileNotFoundError("No trained models found - run src/train.py first.")

    rows = []
    pipelines = {}
    for path in model_files:
        name = path.stem.replace("model_", "")
        pipe = load_object(path)
        pipelines[name] = pipe
        metrics = evaluate_model(name, pipe, X_test, y_test)
        rows.append(metrics)
        logger.info(f"{name:22s} " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "model"))

    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    plot_model_comparison(results_df)

    best_name = results_df.iloc[0]["model"]
    best_pipeline = pipelines[best_name]
    logger.info(f"Best model by ROC AUC: {best_name} (AUC={results_df.iloc[0]['roc_auc']:.4f})")

    save_object(best_pipeline, config.BEST_MODEL_PATH)
    # StandardScaler used inside the best pipeline, exported separately to match
    # the requested models/scaler.pkl layout (useful for external inspection).
    try:
        scaler = best_pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
        save_object(scaler, config.SCALER_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not export standalone scaler: {e}")

    metrics_payload = {
        "best_model": best_name,
        "results": results_df.to_dict(orient="records"),
    }
    save_json(metrics_payload, config.METRICS_PATH)
    logger.info(f"Saved metrics to {config.METRICS_PATH}")

    # test-set predictions for review
    preds = X_test.copy()
    preds["actual_churn"] = y_test.values
    preds["predicted_churn"] = best_pipeline.predict(X_test)
    preds["probability_of_churn"] = best_pipeline.predict_proba(X_test)[:, 1]
    preds.to_csv(config.PREDICTIONS_PATH, index=False)
    logger.info(f"Saved predictions to {config.PREDICTIONS_PATH}")

    return results_df


if __name__ == "__main__":
    run()
