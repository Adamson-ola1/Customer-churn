# Bank Customer Churn Prediction — End-to-End ML Project

Predicts whether a bank customer will churn (close their account) using an
XGBoost model trained on 10,000 customer records, served via a FastAPI
backend and a React dashboard.

## Project Structure
```
customer-churn/
├── backend/          FastAPI service (main.py, schema.py)
├── charts/            EDA & evaluation PNG charts
├── data/
│   ├── raw/            Original dataset (never modified) - data.csv
│   ├── processed/       Cleaned + feature-engineered datasets
│   └── external/         Reserved for supplementary data sources
├── docs/               model_card.md, deployment_guide.md, technical_report.md
├── frontend/            React + Vite dashboard
├── notebooks/           exploration.ipynb (EDA + full walkthrough)
├── models/              trained_model.pkl, scaler.pkl, feature_names.pkl
├── outputs/              predictions.csv, metrics.json, plots/
├── src/                  preprocess.py, feature_engineering.py, train.py,
│                          evaluate.py, predict.py, utils.py
├── config.py             Paths, feature lists, hyperparameters
├── requirements.txt
└── main.py                Full pipeline entry point
```

## Quick Start

### 1. Run the ML pipeline
```bash
pip install -r requirements.txt
python main.py                # preprocess -> engineer features -> train -> evaluate
python main.py --predict      # ...then print a demo prediction
```
This regenerates `data/processed/*.csv`, `models/*.pkl`, `charts/*.png`,
and `outputs/metrics.json` / `outputs/predictions.csv`.

### 2. Start the API
```bash
uvicorn backend.main:app --reload --port 8000
```
Docs at http://localhost:8000/docs

### 3. Start the dashboard
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

See `docs/deployment_guide.md` for production deployment steps.

## Results Summary
Best model: **XGBoost** — ROC AUC 0.863, Accuracy 0.80, Recall 0.74 on a
held-out 20% test set. Full metrics and model comparison charts are in
`outputs/metrics.json` and `charts/model_comparison.png`. See
`docs/model_card.md` for details, limitations, and intended use, and
`docs/technical_report.md` for the full methodology write-up.

## Key EDA Findings
- Churn rises sharply with customer age.
- Customers in Germany churn more than those in France or Spain.
- Customers with 3-4 products and inactive members churn at higher rates.

## Tech Stack
- **ML:** pandas, scikit-learn, XGBoost, matplotlib/seaborn
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** React 18, Vite, React Router, Recharts, Axios
