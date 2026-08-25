# Model Card — Bank Customer Churn Prediction

## Overview
- **Task:** Binary classification — predict whether a bank customer will churn (close their account).
- **Best model:** XGBoost classifier (selected by ROC AUC among Logistic Regression, XGBoost, MLP).
- **Intended use:** Support retention teams by flagging customers with elevated churn risk so they can be prioritized for outreach/offers.
- **Not intended for:** Automated account actions (e.g., auto-closing/denying services) or as the sole basis for credit decisions.

## Training Data
- **Source:** `data/raw/data.csv` — Bank Customer Churn Prediction dataset, 10,000 customers.
- **Features (raw):** credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary.
- **Target:** `churn` (1 = churned, 0 = retained). Class balance ≈ 20% churn / 80% retained.
- **Engineered features:** balance_salary_ratio, zero_balance flag, has_multiple_products flag, age_group bucket, credit_score_category bucket.
- **Split:** 80/20 stratified train/test split, random_state=42.

## Evaluation Results (held-out test set, n=2,000)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **XGBoost (best)** | 0.802 | 0.508 | 0.740 | 0.603 | **0.863** |
| Logistic Regression | 0.780 | 0.473 | 0.742 | 0.578 | 0.848 |
| MLP | 0.802 | 0.512 | 0.518 | 0.515 | 0.797 |

Charts (confusion matrices, ROC curves, model comparison bar chart) are saved under `charts/` and `outputs/plots/`.

## Key Drivers of Churn (from EDA)
- Churn rises sharply with **age**.
- Churn is markedly higher for customers in **Germany** vs. France/Spain.
- Customers holding **3–4 products** and **inactive members** churn at higher rates.
- Class imbalance (~20% positive) was handled via `class_weight="balanced"` (Logistic Regression) and `scale_pos_weight` (XGBoost).

## Limitations & Risks
- **Precision is moderate (~0.51)** — roughly half of customers flagged as "will churn" will not actually churn. The model favors recall to avoid missing at-risk customers, at the cost of more false positives; retention teams should treat scores as a **prioritization signal**, not a certainty.
- Trained on a single snapshot dataset from one bank; may not generalize to other markets, products, or time periods without retraining.
- No fairness/bias audit across country or gender has been performed beyond descriptive EDA — recommended before any high-stakes deployment.
- The dataset's `country` field only includes France, Spain, Germany.

## Maintenance
- Retrain periodically as new labeled churn data becomes available (see `main.py` for the full pipeline).
- Monitor prediction distribution and precision/recall drift in production; recalibrate the 0.5 decision threshold if the operating point should shift toward precision or recall.
