# Technical Report — Bank Customer Churn Prediction

## 1. Problem Statement
Predict which bank customers are likely to churn (close their account) using
demographic, account, and engagement attributes, so retention teams can
intervene proactively.

## 2. Dataset
- **Source file:** `data/raw/data.csv` (10,000 rows, 12 columns).
- **Raw columns:** customer_id, credit_score, country, gender, age, tenure,
  balance, products_number, credit_card, active_member, estimated_salary, churn.
- **Target distribution:** ~20% churn (1), ~80% retained (0) — moderately imbalanced.
- **Data quality:** no missing values, no duplicate rows or customer IDs found
  during cleaning (`src/preprocess.py`).

## 3. Methodology

### 3.1 Cleaning (`src/preprocess.py`)
- Dropped `customer_id` (identifier, no predictive signal).
- Standardized categorical text casing/whitespace.
- Clipped implausible values (age bounds 18–100, credit_score bounds 300–900).
- Output: `data/processed/processed_data.csv`.

### 3.2 Exploratory Data Analysis (`notebooks/exploration.ipynb`)
Key findings:
- Churn increases sharply with customer **age**.
- Customers in **Germany** churn at a notably higher rate than France/Spain.
- Customers with **3–4 products** and **inactive members** show elevated churn.
- Correlation heatmap confirmed no severe multicollinearity among numeric features.

### 3.3 Feature Engineering (`src/feature_engineering.py`)
| Feature | Description |
|---|---|
| `balance_salary_ratio` | balance / (estimated_salary + 1) |
| `zero_balance` | 1 if balance == 0 (checking-only relationship) |
| `has_multiple_products` | 1 if products_number > 1 |
| `age_group` | Binned age (18-30, 31-40, 41-50, 51-60, 60+) |
| `credit_score_category` | Binned credit score (Poor..Excellent) |

Output: `data/processed/feature_data.csv`, 16 columns.

### 3.4 Preprocessing Pipeline (`src/train.py`)
`ColumnTransformer`:
- Numeric: median imputation → `StandardScaler`
- Categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

Wrapped in an sklearn `Pipeline` with the estimator, so the exact same
transformation is applied at training and inference time — no train/serve skew.

### 3.5 Models Trained
| Model | Key hyperparameters |
|---|---|
| Logistic Regression | ElasticNet penalty, `class_weight="balanced"`, saga solver |
| XGBoost | 300 trees, max_depth=4, lr=0.05, `scale_pos_weight` for imbalance |
| MLP (Neural Network) | Two hidden layers (64, 32), ReLU |

Train/test split: 80/20, stratified by target, `random_state=42`.

### 3.6 Evaluation (`src/evaluate.py`)
Metrics computed on the untouched 20% test set (n=2,000): accuracy, precision,
recall, F1, ROC AUC. For each model: confusion matrix and ROC curve PNGs
saved to `charts/` and `outputs/plots/`; a grouped bar chart compares all
three models on all five metrics (`model_comparison.png`).

## 4. Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.8015 | 0.5084 | 0.7396 | 0.6026 | **0.8634** |
| Logistic Regression | 0.7795 | 0.4734 | 0.7420 | 0.5780 | 0.8485 |
| MLP | 0.8015 | 0.5121 | 0.5184 | 0.5153 | 0.7965 |

**XGBoost was selected as the production model** (`models/trained_model.pkl`)
based on the highest ROC AUC, which best reflects ranking quality under class
imbalance and is the most relevant metric when the deployment use case is
prioritizing a ranked list of at-risk customers rather than a hard yes/no cutoff.

## 5. Inference Pipeline
`src/predict.py` defines `ChurnPredictor`, which:
1. Loads `models/trained_model.pkl` and `models/feature_names.pkl`.
2. Reproduces the exact feature-engineering step from section 3.3 for a raw record.
3. Returns `churn_prediction` (0/1), `churn_probability`, and a `risk_level`
   bucket (Low < 0.4 ≤ Medium < 0.7 ≤ High).

This same class powers the CLI (`python -m src.predict`), `main.py --predict`,
and the FastAPI backend's `/predict` and `/predict/batch` endpoints.

## 6. Reproducing This Report
```bash
pip install -r requirements.txt
python main.py          # preprocess -> feature-engineer -> train -> evaluate
```
All numbers above are regenerated at `outputs/metrics.json` on every run.

## 7. Future Work
- Hyperparameter tuning (grid/Bayesian search) for the XGBoost model.
- SHAP-based feature-importance / explainability report per prediction.
- Threshold tuning per business cost of false positives vs. false negatives.
- Monitoring for data/label drift once deployed against live traffic.
