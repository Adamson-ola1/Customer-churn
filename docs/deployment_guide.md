# Deployment Guide

## 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Docker

## 2. Backend (FastAPI)

### Local setup
```bash
cd customer-churn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the full ML pipeline once, to generate models/, outputs/, charts/
python main.py

# Start the API
uvicorn backend.main:app --reload --port 8000
```
The API will be live at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + model-loaded check |
| GET | `/model/info` | Best model name, metrics, feature list |
| POST | `/predict` | Single customer churn prediction |
| POST | `/predict/batch` | Batch churn prediction (list of customers) |

### Production notes
- Set `allow_origins` in `backend/main.py`'s CORS middleware to your actual frontend domain (not `*`).
- Run behind a process manager, e.g.:
  ```bash
  gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
  ```
- Re-run `python main.py` (or just `src/train.py` + `src/evaluate.py`) whenever the model needs to be refreshed, then restart the API process to pick up the new `models/trained_model.pkl`.

## 3. Frontend (React + Vite)

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL if different from default
npm run dev             # local dev server on http://localhost:5173
```

### Production build
```bash
npm run build     # outputs static assets to frontend/dist
npm run preview   # sanity-check the production build locally
```
Deploy the contents of `frontend/dist` to any static host (Netlify, Vercel, S3+CloudFront, Nginx). Ensure `VITE_API_BASE_URL` points at the deployed backend URL at build time.

## 4. Environment Variables
| Variable | Where | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | frontend/.env | Base URL the frontend uses to call the backend API |

## 5. Suggested Docker layout (optional)
- `backend/Dockerfile` — Python slim image, `pip install -r requirements.txt`, `CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- `frontend/Dockerfile` — multi-stage: `npm run build` then serve `dist/` with nginx.
- `docker-compose.yml` — wire the two services together with the frontend calling the backend service name internally.

## 6. Retraining Checklist
1. Drop the new raw CSV into `data/raw/data.csv` (keep the original as a backup — this folder should never be modified in place otherwise).
2. Run `python main.py` to regenerate processed data, models, charts, and `outputs/metrics.json`.
3. Review `docs/model_card.md` metrics table and update it if performance materially changed.
4. Restart the backend service.
