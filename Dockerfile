# ==========================================================
# Customer Churn Prediction - Backend Docker Image
# ==========================================================

FROM python:3.11-slim

# ----------------------------------------------------------
# Environment Variables
# ----------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------
# Working Directory
# ----------------------------------------------------------

WORKDIR /app

# ----------------------------------------------------------
# System Dependencies
# ----------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Install Python Dependencies
# ----------------------------------------------------------

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# Copy Application
# ----------------------------------------------------------

COPY backend ./backend
COPY src ./src
COPY models ./models
COPY outputs ./outputs
COPY config.py .

# ----------------------------------------------------------
# Expose FastAPI Port
# ----------------------------------------------------------

EXPOSE 8000

# ----------------------------------------------------------
# Start FastAPI
# ----------------------------------------------------------

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]