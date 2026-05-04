"""api/health.py — Health check and API info"""

from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter()


@router.get("/health")
async def health():
    """Health check — returns API status and model availability."""
    model_ready = os.path.exists("models/rf_model.joblib")
    return {
        "status":      "ok",
        "timestamp":   datetime.now().isoformat(),
        "version":     "1.0.0",
        "model_ready": model_ready,
        "endpoints": {
            "optimize":  "POST /api/optimize",
            "predict":   "POST /api/predict",
            "simulate":  "POST /api/simulate",
            "report":    "POST /api/simulate/report",
            "solar":     "GET  /api/solar/cities",
            "train":     "POST /api/predict/train",
            "docs":      "GET  /docs",
        }
    }
