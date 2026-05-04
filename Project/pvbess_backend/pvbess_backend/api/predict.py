"""
api/predict.py — ML Prediction endpoint
POST /api/predict  →  Random Forest prediction
GET  /api/predict/train  →  retrain model
GET  /api/predict/metrics  →  model performance metrics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from core.ml_model import predict, train_model, load_model, TARGET_NAMES

router = APIRouter()


class PredictRequest(BaseModel):
    solar_irr:   float = Field(..., ge=2.0, le=8.0)
    load_demand: float = Field(..., ge=10,  le=200)
    load_var:    float = Field(..., ge=5,   le=40)
    solar_var:   float = Field(..., ge=5,   le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "solar_irr": 5.5, "load_demand": 60,
                "load_var": 20,   "solar_var": 25
            }
        }


class PredictResponse(BaseModel):
    status:             str
    energy_usage_kwh:   float
    battery_size_kwh:   float
    pv_size_kwp:        float
    model_metrics:      dict
    feature_importance: dict
    model_info:         dict


@router.post("/predict", response_model=PredictResponse)
async def predict_sizing(req: PredictRequest):
    """
    Use trained Random Forest model to predict microgrid sizing.

    Inputs:  solar irradiance, load demand, load variability, solar variability
    Outputs: energy usage, battery size, PV size
    """
    try:
        result = predict(req.solar_irr, req.load_demand,
                         req.load_var,  req.solar_var)
        return PredictResponse(status="success", **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/metrics")
async def get_metrics():
    """Return trained model performance metrics."""
    try:
        _, _, meta = load_model()
        return {"status": "success", **meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/train")
async def retrain(background_tasks: BackgroundTasks, n_samples: int = 3000):
    """Retrain the Random Forest model in the background."""
    if n_samples < 500 or n_samples > 20000:
        raise HTTPException(status_code=400, detail="n_samples must be 500–20000")
    background_tasks.add_task(train_model, n_samples)
    return {"status": "training_started", "n_samples": n_samples,
            "message": "Model training started in background. Check /api/predict/metrics when done."}
