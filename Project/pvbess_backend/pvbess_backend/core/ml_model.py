"""
=============================================================================
core/ml_model.py
Random Forest ML Model — Training + Prediction
Inputs : solar irradiance, load demand, load variability, solar variability
Outputs: energy usage, battery size, PV size
=============================================================================
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple

MODEL_PATH  = "models/rf_model.joblib"
SCALER_PATH = "models/rf_scaler.joblib"
METRICS_PATH = "models/rf_metrics.joblib"

FEATURE_NAMES = [
    "Solar Irradiance (kWh/m²/day)",
    "Load Demand (kWh/day)",
    "Load Variability (%)",
    "Solar Variability (%)",
]
TARGET_NAMES = ["Energy Usage (kWh)", "Battery Size (kWh)", "PV Size (kWp)"]


def generate_training_data(n: int = 3000, seed: int = 42):
    """Generate physics-based synthetic training data."""
    np.random.seed(seed)

    solar_irr   = np.random.uniform(2.0, 8.0,  n)
    load_demand = np.random.uniform(10,  200,   n)
    load_var    = np.random.uniform(5,   40,    n)
    solar_var   = np.random.uniform(5,   50,    n)

    # Physics-based targets
    energy = (
        load_demand * (1 + load_var/100 * 0.45) * (1 + solar_var/100 * 0.12)
        + np.random.normal(0, load_demand * 0.03, n)
    )
    energy = np.maximum(energy, 5.0)

    battery = (
        load_demand * 0.35
        + load_demand * (load_var/100) * 0.55
        + load_demand * (solar_var/100) * 0.28
        + np.random.normal(0, load_demand * 0.02, n)
    )
    battery = np.maximum(battery, 2.0)

    eff_irr = solar_irr * (1 - solar_var/100 * 0.42)
    pv = (
        energy / (eff_irr * 0.85 + 0.01) * (1 + solar_var/100 * 0.18)
        + np.random.normal(0, 0.25, n)
    )
    pv = np.maximum(pv, 1.0)

    X = np.column_stack([solar_irr, load_demand, load_var, solar_var])
    y = np.column_stack([energy, battery, pv])
    return X, y


def train_model(n_samples: int = 3000) -> Dict:
    """Train the Random Forest model and save to disk."""
    print(f"Training RF model on {n_samples} samples...")
    X, y = generate_training_data(n_samples)

    split = int(0.8 * n_samples)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestRegressor(
        n_estimators     = 200,
        max_depth        = 12,
        min_samples_leaf = 3,
        max_features     = "sqrt",
        random_state     = 42,
        n_jobs           = -1,
    )
    rf.fit(X_tr_s, y_tr)
    pred = rf.predict(X_te_s)

    metrics = {}
    for i, tname in enumerate(TARGET_NAMES):
        r2   = r2_score(y_te[:, i], pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_te[:, i], pred[:, i])))
        mae  = float(mean_absolute_error(y_te[:, i], pred[:, i]))
        mape = float(np.mean(np.abs((y_te[:, i] - pred[:, i]) / (y_te[:, i] + 1e-8))) * 100)
        metrics[tname] = dict(r2=round(r2,4), rmse=round(rmse,3),
                               mae=round(mae,3), mape=round(mape,2))
        print(f"  {tname}: R²={r2:.4f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")

    fi = {
        tname: {fname: round(float(v), 4) for fname, v in zip(FEATURE_NAMES, imp)}
        for tname, imp in zip(TARGET_NAMES, rf.estimators_[0].feature_importances_.reshape(1,-1).repeat(3,0))
    }
    # Proper feature importance from full forest
    fi_mean = np.mean([e.feature_importances_ for e in rf.estimators_], axis=0)
    fi_all = {fname: round(float(v), 4) for fname, v in zip(FEATURE_NAMES, fi_mean)}

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf,      MODEL_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump({"metrics": metrics, "feature_importance": fi_all,
                 "n_train": split, "n_test": len(X)-split}, METRICS_PATH)

    print(f"Model saved to {MODEL_PATH}")
    return metrics


def load_model() -> Tuple[RandomForestRegressor, StandardScaler, Dict]:
    """Load trained model from disk. Train first if not found."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found — training now...")
        train_model()
    rf     = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta   = joblib.load(METRICS_PATH)
    return rf, scaler, meta


def predict(solar_irr: float, load_demand: float,
            load_var: float, solar_var: float) -> Dict:
    """Run ML prediction for a single input set."""
    rf, scaler, meta = load_model()
    x = np.array([[solar_irr, load_demand, load_var, solar_var]])
    x_s = scaler.transform(x)
    pred = rf.predict(x_s)[0]
    return {
        "energy_usage_kwh": round(float(pred[0]), 2),
        "battery_size_kwh": round(float(pred[1]), 2),
        "pv_size_kwp":      round(float(pred[2]), 2),
        "model_metrics":    meta["metrics"],
        "feature_importance": meta["feature_importance"],
        "model_info": {
            "type":      "Random Forest",
            "n_trees":   rf.n_estimators,
            "max_depth": rf.max_depth,
            "n_train":   meta["n_train"],
            "n_test":    meta["n_test"],
        }
    }
