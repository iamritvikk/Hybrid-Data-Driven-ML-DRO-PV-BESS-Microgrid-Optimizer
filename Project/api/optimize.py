"""
api/optimize.py — DRO Optimization endpoint
POST /api/optimize  →  run full DRO + method comparison
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from core.dro_engine import DROInputs, compute_dro, method_comparison

router = APIRouter()


class OptimizeRequest(BaseModel):
    solar_irr:   float = Field(..., ge=2.0, le=8.0,   description="Solar irradiance kWh/m²/day")
    load_demand: float = Field(..., ge=10,  le=200,   description="Daily load demand kWh")
    load_var:    float = Field(..., ge=5,   le=40,    description="Load variability %")
    solar_var:   float = Field(..., ge=5,   le=50,    description="Solar variability %")
    pv_cost:     float = Field(0.8, ge=0.3, le=1.5,  description="PV cost $/W")
    bat_cost:    float = Field(300, ge=150, le=600,   description="Battery cost $/kWh")
    target_rel:  float = Field(92,  ge=80,  le=99,    description="Target reliability %")
    delta:       float = Field(0.5, ge=0.1, le=1.0,   description="DRO robustness δ")
    data_months: int   = Field(6,   ge=1,   le=36,    description="Historical data months")
    lifetime:    int   = Field(20,  ge=10,  le=30,    description="Project lifetime years")

    class Config:
        json_schema_extra = {
            "example": {
                "solar_irr": 5.5, "load_demand": 60, "load_var": 20,
                "solar_var": 25,  "pv_cost": 0.8,    "bat_cost": 300,
                "target_rel": 92, "delta": 0.5,      "data_months": 6,
                "lifetime": 20
            }
        }


class OptimizeResponse(BaseModel):
    status:      str
    # Optimal sizing
    pv_cap_kwp:      float
    bat_cap_kwh:     float
    # Costs
    total_inr:       float
    det_total_inr:   float
    rob_total_inr:   float
    pv_cost_inr:     float
    bat_cost_inr:    float
    saving_pct:      float
    lcoe_inr:        float
    # Performance
    reliability_pct: float
    energy_loss_pct: float
    # DRO internals
    uncertainty_factor:   float
    data_scarcity_factor: float
    wasserstein_delta:    float
    # Charts
    tradeoff_curve:  list
    # Comparison
    method_comparison: dict


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    """
    Run DRO optimization and return optimal PV-BESS sizing.

    Implements Sharma et al. hybrid DRO + ML framework:
    - Computes Wasserstein-radius ambiguity set from δ and variability
    - Finds optimal PV and battery capacities
    - Returns cost-reliability tradeoff curve
    - Compares against deterministic and robust baselines
    """
    try:
        inp = DROInputs(
            solar_irr    = req.solar_irr,
            load_demand  = req.load_demand,
            load_var     = req.load_var,
            solar_var    = req.solar_var,
            pv_cost      = req.pv_cost,
            bat_cost     = req.bat_cost,
            target_rel   = req.target_rel,
            delta        = req.delta,
            data_months  = req.data_months,
            lifetime     = req.lifetime,
        )
        results  = compute_dro(inp)
        comp     = method_comparison(results)

        return OptimizeResponse(
            status                = "success",
            pv_cap_kwp            = results.pv_cap,
            bat_cap_kwh           = results.bat_cap,
            total_inr             = results.total_inr,
            det_total_inr         = results.det_total_inr,
            rob_total_inr         = results.rob_total_inr,
            pv_cost_inr           = results.pv_cost_inr,
            bat_cost_inr          = results.bat_cost_inr,
            saving_pct            = results.saving_pct,
            lcoe_inr              = results.lcoe_inr,
            reliability_pct       = results.reliability,
            energy_loss_pct       = results.energy_loss,
            uncertainty_factor    = results.uncertainty_factor,
            data_scarcity_factor  = results.data_scarcity_factor,
            wasserstein_delta     = results.wasserstein_delta,
            tradeoff_curve        = results.tradeoff_curve,
            method_comparison     = comp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
