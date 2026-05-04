"""
api/simulate.py — Full simulation endpoint (DRO + ML combined)
POST /api/simulate  →  run both DRO and ML and return combined report
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from datetime import date
from core.dro_engine import DROInputs, compute_dro, method_comparison
from core.ml_model import predict

router = APIRouter()


class SimulateRequest(BaseModel):
    # System inputs
    solar_irr:   float = Field(..., ge=2.0, le=8.0)
    load_demand: float = Field(..., ge=10,  le=200)
    load_var:    float = Field(..., ge=5,   le=40)
    solar_var:   float = Field(..., ge=5,   le=50)
    pv_cost:     float = Field(0.8, ge=0.3, le=1.5)
    bat_cost:    float = Field(300, ge=150, le=600)
    target_rel:  float = Field(92,  ge=80,  le=99)
    delta:       float = Field(0.5, ge=0.1, le=1.0)
    data_months: int   = Field(6,   ge=1,   le=36)
    lifetime:    int   = Field(20,  ge=10,  le=30)
    # Report metadata
    project_name:     str = Field("Microgrid Project", max_length=100)
    project_location: str = Field("Not specified",     max_length=100)
    prepared_by:      str = Field("MicrogridAI",       max_length=100)
    prepared_for:     str = Field("Not specified",      max_length=100)


def fmt_inr(v: float) -> str:
    if v >= 1e7: return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5: return f"₹{v/1e5:.1f} L"
    return f"₹{v:,.0f}"


@router.post("/simulate")
async def full_simulation(req: SimulateRequest):
    """
    Run complete DRO + ML simulation and return combined results.
    Includes optimal sizing, method comparison, and ML predictions.
    """
    try:
        inp = DROInputs(
            solar_irr=req.solar_irr, load_demand=req.load_demand,
            load_var=req.load_var,   solar_var=req.solar_var,
            pv_cost=req.pv_cost,     bat_cost=req.bat_cost,
            target_rel=req.target_rel, delta=req.delta,
            data_months=req.data_months, lifetime=req.lifetime,
        )
        dro_res  = compute_dro(inp)
        comp     = method_comparison(dro_res)
        ml_res   = predict(req.solar_irr, req.load_demand,
                           req.load_var,  req.solar_var)

        return {
            "status": "success",
            "dro":    {
                "pv_cap_kwp":          dro_res.pv_cap,
                "bat_cap_kwh":         dro_res.bat_cap,
                "total_inr":           dro_res.total_inr,
                "reliability_pct":     dro_res.reliability,
                "energy_loss_pct":     dro_res.energy_loss,
                "saving_pct":          dro_res.saving_pct,
                "lcoe_inr":            dro_res.lcoe_inr,
                "wasserstein_delta":   dro_res.wasserstein_delta,
                "uncertainty_factor":  dro_res.uncertainty_factor,
                "tradeoff_curve":      dro_res.tradeoff_curve,
            },
            "ml":     {
                "energy_usage_kwh":    ml_res["energy_usage_kwh"],
                "battery_size_kwh":    ml_res["battery_size_kwh"],
                "pv_size_kwp":         ml_res["pv_size_kwp"],
                "model_info":          ml_res["model_info"],
            },
            "comparison": comp,
            "meta": {
                "project_name":     req.project_name,
                "project_location": req.project_location,
                "prepared_by":      req.prepared_by,
                "prepared_for":     req.prepared_for,
                "date":             date.today().isoformat(),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate/report", response_class=PlainTextResponse)
async def generate_report(req: SimulateRequest):
    """Generate a plain-text technical report of the simulation results."""
    try:
        inp = DROInputs(
            solar_irr=req.solar_irr, load_demand=req.load_demand,
            load_var=req.load_var,   solar_var=req.solar_var,
            pv_cost=req.pv_cost,     bat_cost=req.bat_cost,
            target_rel=req.target_rel, delta=req.delta,
            data_months=req.data_months, lifetime=req.lifetime,
        )
        dro = compute_dro(inp)
        ml  = predict(req.solar_irr, req.load_demand,
                      req.load_var,  req.solar_var)
        comp = method_comparison(dro)

        report = f"""
{'='*65}
{req.project_name.upper()}
PV-BESS MICROGRID FEASIBILITY REPORT
{'='*65}
Prepared For   : {req.prepared_for}
Project Site   : {req.project_location}
Prepared By    : {req.prepared_by}
Date           : {date.today().strftime('%d %B %Y')}
Framework      : Hybrid DRO + ML (Sharma et al.)
{'='*65}

1. SYSTEM INPUTS
{'-'*55}
Solar Irradiance        : {req.solar_irr} kWh/m²/day
Daily Load Demand       : {req.load_demand} kWh/day
Load Variability        : {req.load_var}%
Solar Variability       : {req.solar_var}%
Historical Data         : {req.data_months} months
PV Cost                 : ₹{req.pv_cost * 84:.0f}/W
Battery Cost            : ₹{req.bat_cost * 84:.0f}/kWh
Target Reliability      : {req.target_rel}%
DRO Robustness (δ)      : {req.delta}
Project Lifetime        : {req.lifetime} years

2. DRO OPTIMIZATION RESULTS
{'-'*55}
Optimal PV Capacity     : {dro.pv_cap} kWp
Optimal Battery Size    : {dro.bat_cap} kWh
Total Lifecycle Cost    : {fmt_inr(dro.total_inr)}
PV System Cost          : {fmt_inr(dro.pv_cost_inr)}
Battery System Cost     : {fmt_inr(dro.bat_cost_inr)}
Achieved Reliability    : {dro.reliability}%
Energy Loss             : {dro.energy_loss}%
LCOE                    : ₹{dro.lcoe_inr}/kWh
Savings vs Deterministic: {dro.saving_pct}%
Wasserstein Radius (δ)  : {dro.wasserstein_delta}
Uncertainty Factor      : {dro.uncertainty_factor}
Data Scarcity Factor    : {dro.data_scarcity_factor}

3. ML MODEL PREDICTIONS
{'-'*55}
Daily Energy Usage      : {ml['energy_usage_kwh']} kWh
Recommended Battery     : {ml['battery_size_kwh']} kWh
Recommended PV Size     : {ml['pv_size_kwp']} kWp
Model Type              : {ml['model_info']['type']}
Model R² (Energy)       : {ml['model_metrics']['Energy Usage (kWh)']['r2']}
Model R² (Battery)      : {ml['model_metrics']['Battery Size (kWh)']['r2']}
Model R² (PV)           : {ml['model_metrics']['PV Size (kWp)']['r2']}

4. METHOD COMPARISON
{'-'*55}
{'Method':<22} {'Cost':>14} {'Reliability':>13} {'Energy Loss':>12}
{'-'*55}"""

        for m in comp["methods"]:
            report += f"\n{m['name']:<22} {fmt_inr(m['cost_inr']):>14} {m['reliability']:>12.1f}% {m['energy_loss']:>11.1f}%"

        report += f"""

5. DISCLAIMER
{'-'*55}
This report is based on simulation using synthetic data and
physics-based equations. Results are indicative only. Field
surveys, actual irradiance measurements, and engineering
assessments are required before procurement or installation.

{'='*65}
"""
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
