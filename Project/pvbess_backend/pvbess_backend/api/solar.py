"""
api/solar.py — Solar irradiance data endpoints
GET  /api/solar/cities            →  all cities with irradiance
GET  /api/solar/city?name=...     →  lookup by city name
POST /api/solar/location          →  fetch from NASA POWER by lat/lon
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from core.solar_data import (fetch_nasa_irradiance, get_city_irradiance,
                              get_all_cities)

router = APIRouter()


class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90,  le=90,  description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


@router.get("/solar/cities")
async def all_cities():
    """Return all cities with known solar irradiance data."""
    return {"status": "success", "cities": get_all_cities(), "count": len(get_all_cities())}


@router.get("/solar/city")
async def city_lookup(name: str = Query(..., description="City name")):
    """Look up solar irradiance for a named city."""
    result = get_city_irradiance(name)
    if not result:
        raise HTTPException(status_code=404, detail=f"City '{name}' not found in database")
    return {"status": "success", **result}


@router.post("/solar/location")
async def location_irradiance(req: LocationRequest):
    """
    Fetch real solar irradiance from NASA POWER API for any lat/lon.
    Falls back to local estimate if NASA is unavailable.
    """
    irr = await fetch_nasa_irradiance(req.lat, req.lon)
    if irr is None:
        # Rough estimate from latitude if NASA unavailable
        lat_abs = abs(req.lat)
        if lat_abs < 15:   irr = 5.8
        elif lat_abs < 30: irr = 6.2
        elif lat_abs < 45: irr = 4.8
        elif lat_abs < 60: irr = 3.2
        else:              irr = 2.5
        source = "latitude_estimate"
    else:
        source = "nasa_power_api"

    return {
        "status":  "success",
        "lat":     req.lat,
        "lon":     req.lon,
        "irr_annual_mean": irr,
        "unit":    "kWh/m²/day",
        "source":  source,
    }
