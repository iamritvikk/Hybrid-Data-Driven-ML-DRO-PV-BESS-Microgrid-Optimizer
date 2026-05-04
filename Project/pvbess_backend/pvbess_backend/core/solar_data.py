"""
=============================================================================
core/solar_data.py
Solar irradiance data fetcher — NASA POWER API + local cache
NASA POWER provides free daily solar irradiance for any location on Earth
API docs: https://power.larc.nasa.gov/docs/
=============================================================================
"""

import httpx
import json
import os
from typing import Optional, Dict
from datetime import datetime, timedelta

CACHE_DIR  = "models/solar_cache"
NASA_URL   = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# Fallback database (when NASA API is unavailable)
FALLBACK_DB = {
    "New Delhi":      {"irr": 5.5, "lat": 28.6, "lon": 77.2},
    "Mumbai":         {"irr": 5.2, "lat": 19.1, "lon": 72.9},
    "Chennai":        {"irr": 5.8, "lat": 13.1, "lon": 80.3},
    "Jaipur":         {"irr": 6.0, "lat": 26.9, "lon": 75.8},
    "Jodhpur":        {"irr": 6.4, "lat": 26.3, "lon": 73.0},
    "Bangalore":      {"irr": 5.4, "lat": 12.9, "lon": 77.6},
    "Hyderabad":      {"irr": 5.6, "lat": 17.4, "lon": 78.5},
    "Kolkata":        {"irr": 4.8, "lat": 22.6, "lon": 88.4},
    "Leh":            {"irr": 6.7, "lat": 34.2, "lon": 77.6},
    "Ahmedabad":      {"irr": 6.1, "lat": 23.0, "lon": 72.6},
    "Riyadh":         {"irr": 7.2, "lat": 24.7, "lon": 46.7},
    "Dubai":          {"irr": 6.8, "lat": 25.2, "lon": 55.3},
    "Cairo":          {"irr": 7.0, "lat": 30.1, "lon": 31.2},
    "London":         {"irr": 2.8, "lat": 51.5, "lon": -0.1},
    "Berlin":         {"irr": 3.0, "lat": 52.5, "lon": 13.4},
    "Madrid":         {"irr": 5.1, "lat": 40.4, "lon": -3.7},
    "Phoenix":        {"irr": 7.5, "lat": 33.4, "lon": -112.1},
    "Los Angeles":    {"irr": 5.6, "lat": 34.1, "lon": -118.2},
    "Sydney":         {"irr": 4.8, "lat": -33.9, "lon": 151.2},
    "Perth":          {"irr": 6.2, "lat": -31.9, "lon": 115.9},
    "Darwin":         {"irr": 6.5, "lat": -12.5, "lon": 130.8},
    "Bangkok":        {"irr": 5.0, "lat": 13.8, "lon": 100.5},
    "Nairobi":        {"irr": 5.8, "lat": -1.3, "lon": 36.8},
    "Lagos":          {"irr": 5.2, "lat": 6.5,  "lon": 3.4},
    "Karachi":        {"irr": 6.0, "lat": 24.9, "lon": 67.0},
    "Khartoum":       {"irr": 7.3, "lat": 15.6, "lon": 32.5},
    "Male":           {"irr": 5.5, "lat": 4.2,  "lon": 73.5},
    "Honolulu":       {"irr": 5.7, "lat": 21.3, "lon": -157.8},
}

os.makedirs(CACHE_DIR, exist_ok=True)


async def fetch_nasa_irradiance(lat: float, lon: float) -> Optional[float]:
    """
    Fetch annual mean daily solar irradiance from NASA POWER API.
    Returns kWh/m²/day or None if unavailable.
    """
    cache_key = f"{lat:.2f}_{lon:.2f}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Check cache first (valid for 30 days)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        age_days = (datetime.now() - datetime.fromisoformat(cached["fetched_at"])).days
        if age_days < 30:
            return cached["irr_annual_mean"]

    # Call NASA POWER API
    try:
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community":  "RE",
            "longitude":  lon,
            "latitude":   lat,
            "format":     "JSON",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(NASA_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        monthly = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        # Key "ANN" = annual mean in NASA POWER
        irr = float(monthly.get("ANN", 0))
        if irr <= 0:
            return None

        # Cache the result
        with open(cache_file, "w") as f:
            json.dump({"irr_annual_mean": round(irr, 3),
                       "lat": lat, "lon": lon,
                       "fetched_at": datetime.now().isoformat()}, f)
        return round(irr, 3)

    except Exception as e:
        print(f"NASA API error for ({lat},{lon}): {e}")
        return None


def get_city_irradiance(city: str) -> Optional[Dict]:
    """Look up irradiance for a known city from fallback database."""
    city_lower = city.lower().strip()
    for name, data in FALLBACK_DB.items():
        if city_lower in name.lower():
            return {"city": name, "irr": data["irr"],
                    "lat": data["lat"], "lon": data["lon"],
                    "source": "local_database"}
    return None


def get_all_cities() -> list:
    """Return the full city database."""
    return [
        {"city": name, "irr": d["irr"], "lat": d["lat"], "lon": d["lon"]}
        for name, d in FALLBACK_DB.items()
    ]
