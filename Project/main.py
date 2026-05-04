"""
=============================================================================
PV-BESS Microgrid Optimizer — FastAPI Backend
Based on: Sharma et al. Hybrid DRO + ML Framework
=============================================================================
Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
API docs:  http://localhost:8000/docs
=============================================================================
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from api.optimize   import router as optimize_router
from api.predict    import router as predict_router
from api.solar      import router as solar_router
from api.simulate   import router as simulate_router
from api.health     import router as health_router

# ── App init ──────────────────────────────────────────────────
app = FastAPI(
    title       = "PV-BESS Microgrid Optimizer API",
    description = "Hybrid DRO + ML Framework for optimal PV-BESS sizing · Sharma et al.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow frontend to call the API ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],      # restrict to your domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Mount static files (the HTML frontend) ────────────────────
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Include routers ───────────────────────────────────────────
app.include_router(health_router,   prefix="/api",       tags=["Health"])
app.include_router(optimize_router, prefix="/api",       tags=["DRO Optimization"])
app.include_router(predict_router,  prefix="/api",       tags=["ML Prediction"])
app.include_router(solar_router,    prefix="/api",       tags=["Solar Data"])
app.include_router(simulate_router, prefix="/api",       tags=["Simulation"])

# ── Serve frontend ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML frontend."""
    html_path = "static/index.html"
    if os.path.exists(html_path):
        with open(html_path, encoding='utf-8') as f:
            return f.read()
    return HTMLResponse("<h2>Frontend not found. Place pvbess_optimizer.html in /static/index.html</h2>")

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse({"error": "Endpoint not found", "path": str(request.url)}, status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse({"error": "Internal server error", "detail": str(exc)}, status_code=500)
