# PV-BESS Microgrid Optimizer — FastAPI Backend

**Hybrid DRO + ML Framework** | Sharma et al.

---

## Project Structure

```
pvbess_backend/
├── main.py                  ← FastAPI app entry point
├── requirements.txt         ← pip dependencies
├── .env.example             ← environment config template
├── vercel.json              ← Vercel deployment config
│
├── api/                     ← API route handlers
│   ├── optimize.py          ← POST /api/optimize   (DRO solver)
│   ├── predict.py           ← POST /api/predict    (ML model)
│   ├── simulate.py          ← POST /api/simulate   (combined)
│   ├── solar.py             ← GET  /api/solar      (NASA data)
│   └── health.py            ← GET  /api/health
│
├── core/                    ← Business logic
│   ├── dro_engine.py        ← DRO optimization engine
│   ├── ml_model.py          ← Random Forest trainer + predictor
│   └── solar_data.py        ← NASA POWER API client
│
├── models/                  ← Saved ML models (auto-created)
│   ├── rf_model.joblib      ← trained Random Forest
│   ├── rf_scaler.joblib     ← StandardScaler
│   └── rf_metrics.joblib    ← model performance metrics
│
└── static/                  ← Frontend (put your HTML here)
    └── index.html           ← pvbess_optimizer.html goes here
```

---

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open in browser
```
http://localhost:8000        → Frontend
http://localhost:8000/docs   → Interactive API docs (Swagger UI)
http://localhost:8000/redoc  → ReDoc documentation
```

### 5. Train the ML model (first time only)
```bash
# The model auto-trains on first request, or train manually:
curl -X POST http://localhost:8000/api/predict/train
```

---

## API Endpoints

### POST /api/optimize — DRO Optimization
```bash
curl -X POST http://localhost:8000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "solar_irr": 5.5,
    "load_demand": 60,
    "load_var": 20,
    "solar_var": 25,
    "pv_cost": 0.8,
    "bat_cost": 300,
    "target_rel": 92,
    "delta": 0.5,
    "data_months": 6,
    "lifetime": 20
  }'
```

**Response:**
```json
{
  "status": "success",
  "pv_cap_kwp": 18.4,
  "bat_cap_kwh": 24.6,
  "total_inr": 3840000,
  "reliability_pct": 93.5,
  "energy_loss_pct": 8.3,
  "saving_pct": 15.3,
  "lcoe_inr": 5.44,
  "tradeoff_curve": [...],
  "method_comparison": {...}
}
```

---

### POST /api/predict — ML Prediction
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "solar_irr": 5.5,
    "load_demand": 60,
    "load_var": 20,
    "solar_var": 25
  }'
```

**Response:**
```json
{
  "status": "success",
  "energy_usage_kwh": 72.4,
  "battery_size_kwh": 26.1,
  "pv_size_kwp": 19.2,
  "model_metrics": {
    "Energy Usage (kWh)": {"r2": 0.993, "rmse": 4.8},
    "Battery Size (kWh)": {"r2": 0.989, "rmse": 3.1},
    "PV Size (kWp)":      {"r2": 0.978, "rmse": 3.9}
  }
}
```

---

### POST /api/simulate — Full Combined Run
Runs both DRO and ML together and returns combined results.

### POST /api/simulate/report — Text Report
Returns a plain-text technical report as a downloadable file.

### GET /api/solar/cities — City Solar Data
Returns all 27 cities with irradiance from local database.

### POST /api/solar/location — NASA POWER Lookup
Fetch real irradiance from NASA for any latitude/longitude.

### GET /api/health — Health Check
Returns API status and model availability.

---

## Connect the HTML Frontend

Replace the JS math in `pvbess_optimizer.html` with real API calls:

```javascript
// In pvbess_optimizer.html — replace runOpt() body with:
async function runOpt() {
  const response = await fetch('http://localhost:8000/api/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      solar_irr:   gv('solar'),
      load_demand: gv('load'),
      load_var:    gv('loadvar'),
      solar_var:   gv('solarvar'),
      pv_cost:     gv('pvcost'),
      bat_cost:    gv('batcost'),
      target_rel:  gv('reliability'),
      delta:       gv('delta'),
      data_months: gv('datamonths'),
      lifetime:    gv('lifetime'),
    })
  });
  const data = await response.json();

  // Update UI with real backend results
  document.getElementById('r-pv').textContent   = data.pv_cap_kwp + ' kWp';
  document.getElementById('r-bat').textContent  = data.bat_cap_kwh + ' kWh';
  document.getElementById('r-cost').textContent = fmtINR(data.total_inr);
  document.getElementById('r-rel').textContent  = data.reliability_pct + '%';
  document.getElementById('r-loss').textContent = data.energy_loss_pct + '%';
  document.getElementById('r-save').textContent = data.saving_pct + '% less';
}
```

---

## Deployment

### Option 1 — Render (Free, Recommended)
1. Push this folder to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Set: `Build command: pip install -r requirements.txt`
5. Set: `Start command: uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Deploy

### Option 2 — Railway
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option 3 — Vercel
```bash
npm install -g vercel
vercel
```

### Option 4 — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t pvbess-api .
docker run -p 8000:8000 pvbess-api
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI |
| ML model | scikit-learn RandomForest |
| Data validation | Pydantic v2 |
| Server | Uvicorn (ASGI) |
| Solar data | NASA POWER API |
| Model storage | joblib |
| Deployment | Render / Railway / Vercel |

---

## Based on
Sharma R., Chaudhary N., Shekhar N., Kumar N., Dixit A.
*"A Hybrid Data Driven and Distributionally Robust Optimization Framework
for Optimal Sizing of PV-BESS Microgrids in uncertain and Data-Scarce Environment"*
AKGEC, Ghaziabad, 2025.
