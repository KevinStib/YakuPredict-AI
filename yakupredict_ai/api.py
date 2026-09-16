from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__
from .config import METRICS_PATH, STATIC_DIR, TEMPLATE_DIR
from .schemas import HealthResponse, Measurement, PredictionResponse
from .service import evaluate_measurement, get_model
from .storage import recent_predictions

app = FastAPI(
    title="YakuPredict AI",
    version=__version__,
    description=(
        "Sistema académico de apoyo al mantenimiento predictivo en centrales hidroeléctricas "
        "mediante aprendizaje supervisado y detección de anomalías."
    ),
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        return HealthResponse(status="ok", software="YakuPredict AI", version=__version__, model="loaded")
    except FileNotFoundError as exc:
        return HealthResponse(status="degraded", software="YakuPredict AI", version=__version__, model=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(measurement: Measurement) -> PredictionResponse:
    try:
        return evaluate_measurement(measurement)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    return recent_predictions(limit)


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No existe el reporte de entrenamiento. Ejecute el entrenamiento.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"software_name": "YakuPredict AI", "version": __version__},
    )
