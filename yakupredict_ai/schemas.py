from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Measurement(BaseModel):
    flow_m3s: float = Field(..., gt=0, le=250, description="Caudal de agua [m³/s]")
    power_mw: float = Field(..., ge=0, le=250, description="Potencia activa [MW]")
    vibration_mms: float = Field(..., ge=0, le=30, description="Vibración global [mm/s]")
    bearing_temp_c: float = Field(..., ge=-20, le=180, description="Temperatura de cojinete [°C]")
    oil_temp_c: float = Field(..., ge=-20, le=160, description="Temperatura de aceite [°C]")
    hydraulic_pressure_bar: float = Field(..., ge=0, le=250, description="Presión hidráulica [bar]")
    wicket_gate_pct: float = Field(..., ge=0, le=100, description="Apertura del distribuidor [%]")
    stator_current_a: float = Field(..., ge=0, le=5000, description="Corriente de estator [A]")
    ambient_temp_c: float = Field(..., ge=-40, le=80, description="Temperatura ambiente [°C]")
    prev_vibration_mms: Optional[float] = None
    prev_bearing_temp_c: Optional[float] = None
    prev_oil_temp_c: Optional[float] = None
    prev_hydraulic_pressure_bar: Optional[float] = None
    prev_power_mw: Optional[float] = None


class PredictionResponse(BaseModel):
    id: int
    created_at: str
    risk_score: float
    risk_percent: float
    level: str
    priority: int
    prediction: int
    supervised_probability: float
    anomaly_probability: float
    recommendation: str


class HealthResponse(BaseModel):
    status: str
    software: str
    version: str
    model: str
