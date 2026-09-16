from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_FEATURES, RAW_FEATURES


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [name for name in RAW_FEATURES if name not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera características causales: no utiliza etiqueta ni información futura."""
    validate_raw_columns(df)
    frame = df.copy()
    # Se respetan fronteras de escenario para evitar usar el último dato de otro escenario.
    groups = frame.groupby("scenario_id", sort=False) if "scenario_id" in frame.columns else [(None, frame)]
    result_parts = []
    for _, part in groups:
        p = part.copy()
        p["prev_vibration_mms"] = p["vibration_mms"].shift(1).fillna(p["vibration_mms"])
        p["prev_bearing_temp_c"] = p["bearing_temp_c"].shift(1).fillna(p["bearing_temp_c"])
        p["prev_oil_temp_c"] = p["oil_temp_c"].shift(1).fillna(p["oil_temp_c"])
        p["prev_hydraulic_pressure_bar"] = p["hydraulic_pressure_bar"].shift(1).fillna(p["hydraulic_pressure_bar"])
        p["prev_power_mw"] = p["power_mw"].shift(1).fillna(p["power_mw"])
        eps = 1e-6
        p["power_per_flow"] = p["power_mw"] / (p["flow_m3s"].abs() + eps)
        p["thermal_delta_c"] = p["bearing_temp_c"] - p["oil_temp_c"]
        p["vibration_delta"] = p["vibration_mms"] - p["prev_vibration_mms"]
        p["bearing_temp_delta"] = p["bearing_temp_c"] - p["prev_bearing_temp_c"]
        p["oil_temp_delta"] = p["oil_temp_c"] - p["prev_oil_temp_c"]
        p["pressure_delta"] = p["hydraulic_pressure_bar"] - p["prev_hydraulic_pressure_bar"]
        p["power_delta"] = p["power_mw"] - p["prev_power_mw"]
        p["vibration_thermal_interaction"] = p["vibration_mms"] * p["thermal_delta_c"]
        p["rolling_vibration_mean_3"] = p["vibration_mms"].rolling(3, min_periods=1).mean()
        p["rolling_bearing_temp_mean_3"] = p["bearing_temp_c"].rolling(3, min_periods=1).mean()
        result_parts.append(p)
    out = pd.concat(result_parts).sort_index()
    out[MODEL_FEATURES] = out[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out
