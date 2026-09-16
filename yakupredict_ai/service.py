from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .model import load_bundle
from .risk import classify_risk
from .schemas import Measurement, PredictionResponse
from .storage import save_prediction


@lru_cache(maxsize=1)
def get_model():
    return load_bundle()


def measurement_to_frame(measurement: Measurement) -> pd.DataFrame:
    data = measurement.model_dump()
    defaults = {
        "prev_vibration_mms": data["vibration_mms"],
        "prev_bearing_temp_c": data["bearing_temp_c"],
        "prev_oil_temp_c": data["oil_temp_c"],
        "prev_hydraulic_pressure_bar": data["hydraulic_pressure_bar"],
        "prev_power_mw": data["power_mw"],
    }
    for key, value in defaults.items():
        if data[key] is None:
            data[key] = value
    return pd.DataFrame([data])


def evaluate_measurement(measurement: Measurement) -> PredictionResponse:
    scores = get_model().predict_scores(measurement_to_frame(measurement)).iloc[0]
    risk = float(scores["risk_score"])
    p_supervised = float(scores["supervised_probability"])
    p_anomaly = float(scores["anomaly_probability"])
    prediction = int(scores["prediction"])
    decision = classify_risk(risk)
    record_id, created_at = save_prediction(
        payload=measurement.model_dump(),
        risk_score=risk,
        level=decision.level,
        prediction=prediction,
        supervised_probability=p_supervised,
        anomaly_probability=p_anomaly,
        recommendation=decision.recommendation,
    )
    return PredictionResponse(
        id=record_id,
        created_at=created_at,
        risk_score=risk,
        risk_percent=round(risk * 100, 2),
        level=decision.level,
        priority=decision.priority,
        prediction=prediction,
        supervised_probability=p_supervised,
        anomaly_probability=p_anomaly,
        recommendation=decision.recommendation,
    )
