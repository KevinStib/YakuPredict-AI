from pathlib import Path

from fastapi.testclient import TestClient

from yakupredict_ai import api
from yakupredict_ai.config import DB_PATH, MODEL_PATH
from yakupredict_ai.service import get_model


def _ensure_model():
    assert MODEL_PATH.exists(), "El modelo v3 debe generarse antes de las pruebas de API."
    get_model.cache_clear()


def test_health_and_predict():
    _ensure_model()
    if DB_PATH.exists():
        DB_PATH.unlink()
    client = TestClient(api.app)
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['software'] == 'YakuPredict AI'
    response = client.post('/predict', json={
        'flow_m3s':78,'power_mw':72,'vibration_mms':2.2,'bearing_temp_c':66,
        'oil_temp_c':52,'hydraulic_pressure_bar':74,'wicket_gate_pct':75,
        'stator_current_a':1360,'ambient_temp_c':22
    })
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload['risk_score'] <= 1
    assert payload['level'] in {'BAJO','MEDIO','ALTO','CRÍTICO'}
