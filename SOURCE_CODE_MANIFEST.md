# Manifiesto del código fuente

## Núcleo de entrenamiento e IA

- `yakupredict_ai/config.py`: configuración, rutas, variables y parámetros experimentales.
- `yakupredict_ai/data_generator.py`: generación de escenarios sintéticos y etiqueta anticipatoria a 60 min.
- `yakupredict_ai/features.py`: ingeniería de características sin información futura.
- `yakupredict_ai/model.py`: Random Forest, Isolation Forest, calibración híbrida, métricas y serialización.
- `yakupredict_ai/train.py`: orquestación reproducible del entrenamiento y generación de reportes.

## Software / backend

- `yakupredict_ai/api.py`: API REST FastAPI y dashboard.
- `yakupredict_ai/schemas.py`: contratos Pydantic.
- `yakupredict_ai/service.py`: servicio de inferencia.
- `yakupredict_ai/risk.py`: estratificación y recomendación de riesgo.
- `yakupredict_ai/storage.py`: persistencia SQLite e historial.

## Frontend

- `templates/dashboard.html`
- `static/app.css`
- `static/app.js`

## Pruebas

- `tests/test_features.py`
- `tests/test_model.py`
- `tests/test_api.py`

## Reproducibilidad y validación

- `reports/training_report_v3.json`
- `reports/model_comparison_v3.csv`
- `external_validation/uci_hydraulic.py`
- `scripts/capture_screenshots.py`
- `.github/workflows/tests.yml`
