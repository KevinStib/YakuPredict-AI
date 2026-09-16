# YakuPredict AI 3.0.0

Prototipo académico de **análisis anticipatorio de condición y detección de anomalías** para apoyar el mantenimiento predictivo de unidades hidroeléctricas. La versión 3.0 corrige la validación experimental: la etiqueta ya no se define mediante umbrales de las variables observables, la partición se realiza por **escenarios completos**, los pesos de fusión se calibran en validación y se incorpora una prueba de **cambio de distribución**.

## Alcance

- Random Forest supervisado para estimar riesgo de degradación **en los siguientes 60 minutos** dentro del entorno sintético.
- Isolation Forest entrenado exclusivamente con condición normal.
- Fusión ponderada calibrada en escenarios de validación.
- Comparación RF / Isolation Forest / híbrido en validación, prueba y `stress_shift`.
- FastAPI, dashboard web, SQLite, historial y documentación OpenAPI.
- Pruebas automatizadas.
- Script de validación externa opcional con **UCI Condition Monitoring of Hydraulic Systems** (DOI 10.24432/C5CW21). El dataset no se redistribuye.

> Importante: el prototipo **no demuestra** desempeño predictivo en una central hidroeléctrica real. Los resultados principales pertenecen a escenarios sintéticos controlados; la validación industrial requiere datos SCADA y eventos de mantenimiento confirmados.

## Estructura

```text
YakuPredict_AI/
├── yakupredict_ai/
├── data/
├── models/
├── reports/
├── templates/
├── static/
├── screenshots/
├── tests/
├── scripts/
├── docs/
├── external_validation/
├── requirements.txt
└── README.md
```

## Reproducción

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
python -m yakupredict_ai.data_generator
python -m yakupredict_ai.train
pytest -q
uvicorn yakupredict_ai.api:app --reload
```

Dashboard: `http://127.0.0.1:8000/`  
OpenAPI: `http://127.0.0.1:8000/docs`  
Métricas: `http://127.0.0.1:8000/metrics`

## Validación externa pública opcional

Descargue el dataset UCI 447 desde su fuente oficial y extraiga los TXT. Después:

```bash
python external_validation/uci_hydraulic.py --data-dir /ruta/a/Sensors_Target
```

La tarea externa clasifica válvula no óptima frente a óptima usando resúmenes estadísticos de sensores. Sirve para estudiar portabilidad del pipeline sobre datos físicos independientes, **no** para afirmar validez hidroeléctrica.

## Versionado reproducible

Versión de software: **3.0.0**.  
El ZIP entregado con el TFE constituye el snapshot evaluado. Se incluye `reports/training_report_v3.json` y `reports/model_comparison_v3.csv` para verificar los números usados en la memoria.

## Publicación en GitHub

Repositorio académico asociado al TFE de Kevin Stib Cardenas Rosales.

La carpeta `.github/workflows/` ejecuta `pytest` automáticamente en cada `push` y `pull_request`.
