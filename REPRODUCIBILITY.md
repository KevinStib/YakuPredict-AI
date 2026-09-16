# Reproducibilidad de YakuPredict AI 3.0.0

## 1. Entorno recomendado

- Python 3.11 o 3.12
- Windows, Linux o macOS

## 2. Instalación

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Reproducir datos y entrenamiento

```bash
python -m yakupredict_ai.data_generator
python -m yakupredict_ai.train
```

Esto regenera:

- `data/yakupredict_synthetic_scenarios.csv`
- `models/yakupredict_bundle_v3.joblib`
- `reports/training_report_v3.json`
- `reports/model_comparison_v3.csv`

## 4. Ejecutar pruebas

```bash
pytest -q
```

La entrega validada incluye 3 pruebas automatizadas.

## 5. Ejecutar la aplicación

```bash
uvicorn yakupredict_ai.api:app --reload
```

- Dashboard: http://127.0.0.1:8000/
- OpenAPI: http://127.0.0.1:8000/docs
- Métricas: http://127.0.0.1:8000/metrics

## 6. Capturas reproducibles

Instale Playwright si desea regenerar las capturas:

```bash
pip install playwright
playwright install chromium
python scripts/capture_screenshots.py
```

## 7. Validación externa opcional

El dataset UCI no se redistribuye. Tras descargar y extraer *Condition Monitoring of Hydraulic Systems*:

```bash
python external_validation/uci_hydraulic.py --data-dir /ruta/a/Sensors_Target
```

Esta validación comprueba portabilidad del pipeline sobre datos físicos independientes; no constituye validación hidroeléctrica del prototipo.
