# YakuPredict AI 4.0.0 - Real SCADA Validation

YakuPredict AI es un prototipo académico para **análisis anticipatorio de condición, detección de anomalías y apoyo al mantenimiento predictivo hidroeléctrico**. El proyecto combina una validación sintética reproducible con una **validación externa sobre datos SCADA industriales reales** de una central hidroeléctrica.

## Aporte principal

- Random Forest supervisado para riesgo anticipatorio en escenarios sintéticos.
- Isolation Forest para detección de anomalías.
- Fusión híbrida calibrada únicamente sobre escenarios de validación.
- FastAPI, dashboard web, SQLite y trazabilidad de inferencias.
- Pruebas automatizadas.
- Validación externa con datos reales del **Neelum-Jhelum Hydropower Project (Pakistan, 969 MW)**.

## Dataset real

Fuente pública: Yasir Saleem Afridi (2022), *Bearing Vibration Dataset of a Hydropower Project*, figshare.  
DOI: **10.6084/m9.figshare.21290895**  
Licencia: **CC BY 4.0**.

La validación v4 usa seis archivos mensuales G1 (junio-noviembre). Se pronostica a un paso la vibración horizontal TGB +X usando 19 características derivadas de variables SCADA reales. La partición es por bloques temporales completos:

- entrenamiento: junio-septiembre;
- validación: octubre;
- prueba: noviembre.

En prueba real de noviembre, Extra Trees obtuvo RMSE ≈ **0.7332**, frente a **0.8213** del predictor de persistencia, una mejora relativa de aproximadamente **10.73 %**. En octubre, en cambio, la persistencia fue claramente superior, evidenciando **no estacionariedad / cambio de régimen**. Por ello el trabajo no afirma superioridad universal ni una tasa de predicción de fallos por timestamp.

## Reproducción

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt

# Pipeline sintético
python -m yakupredict_ai.data_generator
python -m yakupredict_ai.train
pytest -q

# Descargar datos públicos NJHPP
python external_validation/njhpp_hydropower.py --download

# Validación real NJHPP
python external_validation/njhpp_hydropower_v4.py \
  --data-dir data/external/njhpp

# Aplicación
uvicorn yakupredict_ai.api:app --reload
```

Dashboard: `http://127.0.0.1:8000/`  
OpenAPI: `http://127.0.0.1:8000/docs`

## Estructura

```text
YakuPredict_AI_TFE_v4_RealSCADA/
├── yakupredict_ai/               # IA, API, persistencia y servicios
├── external_validation/          # validación UCI y NJHPP real
├── data/
│   ├── yakupredict_synthetic_scenarios.csv
│   └── external/njhpp/           # descarga reproducible CC BY 4.0
├── models/                       # bundles serializados
├── reports/                      # métricas y predicciones
├── templates/                    # dashboard HTML
├── static/                       # CSS/JS
├── screenshots/
├── tests/
├── scripts/
└── docs/
```

## Alcance científico

La evidencia sintética valida el pipeline completo bajo condiciones controladas. La evidencia real valida **portabilidad a datos SCADA hidroeléctricos reales, pronóstico temporal de vibración y detección de anomalías**. El conjunto público no aporta una etiqueta confirmada de fallo para cada instante, por lo que no se reporta una exactitud de clasificación de fallos industriales por timestamp.

## Autor

Kevin Stib Cardenas Rosales - Trabajo Fin de Estudios, Máster Universitario en Inteligencia Artificial, UNIR, 2026.
