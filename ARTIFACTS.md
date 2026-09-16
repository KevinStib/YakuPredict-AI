# Artefactos generados

Los artefactos binarios y el dataset completo se generan de forma reproducible y no son necesarios para revisar el código fuente.

- `data/yakupredict_synthetic_scenarios.csv`: `python -m yakupredict_ai.data_generator`
- `models/yakupredict_bundle_v3.joblib`: `python -m yakupredict_ai.train`
- `screenshots/*.png`: `python scripts/capture_screenshots.py` después de iniciar la API.

Los resultados numéricos originales se conservan en `reports/`. El snapshot local entregado con el TFE incluye además el dataset generado, el bundle serializado y las capturas del dashboard.
