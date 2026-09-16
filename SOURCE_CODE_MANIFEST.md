# Source Code Manifest - v4.0.0-real-scada

- `yakupredict_ai/`: núcleo de IA, entrenamiento, API, riesgo, almacenamiento y esquemas.
- `external_validation/njhpp_hydropower.py`: descarga reproducible del dataset NJHPP desde figshare.
- `external_validation/njhpp_hydropower_v4.py`: experimento multimensual real bloqueado por meses.
- `external_validation/uci_hydraulic.py`: adaptador complementario UCI.
- `templates/`, `static/`: dashboard.
- `tests/`: pruebas automatizadas.
- `scripts/`: capturas reproducibles.
- `reports/`: resultados sintéticos y reales.
- Los datos reales no necesitan versionarse: el workflow los descarga de figshare bajo CC BY 4.0.
