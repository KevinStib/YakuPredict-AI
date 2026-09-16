# Reproducibilidad

## Entorno

Python 3.11 o 3.12.

```bash
pip install -r requirements.txt
```

## Experimento sintético

```bash
python -m yakupredict_ai.data_generator
python -m yakupredict_ai.train
pytest -q
```

## Experimento real NJHPP

```bash
python external_validation/njhpp_hydropower.py --download
python external_validation/njhpp_hydropower_v4.py --data-dir data/external/njhpp
```

Salidas:

- `reports/njhpp_real_validation_v4.json`
- `reports/njhpp_real_predictions_v4.csv`
- `reports/njhpp_real_metrics_v4.csv`

El protocolo evita mezcla aleatoria entre meses: junio-septiembre entrenan, octubre valida y noviembre prueba. Esta decisión reduce fuga temporal y permite observar explícitamente cambios de régimen.
