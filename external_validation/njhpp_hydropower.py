"""Validacion externa real con SCADA de una central hidroelectrica (NJHPP).

Fuente primaria
---------------
Y. S. Afridi, "Bearing Vibration Dataset of a Hydropower Project", figshare,
2022. DOI: 10.6084/m9.figshare.21290895 (CC BY 4.0).

El conjunto contiene doce meses de vibracion horizontal del cojinete guia de
turbina de la Unidad 01 de una central hidroelectrica de 969 MW en Pakistan y
contiene periodos normales y asociados a un fallo real del cojinete.

Esta rutina NO inventa etiquetas de fallo por muestra. Evalua una tarea de
pronostico verificable: predecir el siguiente valor de vibracion a partir de los
cinco valores anteriores, de forma coherente con la formulacion temporal usada
en la literatura asociada al dataset. La division es cronologica para evitar
fuga de informacion.

Uso:
    python external_validation/njhpp_hydropower.py --download

Tambien puede usarse un archivo descargado manualmente:
    python external_validation/njhpp_hydropower.py --input /ruta/al/archivo
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

FIGSHARE_ARTICLE_ID = 21290895
FIGSHARE_DOI = "10.6084/m9.figshare.21290895"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
DEFAULT_OUTPUT = Path("reports/njhpp_real_validation.json")
DEFAULT_PREDICTIONS = Path("reports/njhpp_real_predictions.csv")
DEFAULT_CACHE = Path("data/external/njhpp")


@dataclass
class SeriesCandidate:
    source_file: str
    column: str
    n: int
    score: float
    values: pd.Series
    timestamp: pd.Series | None = None


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "YakuPredict-AI/4.0 academic-validation"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "YakuPredict-AI/4.0 academic-validation"})
    with urllib.request.urlopen(req, timeout=180) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination


def download_figshare(cache_dir: Path = DEFAULT_CACHE) -> list[Path]:
    """Descarga los archivos declarados por la API publica de figshare."""
    metadata = _http_json(FIGSHARE_API)
    files = metadata.get("files") or []
    if not files:
        raise RuntimeError("La API de figshare no devolvio archivos para el articulo.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for item in files:
        name = item.get("name") or f"figshare_{item.get('id', 'file')}"
        url = item.get("download_url")
        if not url:
            continue
        path = cache_dir / name
        if not path.exists() or path.stat().st_size == 0:
            _download_file(url, path)
        downloaded.append(path)
    if not downloaded:
        raise RuntimeError("No fue posible descargar ningun archivo de figshare.")
    return downloaded


def _expand_paths(paths: Iterable[Path], workdir: Path) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".zip":
            target = workdir / path.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
            expanded.extend(p for p in target.rglob("*") if p.is_file())
        else:
            expanded.append(path)
    return expanded


def _read_table(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix in {".csv", ".txt", ".dat", ".tsv"}:
            attempts = [
                {"sep": None, "engine": "python"},
                {"sep": ","},
                {"sep": ";"},
                {"sep": "\t"},
                {"sep": r"\s+", "engine": "python"},
            ]
            for kwargs in attempts:
                try:
                    frame = pd.read_csv(path, **kwargs)
                    if frame.shape[0] > 5 and frame.shape[1] >= 1:
                        return frame
                except Exception:
                    pass
            return None
    except Exception:
        return None
    return None


def _column_score(name: str, series: pd.Series) -> float:
    lname = str(name).lower()
    score = math.log10(max(len(series), 10))
    keywords = {
        "vib": 8.0,
        "vibration": 8.0,
        "runout": 7.0,
        "guide": 3.0,
        "bearing": 3.0,
        "horizontal": 2.0,
        "turbine": 1.5,
        "tgb": 3.0,
    }
    for token, weight in keywords.items():
        if token in lname:
            score += weight
    if any(token in lname for token in ["index", "id", "serial", "row"]):
        score -= 5.0
    if series.nunique(dropna=True) < 10:
        score -= 8.0
    return score


def _timestamp_candidate(df: pd.DataFrame) -> pd.Series | None:
    for col in df.columns:
        lname = str(col).lower()
        if any(k in lname for k in ["time", "date", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                return parsed
    return None


def discover_vibration_series(paths: Iterable[Path]) -> SeriesCandidate:
    candidates: list[SeriesCandidate] = []
    for path in paths:
        frame = _read_table(path)
        if frame is None or frame.empty:
            continue
        ts = _timestamp_candidate(frame)
        for col in frame.columns:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            valid_fraction = numeric.notna().mean()
            if valid_fraction < 0.75:
                continue
            clean = numeric.dropna().reset_index(drop=True)
            if len(clean) < 100:
                continue
            candidates.append(
                SeriesCandidate(
                    source_file=path.name,
                    column=str(col),
                    n=int(len(clean)),
                    score=_column_score(str(col), clean),
                    values=clean.astype(float),
                    timestamp=(ts.dropna().reset_index(drop=True) if ts is not None and len(ts.dropna()) == len(clean) else None),
                )
            )
    if not candidates:
        raise ValueError("No se encontro una serie numerica de vibracion suficientemente larga.")
    candidates.sort(key=lambda c: (c.score, c.n), reverse=True)
    return candidates[0]


def _estimate_sampling(timestamp: pd.Series | None) -> dict:
    if timestamp is None or len(timestamp) < 3:
        return {"available": False, "median_seconds": None, "label": "no disponible"}
    delta = pd.Series(timestamp).sort_values().diff().dt.total_seconds().dropna()
    delta = delta[(delta > 0) & np.isfinite(delta)]
    if delta.empty:
        return {"available": False, "median_seconds": None, "label": "no disponible"}
    sec = float(delta.median())
    if sec >= 3600:
        label = f"{sec / 3600:.2f} h"
    elif sec >= 60:
        label = f"{sec / 60:.2f} min"
    else:
        label = f"{sec:.2f} s"
    return {"available": True, "median_seconds": sec, "label": label}


def _supervised_frame(series: pd.Series, lags: int = 5, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    s = pd.Series(series, dtype=float).reset_index(drop=True)
    data: dict[str, pd.Series] = {}
    for lag in range(lags, 0, -1):
        data[f"lag_{lag}"] = s.shift(lag)
    x = pd.DataFrame(data)
    x["rolling_mean_3"] = s.shift(1).rolling(3).mean()
    x["rolling_std_3"] = s.shift(1).rolling(3).std()
    x["rolling_mean_5"] = s.shift(1).rolling(5).mean()
    x["delta_1"] = s.shift(1) - s.shift(2)
    x["slope_5"] = (s.shift(1) - s.shift(5)) / 4.0
    y = s.shift(-horizon)
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def evaluate_series(candidate: SeriesCandidate, output_predictions: Path = DEFAULT_PREDICTIONS) -> dict:
    raw = candidate.values.astype(float).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    x, y = _supervised_frame(raw, lags=5, horizon=1)
    n = len(x)
    if n < 300:
        raise ValueError(f"Serie insuficiente para validacion cronologica: {n} muestras supervisadas.")

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    x_train, x_val, x_test = x.iloc[:train_end], x.iloc[train_end:val_end], x.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    pred_val = rf.predict(x_val)
    pred_test = rf.predict(x_test)

    persistence_val = x_val["lag_1"].to_numpy()
    persistence_test = x_test["lag_1"].to_numpy()

    scaler = RobustScaler().fit(x_train)
    iforest = IsolationForest(n_estimators=400, contamination="auto", random_state=42, n_jobs=-1)
    iforest.fit(scaler.transform(x_train))
    anomaly_test = -iforest.score_samples(scaler.transform(x_test))

    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "rf_forecast": pred_test,
        "persistence": persistence_test,
        "absolute_error_rf": np.abs(y_test.to_numpy() - pred_test),
        "anomaly_score": anomaly_test,
    })
    pred_df.to_csv(output_predictions, index=False)

    sampling = _estimate_sampling(candidate.timestamp)
    report = {
        "dataset": "Bearing Vibration Dataset of a Hydropower Project",
        "provider": "figshare",
        "author": "Yasir Saleem Afridi",
        "doi": FIGSHARE_DOI,
        "license": "CC BY 4.0",
        "domain": "real industrial hydropower SCADA",
        "plant_context": "969 MW hydropower project in Pakistan, Unit 01 Turbine Guide Bearing",
        "source_file": candidate.source_file,
        "selected_column": candidate.column,
        "raw_samples": int(len(raw)),
        "supervised_samples": int(n),
        "split": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
            "strategy": "chronological 70/15/15 without shuffling",
        },
        "task": "one-step-ahead vibration forecasting from five previous observations",
        "sampling": sampling,
        "random_forest_validation": _regression_metrics(y_val.to_numpy(), pred_val),
        "random_forest_test": _regression_metrics(y_test.to_numpy(), pred_test),
        "persistence_validation": _regression_metrics(y_val.to_numpy(), persistence_val),
        "persistence_test": _regression_metrics(y_test.to_numpy(), persistence_test),
        "relative_improvement_rmse_test_pct": float(
            100.0 * (mean_squared_error(y_test, persistence_test) ** 0.5 - mean_squared_error(y_test, pred_test) ** 0.5)
            / max(mean_squared_error(y_test, persistence_test) ** 0.5, 1e-12)
        ),
        "anomaly_score_test": {
            "mean": float(np.mean(anomaly_test)),
            "std": float(np.std(anomaly_test)),
            "p95": float(np.quantile(anomaly_test, 0.95)),
            "max": float(np.max(anomaly_test)),
        },
        "methodological_scope": (
            "External validation on real industrial hydropower SCADA data. It validates vibration "
            "forecasting and condition-monitoring portability, not per-sample fault classification, "
            "because the public metadata does not provide a confirmed fault label for every timestamp."
        ),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="Archivo o carpeta local del dataset NJHPP.")
    ap.add_argument("--download", action="store_true", help="Descargar desde la API publica de figshare.")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = ap.parse_args()

    if not args.input and not args.download:
        ap.error("Use --download o --input.")

    if args.download:
        paths = download_figshare(args.cache_dir)
    else:
        if args.input.is_dir():
            paths = [p for p in args.input.rglob("*") if p.is_file()]
        else:
            paths = [args.input]

    with tempfile.TemporaryDirectory(prefix="yakupredict_njhpp_") as tmp:
        files = _expand_paths(paths, Path(tmp))
        candidate = discover_vibration_series(files)
        report = evaluate_series(candidate, args.predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
