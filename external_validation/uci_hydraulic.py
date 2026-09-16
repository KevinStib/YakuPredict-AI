"""Validación externa opcional con UCI Condition Monitoring of Hydraulic Systems.

El dataset no se redistribuye en este repositorio. Descarga oficial:
https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
DOI: 10.24432/C5CW21

Uso tras extraer los TXT en una carpeta:
    python external_validation/uci_hydraulic.py --data-dir /ruta/Sensors_Target

La rutina agrega cada ciclo por media, desviación, mínimo y máximo y evalúa una
clasificación del estado de la válvula. No se presenta como validación hidroeléctrica:
su función es comprobar la portabilidad del pipeline sobre datos físicos independientes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

SENSORS = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]


def summarize_sensor(path: Path, name: str) -> pd.DataFrame:
    x = pd.read_csv(path, sep="\t", header=None)
    return pd.DataFrame({
        f"{name}_mean": x.mean(axis=1),
        f"{name}_std": x.std(axis=1),
        f"{name}_min": x.min(axis=1),
        f"{name}_max": x.max(axis=1),
    })


def load_dataset(data_dir: Path):
    frames = [summarize_sensor(data_dir / f"{s}.txt", s) for s in SENSORS if (data_dir / f"{s}.txt").exists()]
    if len(frames) < 5:
        raise FileNotFoundError("Se requieren al menos cinco archivos de sensores y profile.txt.")
    X = pd.concat(frames, axis=1)
    profile = pd.read_csv(data_dir / "profile.txt", sep="\t", header=None,
                          names=["cooler", "valve", "pump_leakage", "accumulator", "stable"])
    y = (profile["valve"] < 100).astype(int)
    groups = np.arange(len(y)) // 25
    return X, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("reports/uci_external_validation.json"))
    args = ap.parse_args()
    X, y, groups = load_dataset(args.data_dir)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(X, y, groups))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    report = {
        "dataset": "UCI Condition Monitoring of Hydraulic Systems",
        "doi": "10.24432/C5CW21",
        "task": "valve non-optimal vs optimal",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "classification_report": classification_report(y.iloc[te], pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y.iloc[te], pred).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
