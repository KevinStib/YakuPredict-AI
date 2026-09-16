from __future__ import annotations

import json
import pandas as pd

from .config import COMPARISON_PATH, DATASET_PATH, METRICS_PATH, MODEL_PATH
from .data_generator import save_dataset
from .model import feature_importance, save_bundle, train_bundle


def main() -> None:
    if not DATASET_PATH.exists():
        save_dataset(DATASET_PATH)
    dataframe = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    bundle, report, comparison = train_bundle(dataframe)
    save_bundle(bundle, MODEL_PATH)
    report["feature_importance"] = feature_importance(bundle)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison.to_csv(COMPARISON_PATH, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(comparison[["split", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]].to_string(index=False))
    print(f"Modelo: {MODEL_PATH}")
    print(f"Reporte: {METRICS_PATH}")
    print(f"Comparativa: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
