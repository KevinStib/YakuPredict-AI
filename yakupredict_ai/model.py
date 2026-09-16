from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import MODEL_FEATURES, MODEL_PATH, SETTINGS
from .features import engineer_features, validate_raw_columns

TARGET = "future_degradation_risk"


@dataclass
class HybridBundle:
    supervised: Pipeline
    anomaly: Pipeline
    anomaly_min: float
    anomaly_max: float
    threshold: float
    feature_names: list[str]
    supervised_weight: float
    anomaly_weight: float

    def _anomaly_probability(self, x: pd.DataFrame) -> np.ndarray:
        raw = -self.anomaly.decision_function(x[self.feature_names])
        denominator = max(self.anomaly_max - self.anomaly_min, 1e-9)
        return np.clip((raw - self.anomaly_min) / denominator, 0.0, 1.0)

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_raw_columns(df)
        features = engineer_features(df)
        x = features[self.feature_names]
        p_supervised = self.supervised.predict_proba(x)[:, 1]
        p_anomaly = self._anomaly_probability(x)
        risk = self.supervised_weight * p_supervised + self.anomaly_weight * p_anomaly
        prediction = (risk >= self.threshold).astype(int)
        return pd.DataFrame({
            "supervised_probability": p_supervised,
            "anomaly_probability": p_anomaly,
            "risk_score": risk,
            "prediction": prediction,
        }, index=df.index)


def _build_supervised(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", RandomForestClassifier(
            n_estimators=360,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def _build_anomaly(seed: int, contamination: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    f1_values = [f1_score(y_true, scores >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1_values))])


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }


def _partition_by_scenario(features: pd.DataFrame):
    ntr = SETTINGS.n_train_scenarios
    nva = SETTINGS.n_validation_scenarios
    nte = SETTINGS.n_test_scenarios
    train_ids = list(range(0, ntr))
    val_ids = list(range(ntr, ntr + nva))
    test_ids = list(range(ntr + nva, ntr + nva + nte))
    stress_ids = list(range(ntr + nva + nte, ntr + nva + nte + SETTINGS.n_stress_scenarios))
    return (
        features[features.scenario_id.isin(train_ids)].copy(),
        features[features.scenario_id.isin(val_ids)].copy(),
        features[features.scenario_id.isin(test_ids)].copy(),
        features[features.scenario_id.isin(stress_ids)].copy(),
        {"train": train_ids, "validation": val_ids, "test": test_ids, "stress": stress_ids},
    )


def _normalize_anomaly(anomaly: Pipeline, x_train: pd.DataFrame, x: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    train_raw = -anomaly.decision_function(x_train)
    a_min = float(np.quantile(train_raw, 0.02))
    a_max = float(np.quantile(train_raw, 0.98))
    denom = max(a_max - a_min, 1e-9)
    raw = -anomaly.decision_function(x)
    return np.clip((raw - a_min) / denom, 0.0, 1.0), a_min, a_max


def train_bundle(df: pd.DataFrame) -> tuple[HybridBundle, dict[str, Any], pd.DataFrame]:
    validate_raw_columns(df)
    if TARGET not in df.columns or "scenario_id" not in df.columns:
        raise ValueError(f"El dataset debe contener '{TARGET}' y 'scenario_id'.")

    feat = engineer_features(df)
    train, validation, test, stress, scenario_ids = _partition_by_scenario(feat)

    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET].to_numpy()
    x_val = validation[MODEL_FEATURES]
    y_val = validation[TARGET].to_numpy()
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET].to_numpy()
    x_stress = stress[MODEL_FEATURES]
    y_stress = stress[TARGET].to_numpy()

    supervised = _build_supervised(SETTINGS.random_state)
    supervised.fit(x_train, y_train)

    anomaly = _build_anomaly(SETTINGS.random_state, SETTINGS.contamination)
    normal_train = train.loc[train[TARGET] == 0, MODEL_FEATURES]
    anomaly.fit(normal_train)

    p_sup_val = supervised.predict_proba(x_val)[:, 1]
    p_anom_val, a_min, a_max = _normalize_anomaly(anomaly, x_train, x_val)

    # Umbrales individuales se calibran en validación para una comparación justa.
    t_sup = _best_threshold(y_val, p_sup_val)
    t_anom = _best_threshold(y_val, p_anom_val)

    best = {"f1": -1.0, "w": None, "threshold": None}
    for w in np.arange(0.0, 1.0001, SETTINGS.weight_grid_step):
        scores = w * p_sup_val + (1.0 - w) * p_anom_val
        threshold = _best_threshold(y_val, scores)
        f1 = f1_score(y_val, scores >= threshold, zero_division=0)
        if f1 > best["f1"] + 1e-12:
            best = {"f1": float(f1), "w": float(round(w, 10)), "threshold": float(threshold)}

    w_sup = float(best["w"])
    w_anom = 1.0 - w_sup
    threshold = float(best["threshold"])

    def anomaly_scores(x: pd.DataFrame) -> np.ndarray:
        raw = -anomaly.decision_function(x)
        denom = max(a_max - a_min, 1e-9)
        return np.clip((raw - a_min) / denom, 0.0, 1.0)

    def score_set(x: pd.DataFrame):
        ps = supervised.predict_proba(x)[:, 1]
        pa = anomaly_scores(x)
        return ps, pa, w_sup * ps + w_anom * pa

    p_sup_test, p_anom_test, hybrid_test = score_set(x_test)
    p_sup_stress, p_anom_stress, hybrid_stress = score_set(x_stress)

    comparison_rows = []
    for split_name, y, ps, pa, ph in [
        ("validation", y_val, p_sup_val, p_anom_val, w_sup * p_sup_val + w_anom * p_anom_val),
        ("test", y_test, p_sup_test, p_anom_test, hybrid_test),
        ("stress_shift", y_stress, p_sup_stress, p_anom_stress, hybrid_stress),
    ]:
        for model_name, scores, th in [
            ("Random Forest", ps, t_sup),
            ("Isolation Forest", pa, t_anom),
            ("Híbrido", ph, threshold),
        ]:
            row = {"split": split_name, "model": model_name}
            row.update(_metrics(y, scores, th))
            row["confusion_matrix"] = str(row["confusion_matrix"])
            comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    bundle = HybridBundle(
        supervised=supervised,
        anomaly=anomaly,
        anomaly_min=a_min,
        anomaly_max=a_max,
        threshold=threshold,
        feature_names=list(MODEL_FEATURES),
        supervised_weight=w_sup,
        anomaly_weight=w_anom,
    )

    report = {
        "software": "YakuPredict AI",
        "version": "3.0.0",
        "target": TARGET,
        "prediction_horizon_steps": SETTINGS.prediction_horizon_steps,
        "prediction_horizon_minutes": SETTINGS.prediction_horizon_steps * 5,
        "rows": int(len(df)),
        "scenario_ids": scenario_ids,
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "stress_shift": int(len(stress)),
        },
        "positive_rates": {
            "train": float(train[TARGET].mean()),
            "validation": float(validation[TARGET].mean()),
            "test": float(test[TARGET].mean()),
            "stress_shift": float(stress[TARGET].mean()),
        },
        "fusion": {
            "supervised_weight": w_sup,
            "anomaly_weight": w_anom,
            "threshold": threshold,
            "selection_criterion": "maximización de F1 en escenarios de validación",
        },
        "individual_thresholds": {"Random Forest": t_sup, "Isolation Forest": t_anom},
        "validation": _metrics(y_val, w_sup * p_sup_val + w_anom * p_anom_val, threshold),
        "test": _metrics(y_test, hybrid_test, threshold),
        "stress_shift": _metrics(y_stress, hybrid_stress, threshold),
    }
    return bundle, report, comparison


def feature_importance(bundle: HybridBundle) -> dict[str, float]:
    model = bundle.supervised.named_steps["model"]
    return dict(sorted(zip(bundle.feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True))


def save_bundle(bundle: HybridBundle, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = MODEL_PATH) -> HybridBundle:
    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo en {path}. Ejecute: python -m yakupredict_ai.train")
    return joblib.load(path)
