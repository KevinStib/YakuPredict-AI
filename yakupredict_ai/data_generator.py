from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_PATH, SETTINGS


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: int
    regime: str
    load_center: float
    ambient_offset: float
    noise_scale: float
    degradation_gain: float
    pressure_shift: float
    sensor_bias: float


def _scenario_config(scenario_id: int, stress: bool = False) -> ScenarioConfig:
    rng = np.random.default_rng(10_000 + scenario_id)
    if stress:
        return ScenarioConfig(
            scenario_id=scenario_id,
            regime="stress_shift",
            load_center=float(rng.uniform(0.58, 0.82)),
            ambient_offset=float(rng.uniform(4.0, 9.0)),
            noise_scale=float(rng.uniform(1.5, 2.2)),
            degradation_gain=float(rng.uniform(0.72, 1.18)),
            pressure_shift=float(rng.uniform(-4.5, 3.0)),
            sensor_bias=float(rng.uniform(-0.35, 0.45)),
        )
    return ScenarioConfig(
        scenario_id=scenario_id,
        regime="nominal_family",
        load_center=float(rng.uniform(0.64, 0.76)),
        ambient_offset=float(rng.uniform(-2.0, 2.0)),
        noise_scale=float(rng.uniform(0.85, 1.20)),
        degradation_gain=float(rng.uniform(0.90, 1.10)),
        pressure_shift=float(rng.uniform(-1.2, 1.2)),
        sensor_bias=float(rng.uniform(-0.12, 0.12)),
    )


def _degradation_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera procesos latentes de degradación; nunca se entregan como entrada al modelo."""
    degradation = np.zeros(n, dtype=float)
    cursor = int(rng.integers(120, 260))
    while cursor < n - 180:
        gap = int(rng.integers(170, 360))
        cursor += gap
        if cursor >= n - 120:
            break
        length = int(rng.integers(70, 170))
        peak = float(rng.uniform(0.76, 1.0))
        end = min(cursor + length, n)
        ramp = np.linspace(0.02, peak, end - cursor)
        degradation[cursor:end] = np.maximum(degradation[cursor:end], ramp)
        recovery_end = min(end + int(rng.integers(20, 60)), n)
        if recovery_end > end:
            recovery = np.linspace(max(peak * 0.45, 0.25), 0.0, recovery_end - end)
            degradation[end:recovery_end] = np.maximum(degradation[end:recovery_end], recovery)
        cursor = recovery_end
    return degradation


def _future_label(degradation: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """Etiqueta anticipatoria: 1 si la degradación latente superará el umbral en el horizonte futuro."""
    n = len(degradation)
    label = np.zeros(n, dtype=int)
    for i in range(n):
        future = degradation[i + 1 : min(i + 1 + horizon, n)]
        label[i] = int(len(future) > 0 and np.max(future) >= threshold)
    return label


def generate_scenario(n: int, seed: int, config: ScenarioConfig) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ts = pd.date_range("2026-01-01", periods=n, freq="5min") + pd.to_timedelta(config.scenario_id * n * 5, unit="min")

    daily = 0.105 * np.sin(2 * np.pi * t / 288 + rng.uniform(-0.4, 0.4))
    weekly = 0.045 * np.sin(2 * np.pi * t / (288 * 7) + rng.uniform(-0.3, 0.3))
    load = np.clip(config.load_center + daily + weekly + rng.normal(0, 0.024 * config.noise_scale, n), 0.34, 1.00)
    ambient = 19 + config.ambient_offset + 5.2 * np.sin(2 * np.pi * (t - 64) / 288) + rng.normal(0, 0.75 * config.noise_scale, n)

    degradation = np.clip(_degradation_profile(n, rng) * config.degradation_gain, 0, 1)

    wicket = np.clip(16 + 82 * load + rng.normal(0, 1.3 * config.noise_scale, n), 3, 100)
    flow = np.clip(25 + 76 * load + rng.normal(0, 1.6 * config.noise_scale, n), 18, 115)
    pressure = 61 + 18 * load + config.pressure_shift + rng.normal(0, 0.62 * config.noise_scale, n)
    current = 470 + 1170 * load + rng.normal(0, 28 * config.noise_scale, n)

    # La degradación afecta las señales de forma continua, sin reglas de etiqueta basadas en umbrales observables.
    vibration = 1.12 + 1.05 * load + 3.85 * degradation + config.sensor_bias + rng.normal(0, 0.20 * config.noise_scale, n)
    oil_temp = 39.5 + 15.2 * load + 8.4 * degradation + 0.10 * ambient + rng.normal(0, 0.85 * config.noise_scale, n)
    bearing_temp = 47 + 17.5 * load + 14.8 * degradation + 0.13 * ambient + rng.normal(0, 1.05 * config.noise_scale, n)
    pressure = pressure - 4.5 * degradation + rng.normal(0, 0.32 * config.noise_scale, n)
    current = current + 48 * degradation
    base_power = 6.5 + 89 * load
    power = base_power * (1 - 0.038 * degradation) + rng.normal(0, 1.05 * config.noise_scale, n)
    power = np.clip(power, 4, 105)

    future_risk = _future_label(
        degradation,
        horizon=SETTINGS.prediction_horizon_steps,
        threshold=SETTINGS.future_degradation_threshold,
    )
    current_degraded = (degradation >= SETTINGS.future_degradation_threshold).astype(int)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "scenario_id": config.scenario_id,
            "regime": config.regime,
            "flow_m3s": flow,
            "power_mw": power,
            "vibration_mms": vibration,
            "bearing_temp_c": bearing_temp,
            "oil_temp_c": oil_temp,
            "hydraulic_pressure_bar": pressure,
            "wicket_gate_pct": wicket,
            "stator_current_a": current,
            "ambient_temp_c": ambient,
            "degradation_index_hidden": degradation,
            "current_degraded_hidden": current_degraded,
            "future_degradation_risk": future_risk,
        }
    )


def generate_dataset(rows_per_scenario: int = SETTINGS.dataset_rows_per_scenario, seed: int = SETTINGS.random_state) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total_regular = SETTINGS.n_train_scenarios + SETTINGS.n_validation_scenarios + SETTINGS.n_test_scenarios
    for sid in range(total_regular):
        frames.append(generate_scenario(rows_per_scenario, seed + sid * 101, _scenario_config(sid, stress=False)))
    for offset in range(SETTINGS.n_stress_scenarios):
        sid = total_regular + offset
        frames.append(generate_scenario(rows_per_scenario, seed + sid * 101, _scenario_config(sid, stress=True)))
    return pd.concat(frames, ignore_index=True)


def save_dataset(path: Path = DATASET_PATH, rows_per_scenario: int = SETTINGS.dataset_rows_per_scenario, seed: int = SETTINGS.random_state) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(rows_per_scenario=rows_per_scenario, seed=seed).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera escenarios sintéticos reproducibles de YakuPredict AI 3.0.")
    parser.add_argument("--rows-per-scenario", type=int, default=SETTINGS.dataset_rows_per_scenario)
    parser.add_argument("--seed", type=int, default=SETTINGS.random_state)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    path = save_dataset(args.output, args.rows_per_scenario, args.seed)
    print(f"Dataset generado: {path}")


if __name__ == "__main__":
    main()
