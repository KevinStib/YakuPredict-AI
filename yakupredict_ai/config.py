from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TEMPLATE_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
SCREENSHOT_DIR = ROOT / "screenshots"
DOCS_DIR = ROOT / "docs"

DATASET_PATH = DATA_DIR / "yakupredict_synthetic_scenarios.csv"
MODEL_PATH = MODEL_DIR / "yakupredict_bundle_v3.joblib"
METRICS_PATH = REPORT_DIR / "training_report_v3.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison_v3.csv"
DB_PATH = ROOT / "yakupredict_predictions.sqlite3"

RAW_FEATURES = [
    "flow_m3s",
    "power_mw",
    "vibration_mms",
    "bearing_temp_c",
    "oil_temp_c",
    "hydraulic_pressure_bar",
    "wicket_gate_pct",
    "stator_current_a",
    "ambient_temp_c",
]

ENGINEERED_FEATURES = [
    "power_per_flow",
    "thermal_delta_c",
    "vibration_delta",
    "bearing_temp_delta",
    "oil_temp_delta",
    "pressure_delta",
    "power_delta",
    "vibration_thermal_interaction",
    "rolling_vibration_mean_3",
    "rolling_bearing_temp_mean_3",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    contamination: float = 0.08
    dataset_rows_per_scenario: int = 1500
    n_train_scenarios: int = 6
    n_validation_scenarios: int = 2
    n_test_scenarios: int = 2
    n_stress_scenarios: int = 3
    prediction_horizon_steps: int = 12  # 60 min para resolución de 5 min
    future_degradation_threshold: float = 0.70
    weight_grid_step: float = 0.05


SETTINGS = Settings()
