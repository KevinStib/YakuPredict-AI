from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DB_PATH


def init_db(path: Path = DB_PATH) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                risk_score REAL NOT NULL,
                level TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                supervised_probability REAL NOT NULL,
                anomaly_probability REAL NOT NULL,
                recommendation TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction(
    *,
    payload: dict[str, Any],
    risk_score: float,
    level: str,
    prediction: int,
    supervised_probability: float,
    anomaly_probability: float,
    recommendation: str,
    path: Path = DB_PATH,
) -> tuple[int, str]:
    init_db(path)
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions(
                created_at, risk_score, level, prediction,
                supervised_probability, anomaly_probability,
                recommendation, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                float(risk_score),
                level,
                int(prediction),
                float(supervised_probability),
                float(anomaly_probability),
                recommendation,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid), created_at


def recent_predictions(limit: int = 20, path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, risk_score, level, prediction,
                   supervised_probability, anomaly_probability,
                   recommendation, payload_json
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "risk_score": row["risk_score"],
            "risk_percent": round(row["risk_score"] * 100, 2),
            "level": row["level"],
            "prediction": row["prediction"],
            "supervised_probability": row["supervised_probability"],
            "anomaly_probability": row["anomaly_probability"],
            "recommendation": row["recommendation"],
            "payload": json.loads(row["payload_json"]),
        }
        for row in rows
    ]


def clear_predictions(path: Path = DB_PATH) -> None:
    if path.exists():
        path.unlink()
