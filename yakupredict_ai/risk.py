from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    level: str
    recommendation: str
    priority: int


def classify_risk(score: float) -> RiskDecision:
    """Traduce el score analítico a una categoría orientativa de mantenimiento."""
    if score >= 0.75:
        return RiskDecision(
            "CRÍTICO",
            "Inspección prioritaria. Contrastar vibración, temperatura y presión con tendencias y procedimientos de planta.",
            4,
        )
    if score >= 0.50:
        return RiskDecision(
            "ALTO",
            "Programar inspección en la próxima ventana operativa y confirmar la condición con señales complementarias.",
            3,
        )
    if score >= 0.30:
        return RiskDecision(
            "MEDIO",
            "Incrementar vigilancia, revisar tendencia histórica y verificar la coherencia de la instrumentación.",
            2,
        )
    return RiskDecision(
        "BAJO",
        "Condición compatible con operación normal. Mantener monitorización y seguimiento de tendencia.",
        1,
    )
