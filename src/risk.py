"""Configurable, explainable crowd-risk scoring and time-series logging."""

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class RiskSnapshot:
    zone_risk: np.ndarray
    overall_risk: float
    level: str
    top_zone: str
    why: str
    trend: float
    flow_deviation: np.ndarray
    speed_surge: np.ndarray


def risk_level(value: float) -> str:
    thresholds = config.RISK_LEVEL_THRESHOLDS
    if value >= thresholds["orange"]:
        return "Red"
    if value >= thresholds["yellow"]:
        return "Orange"
    if value >= thresholds["green"]:
        return "Yellow"
    return "Green"


class RiskEngine:
    """Score zones using fixed-scale density and measured flow features."""

    def __init__(self) -> None:
        self.history: list[float] = []

    def score(
        self,
        zone_counts: np.ndarray,
        speeds: np.ndarray,
        speed_drops: np.ndarray,
        conflicts: np.ndarray,
        baseline_speeds: np.ndarray,
        speed_acceleration: np.ndarray,
        dispersion_rise: np.ndarray,
    ) -> RiskSnapshot:
        density = np.clip(zone_counts / config.MAX_ZONE_PEOPLE, 0.0, 1.0)
        presence = np.clip(zone_counts / max(config.MAX_ZONE_PEOPLE * 0.25, 1.0), 0.0, 1.0)
        flow_deviation = np.clip(
            np.abs(speeds - baseline_speeds)
            / np.maximum(baseline_speeds, config.FLOW_SPEED_REFERENCE_PX_PER_SECOND),
            0.0,
            1.0,
        )
        speed_surge = np.clip(
            np.maximum(speed_acceleration, 0.0)
            / max(config.FLOW_SPEED_REFERENCE_PX_PER_SECOND, 1.0)
            + np.maximum(dispersion_rise, 0.0),
            0.0,
            1.0,
        )
        trend = 0.0
        if self.history:
            trend = float(np.clip((self.history[-1] - np.mean(self.history[-3:])) / 100.0, 0.0, 1.0))
        weights = config.RISK_WEIGHTS
        zone_risk = 100.0 * (
            weights["density"] * density
            + presence * (
                weights["flow_deviation"] * flow_deviation
                + weights["speed_drop"] * speed_drops
                + weights["direction_conflict"] * conflicts
                + weights["speed_surge"] * speed_surge
            )
            + weights["trend"] * trend
        )
        zone_risk = zone_risk * (presence > 0).astype(float)
        maximum = float(zone_risk.max()) if zone_risk.size else 0.0
        self.history.append(maximum)
        overall = float(np.clip(maximum, 0.0, 100.0))
        top_index = int(np.argmax(zone_risk)) if zone_risk.size else 0
        top_zone = f"Z{top_index + 1}"
        reasons = []
        if density[top_index] >= config.DENSITY_HIGH_THRESHOLD:
            reasons.append("high fixed-scale density")
        if speed_drops[top_index] >= config.SPEED_DROP_ALERT_THRESHOLD:
            reasons.append("speed drop")
        if conflicts[top_index] >= config.DIRECTION_CONFLICT_THRESHOLD:
            reasons.append("direction conflict")
        if speed_surge[top_index] >= config.SPEED_SURGE_THRESHOLD:
            reasons.append("sudden speed surge")
        if not reasons:
            reasons.append("measured flow and density signals")
        return RiskSnapshot(
            zone_risk, overall, risk_level(overall), top_zone, ", ".join(reasons),
            trend, flow_deviation, speed_surge,
        )


def sustained_orange_index(risks: list[float]) -> int | None:
    """Return the first sample in a sustained Orange run."""
    threshold = config.RISK_LEVEL_THRESHOLDS["yellow"]
    needed = max(1, config.SUSTAINED_ORANGE_SAMPLES)
    for index in range(len(risks) - needed + 1):
        if all(value >= threshold for value in risks[index:index + needed]):
            return index
    return None


def sustained_orange_indices(risks: list[float]) -> list[int]:
    """Return every first index of a sustained Orange run."""
    threshold = config.RISK_LEVEL_THRESHOLDS["yellow"]
    needed = max(1, config.SUSTAINED_ORANGE_SAMPLES)
    return [
        index for index in range(len(risks) - needed + 1)
        if all(value >= threshold for value in risks[index:index + needed])
        and (index == 0 or risks[index - 1] < threshold)
    ]


def early_warning_lead_time(
    risks: list[float], timestamps: list[float], panic_times: list[float]
) -> float | None:
    """Return seconds between sustained Orange and the first configured panic marker."""
    if not panic_times:
        return None
    orange_index = sustained_orange_index(risks)
    if orange_index is None:
        return None
    panic_time = min(panic_times)
    return panic_time - timestamps[orange_index]


def write_risk_log(rows: list[dict[str, object]], path: Path = config.RISK_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_seconds", "overall_risk", "level", "top_zone", "why", "trend", "zone_risk"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
