"""Farneback optical-flow features calculated independently for each zone."""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class FlowSnapshot:
    speed_by_zone: np.ndarray
    speed_drop_by_zone: np.ndarray
    direction_conflict_by_zone: np.ndarray
    direction_dispersion_by_zone: np.ndarray
    slow_zones: list[str]


def _zone_slices(shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    height, width = shape[:2]
    return [
        (
            int(col * width / config.GRID_COLS),
            int((col + 1) * width / config.GRID_COLS),
            int(row * height / config.GRID_ROWS),
            int((row + 1) * height / config.GRID_ROWS),
        )
        for row in range(config.GRID_ROWS)
        for col in range(config.GRID_COLS)
    ]


def calculate_flow(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    previous_speeds: np.ndarray | None,
    fps: float,
) -> FlowSnapshot:
    """Return Farneback speed, drop, conflict, and slow-zone features."""
    zones = config.GRID_ROWS * config.GRID_COLS
    if previous_gray is None:
        zeros = np.zeros(zones, dtype=np.float32)
        return FlowSnapshot(zeros, zeros.copy(), zeros.copy(), zeros.copy(), [])
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    speed = np.zeros(zones, dtype=np.float32)
    conflict = np.zeros(zones, dtype=np.float32)
    for index, (x1, x2, y1, y2) in enumerate(_zone_slices(current_gray.shape)):
        vectors = flow[y1:y2, x1:x2].reshape(-1, 2)
        if vectors.size == 0:
            continue
        magnitudes = np.linalg.norm(vectors, axis=1) * max(fps, 1.0)
        speed[index] = float(np.mean(magnitudes))
        mean_vector = np.mean(vectors, axis=0)
        mean_magnitude = float(np.mean(np.linalg.norm(vectors, axis=1)))
        if mean_magnitude > 1e-6:
            conflict[index] = float(
                np.clip(1.0 - np.linalg.norm(mean_vector) / mean_magnitude, 0.0, 1.0)
            )
    prior = previous_speeds if previous_speeds is not None else speed
    speed_drop = np.clip((prior - speed) / np.maximum(prior, 1.0), 0.0, 1.0)
    slow_zones = [
        f"Z{index + 1}"
        for index, value in enumerate(speed)
        if value <= config.FLOW_SLOW_SPEED_PX_PER_SECOND
    ]
    return FlowSnapshot(speed, speed_drop.astype(np.float32), conflict, conflict.copy(), slow_zones)
