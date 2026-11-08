"""Grid density and heatmap calculations from person detections."""

from typing import Sequence

import cv2
import numpy as np

from . import config
from .detector import Detection


def zone_counts(
    detections: Sequence[Detection],
    frame_shape: tuple[int, ...],
    rows: int = config.GRID_ROWS,
    cols: int = config.GRID_COLS,
) -> np.ndarray:
    """Count detection centers in a configurable rows x columns grid."""
    height, width = frame_shape[:2]
    counts = np.zeros((rows, cols), dtype=np.int32)
    for detection in detections:
        x, y = detection.center
        col = min(cols - 1, max(0, int(x * cols / max(width, 1))))
        row = min(rows - 1, max(0, int(y * rows / max(height, 1))))
        counts[row, col] += 1
    return counts


def density_heatmap(
    detections: Sequence[Detection],
    frame_shape: tuple[int, ...],
    blur_sigma: float = config.HEATMAP_SIGMA,
    scale_people: float = config.HEATMAP_SCALE_PEOPLE,
) -> np.ndarray:
    """Create a fixed-scale Gaussian-smoothed density heatmap."""
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        x, y = detection.center
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1.0
    if blur_sigma > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), blur_sigma)
    return np.clip(heatmap / max(scale_people, 1.0), 0.0, 1.0)


def heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a JET density heatmap on a BGR frame."""
    if heatmap.size == 0:
        return frame.copy()
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)


def draw_zone_grid(
    frame: np.ndarray,
    counts: np.ndarray,
    color: tuple[int, int, int] = (255, 180, 0),
) -> np.ndarray:
    """Draw zone boundaries and counts on a frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    rows, cols = counts.shape
    for row in range(rows):
        for col in range(cols):
            x1, x2 = int(col * width / cols), int((col + 1) * width / cols)
            y1, y2 = int(row * height / rows), int((row + 1) * height / rows)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                output,
                f"Z{row * cols + col + 1}: {int(counts[row, col])}",
                (x1 + 8, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return output
