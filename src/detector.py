"""YOLO-based person detection and counting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    """One detected person in xyxy pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class PersonDetector:
    """Lazy-loading YOLO person detector that works on CPU or GPU."""

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL_NAME,
        confidence: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed. Run the project venv install command."
                ) from exc
            self._model = YOLO(self.model_name)
        return self._model

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Return person detections for a BGR OpenCV frame."""
        if frame is None or frame.size == 0:
            return []
        results = self._load_model().predict(
            source=frame,
            classes=[config.YOLO_PERSON_CLASS],
            conf=self.confidence,
            imgsz=config.INFERENCE_WIDTH,
            device="cpu",
            verbose=False,
        )
        detections: list[Detection] = []
        boxes = results[0].boxes if results else None
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        for coords, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = [int(value) for value in coords]
            detections.append(Detection(x1, y1, x2, y2, float(confidence)))
        return detections


def resize_for_inference(frame: np.ndarray, width: int = config.INFERENCE_WIDTH) -> np.ndarray:
    """Resize while preserving aspect ratio and avoid enlarging small frames."""
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    new_height = max(1, round(height * width / current_width))
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw person boxes and a corner count panel on a copy of the frame."""
    output = frame.copy()
    for detection in detections:
        cv2.rectangle(
            output,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            (0, 220, 255),
            2,
        )
    cv2.rectangle(output, (8, 8), (170, 48), (7, 13, 24), -1)
    cv2.putText(
        output,
        f"People: {len(detections)}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def video_is_readable(path: Path) -> bool:
    """Return whether OpenCV can open the selected video."""
    capture = cv2.VideoCapture(str(path))
    readable = capture.isOpened()
    capture.release()
    return readable
