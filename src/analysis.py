"""Real-video sampled analysis used by the reliable Streamlit demo."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from . import config
from .density import density_heatmap, draw_zone_grid, heatmap_overlay, zone_counts
from .detector import PersonDetector, annotate_frame, resize_for_inference
from .flow import calculate_flow
from .risk import (
    RiskEngine,
    RiskSnapshot,
    early_warning_lead_time,
    sustained_orange_indices,
    write_risk_log,
)


@dataclass
class AnalysisResult:
    frames: list[np.ndarray]
    heatmaps: list[np.ndarray]
    timestamps: list[float]
    counts: list[float]
    zone_counts: list[np.ndarray]
    risks: list[RiskSnapshot]
    slow_zones: list[list[str]]
    action_inputs: list[tuple[str, str, int, float, float]]
    video_fps: float
    effective_fps: float
    processing_fps: float
    early_warning: float | None
    panic_validation: list[dict[str, object]]
    scene_boundaries: list[float]
    mean_flow_speed: list[float]
    surge_signal: list[float]


def _rolling(values: np.ndarray, window: int, median: bool) -> np.ndarray:
    """Smooth each sampled feature using a centered rolling statistic."""
    if len(values) == 0 or window <= 1:
        return values.astype(np.float32)
    window = min(window, len(values))
    if median and window % 2 == 0:
        window -= 1
    window = max(1, window)
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(values, dtype=np.float32)
    for index in range(len(values)):
        sample_window = padded[index:index + window]
        output[index] = np.median(sample_window, axis=0) if median else np.mean(sample_window, axis=0)
    return output


def _scene_index(timestamp: float) -> int:
    return sum(timestamp >= start for start in sorted(config.SCENE_STARTS_S))


def _scene_baselines(
    timestamps: list[float], speeds: np.ndarray, scene_boundaries: list[float]
) -> np.ndarray:
    baselines = np.zeros_like(speeds, dtype=np.float32)
    starts = [0.0] + scene_boundaries
    ends = scene_boundaries + [float("inf")]
    for start, end in zip(starts, ends):
        indices = [
            index for index, timestamp in enumerate(timestamps)
            if start <= timestamp < end and timestamp < start + config.FLOW_BASELINE_SECONDS
        ]
        if indices:
            baselines[indices[0]:indices[-1] + 1] = np.mean(speeds[indices], axis=0)
            later = [index for index, timestamp in enumerate(timestamps) if start <= timestamp < end]
            baselines[later[0]:later[-1] + 1] = np.mean(speeds[indices], axis=0)
    return baselines


def _panic_validation(timestamps: list[float], risks: list[RiskSnapshot]) -> list[dict[str, object]]:
    values = np.asarray([risk.overall_risk for risk in risks], dtype=np.float32)
    orange_times = [
        timestamps[index] for index in sustained_orange_indices(values.tolist())
    ]
    rows = []
    for panic_time in sorted(config.PANIC_STARTS_S):
        before = values[
            (np.asarray(timestamps) >= panic_time - 10.0)
            & (np.asarray(timestamps) < panic_time)
        ]
        after = values[
            (np.asarray(timestamps) >= panic_time)
            & (np.asarray(timestamps) < panic_time + 10.0)
        ]
        reached = "n/a"
        if any(orange_time < panic_time for orange_time in orange_times):
            reached = "before marker"
        elif any(orange_time >= panic_time for orange_time in orange_times):
            reached = "after marker"
        rows.append({
            "panic_time": panic_time,
            "mean_risk_before_10s": float(np.mean(before)) if before.size else None,
            "mean_risk_after_10s": float(np.mean(after)) if after.size else None,
            "sustained_orange": reached,
        })
    return rows


def analyze_video(path, progress=None) -> AnalysisResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(video_fps / config.ANALYSIS_SAMPLES_PER_SECOND))
    sample_seconds = stride / video_fps
    detector = PersonDetector()
    frames: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    timestamps: list[float] = []
    counts: list[int] = []
    all_zone_counts: list[np.ndarray] = []
    raw_speeds: list[np.ndarray] = []
    raw_conflicts: list[np.ndarray] = []
    slow_zones: list[list[str]] = []
    previous_gray = None
    started = time.perf_counter()
    frame_index = 0
    while True:
        ok, raw_frame = capture.read()
        if not ok:
            break
        frame = resize_for_inference(raw_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = calculate_flow(previous_gray, gray, None, video_fps)
        previous_gray = gray
        if frame_index % stride == 0:
            detections = detector.predict(frame)
            zones = zone_counts(detections, frame.shape).reshape(-1)
            timestamp = frame_index / video_fps
            frames.append(draw_zone_grid(
                annotate_frame(frame, detections),
                zones.reshape(config.GRID_ROWS, config.GRID_COLS),
            ))
            heatmaps.append(heatmap_overlay(frame, density_heatmap(detections, frame.shape)))
            timestamps.append(timestamp)
            counts.append(len(detections))
            all_zone_counts.append(zones)
            raw_speeds.append(flow.speed_by_zone)
            raw_conflicts.append(flow.direction_conflict_by_zone)
            slow_zones.append(flow.slow_zones)
            if progress:
                progress(min((frame_index + 1) / max(total, 1), 1.0))
        frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError("The video opened but no sampled frame could be decoded.")

    zone_array = np.asarray(all_zone_counts, dtype=np.float32)
    speed_array = np.asarray(raw_speeds, dtype=np.float32)
    conflict_array = np.asarray(raw_conflicts, dtype=np.float32)
    smoothing_window = max(1, round(config.DENSITY_SMOOTHING_WINDOW_SECONDS / sample_seconds))
    flow_window = max(1, round(config.FLOW_SMOOTHING_WINDOW_SECONDS / sample_seconds))
    smoothed_counts = _rolling(zone_array, smoothing_window, median=True)
    smoothed_speeds = _rolling(speed_array, flow_window, median=False)
    smoothed_conflicts = _rolling(conflict_array, flow_window, median=False)
    speed_drops = np.zeros_like(smoothed_speeds)
    if len(smoothed_speeds) > 1:
        speed_drops[1:] = np.clip(
            (smoothed_speeds[:-1] - smoothed_speeds[1:])
            / np.maximum(smoothed_speeds[:-1], 1.0),
            0.0,
            1.0,
        )
    acceleration = np.zeros_like(smoothed_speeds)
    dispersion_rise = np.zeros_like(smoothed_conflicts)
    if len(smoothed_speeds) > 1:
        acceleration[1:] = (smoothed_speeds[1:] - smoothed_speeds[:-1]) / sample_seconds
        dispersion_rise[1:] = smoothed_conflicts[1:] - smoothed_conflicts[:-1]
    scene_boundaries = sorted(
        start for start in config.SCENE_STARTS_S
        if timestamps[0] <= start <= timestamps[-1]
    )
    baselines = _scene_baselines(timestamps, smoothed_speeds, scene_boundaries)
    engine = RiskEngine()
    risks: list[RiskSnapshot] = []
    action_inputs: list[tuple[str, str, int, float, float]] = []
    mean_flow_speed: list[float] = []
    surge_signal: list[float] = []
    for index, zones in enumerate(smoothed_counts):
        risk = engine.score(
            zones,
            smoothed_speeds[index],
            speed_drops[index],
            smoothed_conflicts[index],
            baselines[index],
            acceleration[index],
            dispersion_rise[index],
        )
        risks.append(risk)
        mean_flow_speed.append(float(np.mean(speed_array[index])))
        surge_signal.append(float(np.mean(risk.speed_surge)))
        maximum_zone = int(np.max(zones)) if zones.size else 0
        action_inputs.append((risk.level, risk.top_zone, maximum_zone, risk.overall_risk, timestamps[index]))
    write_risk_log([
        {
            "timestamp_seconds": timestamp,
            "overall_risk": round(risk.overall_risk, 3),
            "level": risk.level,
            "top_zone": risk.top_zone,
            "why": risk.why,
            "trend": round(risk.trend, 3),
            "zone_risk": ",".join(f"{value:.3f}" for value in risk.zone_risk),
        }
        for timestamp, risk in zip(timestamps, risks)
    ])
    elapsed = max(time.perf_counter() - started, 1e-6)
    return AnalysisResult(
        frames,
        heatmaps,
        timestamps,
        [float(np.sum(zones)) for zones in smoothed_counts],
        [zones for zones in smoothed_counts],
        risks,
        slow_zones,
        action_inputs,
        video_fps,
        video_fps / stride,
        len(frames) / elapsed,
        early_warning_lead_time(
            [risk.overall_risk for risk in risks],
            timestamps,
            config.PANIC_STARTS_S,
        ),
        _panic_validation(timestamps, risks),
        scene_boundaries,
        mean_flow_speed,
        surge_signal,
    )
