"""Central configuration for CrowdShield AI.

Keep thresholds, paths, and model settings here so the dashboard and scripts
share the same behavior.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_DIR = PROJECT_ROOT / "models"

UMN_VIDEO_PATH = DATA_DIR / "Crowd-Activity-All.avi"
VIDEOS_DIR = DATA_DIR / "videos"
SHANGHAI_ROOT = DATA_DIR / "ShanghaiTech"
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_PERSON_CLASS = 0
CONFIDENCE_THRESHOLD = 0.30
INFERENCE_WIDTH = 320
FRAME_STRIDE = 3
ANALYSIS_SAMPLES_PER_SECOND = 4.5
GRID_ROWS = 3
GRID_COLS = 3
HEATMAP_SIGMA = 15.0
HEATMAP_SCALE_PEOPLE = 12.0
MAX_ZONE_PEOPLE = 12.0
ENABLE_DENSITY_MODEL = False  # TODO: add a pretrained density model hook.
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "annotated_demo.mp4"
RISK_LOG_PATH = OUTPUT_DIR / "risk_log.csv"
EVAL_RESULTS_PATH = OUTPUT_DIR / "eval_results.csv"

RISK_WEIGHTS = {
    "density": 0.40,
    "flow_deviation": 0.12,
    "speed_drop": 0.16,
    "direction_conflict": 0.10,
    "speed_surge": 0.16,
    "trend": 0.06,
}
RISK_LEVEL_THRESHOLDS = {"green": 30.0, "yellow": 55.0, "orange": 80.0}
FLOW_HISTORY_SECONDS = 5.0
FLOW_SLOW_SPEED_PX_PER_SECOND = 2.0
FLOW_DROP_WINDOW_SECONDS = 3.0
TREND_WINDOW_SECONDS = 5.0
RISK_SAMPLE_SECONDS = 0.5
DEMO_MAX_SAMPLES = 120
FLOW_SPEED_REFERENCE_PX_PER_SECOND = 10.0
FLOW_BASELINE_SECONDS = 5.0
DENSITY_SMOOTHING_WINDOW_SECONDS = 6.5
FLOW_SMOOTHING_WINDOW_SECONDS = 6.5
SUSTAINED_ORANGE_SAMPLES = 3
SCENE_STARTS_S: list[float] = [0]      # user-provided scene boundary
PANIC_STARTS_S: list[float] = [19]     # user-provided panic marker
DENSITY_HIGH_THRESHOLD = 0.7
SPEED_DROP_ALERT_THRESHOLD = 0.3
DIRECTION_CONFLICT_THRESHOLD = 0.3
SPEED_SURGE_THRESHOLD = 0.5
ALERT_COOLDOWN_SECONDS = 30.0
VOICE_LANGUAGES = ("en", "hi", "te")
VOICE_CACHE_DIR = OUTPUT_DIR / "voice_cache"