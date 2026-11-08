"""ShanghaiTech counting evaluation using the YOLO person detector."""

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
from scipy.io import loadmat

from . import config
from .detector import PersonDetector


def ground_truth_count(mat_path: Path) -> int:
    """Count annotated head points in a ShanghaiTech GT .mat file."""
    data = loadmat(mat_path)
    points = data.get("image_info")
    if points is None:
        raise ValueError(f"Missing image_info in {mat_path}")
    nested = np.asarray(points[0, 0][0, 0][0])
    return int(nested.shape[0])


def evaluate_part(part_dir: Path, detector: PersonDetector) -> tuple[float, float, int]:
    """Evaluate every test image in one ShanghaiTech part."""
    image_dir = part_dir / "test_data" / "images"
    gt_dir = part_dir / "test_data" / "ground-truth"
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    processed = 0
    for mat_path in sorted(gt_dir.glob("GT_*.mat")):
        image_path = image_dir / f"{mat_path.stem.replace('GT_', '')}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{mat_path.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        true_count = ground_truth_count(mat_path)
        predicted_count = len(detector.predict(image))
        error = predicted_count - true_count
        absolute_errors.append(abs(error))
        squared_errors.append(error * error)
        processed += 1
    if not absolute_errors:
        raise RuntimeError(f"No evaluable images found in {part_dir}")
    return (
        float(np.mean(absolute_errors)),
        float(np.sqrt(np.mean(squared_errors))),
        processed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=config.SHANGHAI_ROOT)
    args = parser.parse_args()
    detector = PersonDetector()
    rows: list[dict[str, object]] = []
    for part in ("part_A", "part_B"):
        mae, mse, images = evaluate_part(args.root / part, detector)
        rows.append({"part": part, "images": images, "MAE": mae, "MSE": mse})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.EVAL_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["part", "images", "MAE", "MSE"])
        writer.writeheader()
        writer.writerows(rows)
    print("ShanghaiTech evaluation (YOLO under-counting in dense crowds is expected)")
    print(f"{'Part':<10}{'Images':<10}{'MAE':<12}{'MSE':<12}")
    for row in rows:
        print(f"{row['part']:<10}{row['images']:<10}{row['MAE']:<12.3f}{row['MSE']:<12.3f}")


if __name__ == "__main__":
    main()
