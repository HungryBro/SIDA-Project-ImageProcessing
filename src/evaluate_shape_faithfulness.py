"""Run a simple deletion test: CAM-selected pixels vs random pixels."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "shapes_v2" / "baseline_v2" / "weights" / "best.pt"
IMAGE_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "labels"
CAM_DIR = PROJECT_ROOT / "outputs" / "shapes_v2" / "baseline_v2" / "cam" / "gradcam" / "raw_heatmaps"
MASK_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "masks"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shapes_v2" / "baseline_v2" / "evaluation" / "faithfulness_metrics.csv"
CLASS_NAMES = ("circle", "triangle", "square")
IMAGE_SIZE = 640


def read_target(label_path: Path, class_id: int) -> np.ndarray:
    for line in label_path.read_text().splitlines():
        values = list(map(float, line.split()))
        if int(values[0]) != class_id:
            continue
        _, cx, cy, width, height = values
        return np.array(
            [
                (cx - width / 2) * IMAGE_SIZE,
                (cy - height / 2) * IMAGE_SIZE,
                (cx + width / 2) * IMAGE_SIZE,
                (cy + height / 2) * IMAGE_SIZE,
            ],
            dtype=np.float32,
        )
    raise ValueError(f"Class {class_id} not found in {label_path}")


def iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def confidence_for_target(result, class_id: int, target_box: np.ndarray) -> float:
    if result.boxes is None:
        return 0.0
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    candidates = [
        float(confidence)
        for box, predicted_class, confidence in zip(boxes, classes, confidences)
        if predicted_class == class_id and iou(box, target_box) >= 0.5
    ]
    return max(candidates, default=0.0)


def masked_image(image: np.ndarray, mask: np.ndarray, fill_value: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[mask] = fill_value
    return output


def run(
    model_path: Path = MODEL_PATH,
    image_dir: Path = IMAGE_DIR,
    label_dir: Path = LABEL_DIR,
    cam_dir: Path = CAM_DIR,
    mask_dir: Path = MASK_DIR,
    output_path: Path = OUTPUT_PATH,
) -> None:
    model = YOLO(str(model_path))
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        for heatmap_path in sorted((cam_dir / class_name).glob("*.npy")):
            stem = heatmap_path.stem
            image = cv2.imread(str(image_dir / f"{stem}.jpg"))
            heatmap = np.load(heatmap_path).astype(np.float32)
            object_mask = cv2.imread(
                str(mask_dir / f"{stem}_{class_name}.png"), cv2.IMREAD_GRAYSCALE
            )
            if image is None or object_mask is None:
                continue
            target_box = read_target(label_dir / f"{stem}.txt", class_id)

            x1, y1, x2, y2 = map(int, target_box)
            region = np.zeros_like(object_mask, dtype=bool)
            region[y1:y2, x1:x2] = object_mask[y1:y2, x1:x2] > 0
            region_indices = np.flatnonzero(region)
            if not len(region_indices):
                continue

            count = max(1, int(len(region_indices) * 0.2))
            ranked = region_indices[np.argsort(heatmap.flat[region_indices])]
            top_indices = ranked[-count:]
            random_indices = rng.choice(region_indices, size=count, replace=False)
            top_mask = np.zeros_like(region)
            random_mask = np.zeros_like(region)
            top_mask.flat[top_indices] = True
            random_mask.flat[random_indices] = True

            background_pixels = image[object_mask == 0]
            fill_value = (
                np.median(background_pixels, axis=0).astype(np.uint8)
                if len(background_pixels)
                else np.array([230, 230, 230], dtype=np.uint8)
            )
            original_result = model.predict(
                source=image, conf=0.0, iou=0.7, device="cpu", verbose=False
            )[0]
            top_result = model.predict(
                source=masked_image(image, top_mask, fill_value),
                conf=0.0,
                iou=0.7,
                device="cpu",
                verbose=False,
            )[0]
            random_result = model.predict(
                source=masked_image(image, random_mask, fill_value),
                conf=0.0,
                iou=0.7,
                device="cpu",
                verbose=False,
            )[0]
            original_conf = confidence_for_target(original_result, class_id, target_box)
            top_conf = confidence_for_target(top_result, class_id, target_box)
            random_conf = confidence_for_target(random_result, class_id, target_box)
            rows.append(
                {
                    "image": stem,
                    "class_name": class_name,
                    "original_confidence": round(original_conf, 6),
                    "top_cam_confidence": round(top_conf, 6),
                    "random_confidence": round(random_conf, 6),
                    "top_cam_drop": round(original_conf - top_conf, 6),
                    "random_drop": round(original_conf - random_conf, 6),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} faithfulness rows to {output_path}")
    for class_name in CLASS_NAMES:
        subset = [row for row in rows if row["class_name"] == class_name]
        if subset:
            print(
                f"{class_name}: top_cam_drop={np.mean([r['top_cam_drop'] for r in subset]):.3f}, "
                f"random_drop={np.mean([r['random_drop'] for r in subset]):.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument("--cam-dir", type=Path, default=CAM_DIR)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    run(args.model, args.images, args.labels, args.cam_dir, args.mask_dir, args.output)
