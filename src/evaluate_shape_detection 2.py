"""Evaluate YOLO11 detection on the multi-object shape stress test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "shapes" / "baseline" / "weights" / "best.pt"
IMAGE_DIR = PROJECT_ROOT / "data" / "shapes" / "images" / "multi_test"
LABEL_DIR = PROJECT_ROOT / "data" / "shapes" / "labels" / "multi_test"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shapes" / "multi_detection_metrics.json"
CLASS_NAMES = ("circle", "triangle", "square")
IMAGE_SIZE = 640


def read_labels(path: Path) -> list[tuple[int, np.ndarray]]:
    result = []
    for line in path.read_text().splitlines():
        class_id, cx, cy, width, height = map(float, line.split())
        result.append(
            (
                int(class_id),
                np.array(
                    [
                        (cx - width / 2) * IMAGE_SIZE,
                        (cy - height / 2) * IMAGE_SIZE,
                        (cx + width / 2) * IMAGE_SIZE,
                        (cy + height / 2) * IMAGE_SIZE,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return result


def iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return float(intersection / union) if union else 0.0


def evaluate() -> None:
    model = YOLO(str(MODEL_PATH))
    stats = {
        name: {"tp": 0, "fp": 0, "fn": 0, "matched_iou": []}
        for name in CLASS_NAMES
    }

    for result in model.predict(
        source=str(IMAGE_DIR), conf=0.25, iou=0.7, device="cpu", verbose=False
    ):
        image_path = Path(result.path)
        ground_truth = read_labels(LABEL_DIR / f"{image_path.stem}.txt")
        predictions = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            predictions = [
                (int(class_id), box, float(confidence))
                for box, class_id, confidence in zip(boxes, classes, confidences)
            ]

        used_predictions: set[int] = set()
        for class_id, target_box in ground_truth:
            candidates = [
                (index, prediction)
                for index, prediction in enumerate(predictions)
                if index not in used_predictions and prediction[0] == class_id
            ]
            best = max(
                candidates,
                key=lambda item: iou(item[1][1], target_box),
                default=None,
            )
            name = CLASS_NAMES[class_id]
            if best is not None and iou(best[1][1], target_box) >= 0.5:
                used_predictions.add(best[0])
                stats[name]["tp"] += 1
                stats[name]["matched_iou"].append(iou(best[1][1], target_box))
            else:
                stats[name]["fn"] += 1

        for index, prediction in enumerate(predictions):
            if index not in used_predictions:
                stats[CLASS_NAMES[prediction[0]]]["fp"] += 1

    summary = {}
    for name, values in stats.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        summary[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "mean_iou": round(float(np.mean(values["matched_iou"])), 6)
            if values["matched_iou"]
            else 0.0,
        }

    total = {key: sum(summary[name][key] for name in CLASS_NAMES) for key in ("tp", "fp", "fn")}
    total["precision"] = round(total["tp"] / (total["tp"] + total["fp"]), 6)
    total["recall"] = round(total["tp"] / (total["tp"] + total["fn"]), 6)
    output = {"iou_threshold": 0.5, "classes": summary, "overall": total}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    evaluate()
