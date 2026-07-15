"""Evaluate object detection with one-to-one matching and error taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ("circle", "triangle", "square")
IMAGE_SIZE = 640


def read_labels(path: Path, image_size: int) -> list[tuple[int, np.ndarray]]:
    result = []
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, width, height = map(float, line.split())
        result.append(
            (
                int(class_id),
                np.array(
                    [
                        (cx - width / 2) * image_size,
                        (cy - height / 2) * image_size,
                        (cx + width / 2) * image_size,
                        (cy + height / 2) * image_size,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return result


def iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_second = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_first + area_second - intersection
    return float(intersection / union) if union else 0.0


def prediction_rows(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    return [
        {
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "box": [float(value) for value in box],
            "confidence": float(confidence),
        }
        for box, class_id, confidence in zip(boxes, classes, confidences)
    ]


def evaluate(
    model_path: Path,
    image_dir: Path,
    label_dir: Path,
    output_path: Path,
    *,
    conf: float,
    nms_iou: float,
    match_iou: float,
    device: str,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    image_dir = image_dir.resolve()
    label_dir = label_dir.resolve()
    output_path = output_path.resolve()
    model = YOLO(str(model_path))
    stats = {
        name: {"tp": 0, "fp": 0, "fn": 0, "matched_iou": []}
        for name in CLASS_NAMES
    }
    error_counts = {
        "correct": 0,
        "wrong_class": 0,
        "missed_detection": 0,
        "duplicate_same_class": 0,
        "duplicate_cross_class": 0,
        "background_false_positive": 0,
    }
    per_image: list[dict[str, Any]] = []

    results = model.predict(
        source=str(image_dir),
        conf=conf,
        iou=nms_iou,
        device=device,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path)
        ground_truth = read_labels(label_dir / f"{image_path.stem}.txt", IMAGE_SIZE)
        predictions = prediction_rows(result)
        unmatched_gt = set(range(len(ground_truth)))
        unmatched_predictions = set(range(len(predictions)))
        matches: list[tuple[int, int, float]] = []

        # Greedy one-to-one matching by highest IoU, independent of class. This
        # lets us distinguish a wrong class from a missed localization.
        candidate_pairs = sorted(
            (
                iou(np.asarray(prediction["box"]), target_box),
                prediction_index,
                gt_index,
            )
            for prediction_index, prediction in enumerate(predictions)
            for gt_index, (_, target_box) in enumerate(ground_truth)
        )[::-1]
        for overlap, prediction_index, gt_index in candidate_pairs:
            if overlap < match_iou:
                break
            if prediction_index not in unmatched_predictions or gt_index not in unmatched_gt:
                continue
            unmatched_predictions.remove(prediction_index)
            unmatched_gt.remove(gt_index)
            matches.append((prediction_index, gt_index, overlap))

        image_errors: list[dict[str, Any]] = []
        for prediction_index, gt_index, overlap in matches:
            prediction = predictions[prediction_index]
            gt_class_id, _ = ground_truth[gt_index]
            gt_name = CLASS_NAMES[gt_class_id]
            predicted_name = prediction["class_name"]
            if prediction["class_id"] == gt_class_id:
                stats[gt_name]["tp"] += 1
                stats[gt_name]["matched_iou"].append(overlap)
                error_counts["correct"] += 1
                error_type = "correct"
            else:
                stats[gt_name]["fn"] += 1
                stats[predicted_name]["fp"] += 1
                error_counts["wrong_class"] += 1
                error_type = "wrong_class"
            image_errors.append(
                {
                    "type": error_type,
                    "prediction_index": prediction_index,
                    "ground_truth_index": gt_index,
                    "predicted_class": predicted_name,
                    "ground_truth_class": gt_name,
                    "iou": round(overlap, 6),
                    "confidence": round(float(prediction["confidence"]), 6),
                }
            )

        for gt_index in sorted(unmatched_gt):
            gt_class_id, _ = ground_truth[gt_index]
            gt_name = CLASS_NAMES[gt_class_id]
            stats[gt_name]["fn"] += 1
            error_counts["missed_detection"] += 1
            image_errors.append(
                {
                    "type": "missed_detection",
                    "ground_truth_index": gt_index,
                    "ground_truth_class": gt_name,
                }
            )

        for prediction_index in sorted(unmatched_predictions):
            prediction = predictions[prediction_index]
            predicted_name = prediction["class_name"]
            overlaps = [
                (iou(np.asarray(prediction["box"]), target_box), CLASS_NAMES[class_id])
                for class_id, target_box in ground_truth
            ]
            best_overlap, best_gt_name = max(overlaps, default=(0.0, None))
            stats[predicted_name]["fp"] += 1
            if best_overlap >= 0.20 and best_gt_name == predicted_name:
                error_type = "duplicate_same_class"
            elif best_overlap >= 0.20:
                error_type = "duplicate_cross_class"
            else:
                error_type = "background_false_positive"
            error_counts[error_type] += 1
            image_errors.append(
                {
                    "type": error_type,
                    "prediction_index": prediction_index,
                    "predicted_class": predicted_name,
                    "confidence": round(float(prediction["confidence"]), 6),
                    "best_overlap": round(float(best_overlap), 6),
                    "nearest_ground_truth_class": best_gt_name,
                }
            )

        per_image.append(
            {
                "image": image_path.relative_to(image_dir).as_posix(),
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
                "exact_count": len(ground_truth) == len(predictions),
                "predictions": predictions,
                "errors": image_errors,
            }
        )

    summary: dict[str, Any] = {}
    for name, values in stats.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "mean_iou": round(float(np.mean(values["matched_iou"])), 6)
            if values["matched_iou"]
            else 0.0,
        }

    total_tp = sum(summary[name]["tp"] for name in CLASS_NAMES)
    total_fp = sum(summary[name]["fp"] for name in CLASS_NAMES)
    total_fn = sum(summary[name]["fn"] for name in CLASS_NAMES)
    exact_count = sum(1 for row in per_image if row["exact_count"])
    output: dict[str, Any] = {
        "model": str(model_path),
        "images": str(image_dir),
        "labels": str(label_dir),
        "config": {
            "confidence": conf,
            "nms_iou": nms_iou,
            "match_iou": match_iou,
            "device": device,
        },
        "classes": summary,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(total_tp / (total_tp + total_fp), 6)
            if total_tp + total_fp
            else 0.0,
            "recall": round(total_tp / (total_tp + total_fn), 6)
            if total_tp + total_fn
            else 0.0,
            "exact_count_accuracy": round(exact_count / len(per_image), 6)
            if per_image
            else 0.0,
            "images": len(per_image),
        },
        "error_counts": error_counts,
        "per_image": per_image,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "outputs/shapes_v2/baseline_v2/weights/best.pt",
    )
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/shapes_v2/images/multi_test")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data/shapes_v2/labels/multi_test")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/shapes_v2/baseline_v2/evaluation/multi_detection_metrics.json",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = evaluate(
        args.model,
        args.images,
        args.labels,
        args.output,
        conf=args.conf,
        nms_iou=args.nms_iou,
        match_iou=args.match_iou,
        device=args.device,
    )
    print(json.dumps({key: output[key] for key in ("overall", "classes", "error_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
