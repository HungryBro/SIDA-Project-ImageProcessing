"""Run class-specific Grad-CAM on the synthetic shape stress-test images."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gradcam_yolo import yolov8_heatmap  # noqa: E402


CLASS_NAMES = ("circle", "triangle", "square")
MODEL_PATH = PROJECT_ROOT / "outputs" / "shapes" / "baseline_v2" / "weights" / "best.pt"
INPUT_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shapes" / "cam" / "gradcam"
YOLO11_TARGET_LAYERS = [22]


def read_ground_truth(label_path: Path, image_size: int) -> dict[int, np.ndarray]:
    targets: dict[int, np.ndarray] = {}
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id, center_x, center_y, width, height = map(float, fields)
        targets[int(class_id)] = np.array(
            [
                (center_x - width / 2) * image_size,
                (center_y - height / 2) * image_size,
                (center_x + width / 2) * image_size,
                (center_y + height / 2) * image_size,
            ],
            dtype=np.float32,
        )
    return targets


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def save_heatmap(raw_heatmap: np.ndarray, path: Path) -> None:
    heatmap = np.asarray(raw_heatmap).squeeze()
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, None)
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap = heatmap / maximum
    cv2.imwrite(str(path), (heatmap * 255).astype(np.uint8))
    np.save(path.with_suffix(".npy"), heatmap.astype(np.float32))


def run(
    model_path: Path = MODEL_PATH,
    input_dir: Path = INPUT_DIR,
    label_dir: Path = LABEL_DIR,
    output_dir: Path = OUTPUT_DIR,
    conf_threshold: float = 0.15,
) -> None:
    model_path = model_path.resolve()
    input_dir = input_dir.resolve()
    label_dir = label_dir.resolve()
    output_dir = output_dir.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    overlay_dir = output_dir / "overlays"
    raw_dir = output_dir / "raw_heatmaps"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    image_paths = sorted(input_dir.glob("*.jpg"))

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_overlay_dir = overlay_dir / class_name
        class_raw_dir = raw_dir / class_name
        class_overlay_dir.mkdir(parents=True, exist_ok=True)
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running Grad-CAM for class {class_id}: {class_name}")
        cam = yolov8_heatmap(
            weight=str(model_path),
            method="GradCAM",
            layer=YOLO11_TARGET_LAYERS,
            conf_threshold=conf_threshold,
            ratio=0.10,
            show_box=True,
            target_class_ids=[class_id],
            target_output_type="class",
        )
        try:
            for image_path in image_paths:
                stem = image_path.stem
                try:
                    ground_truth = read_ground_truth(
                        label_dir / f"{stem}.txt", image_size=640
                    ).get(class_id)
                    cam.target.target_box = (
                        torch.tensor(ground_truth, dtype=torch.float32, device=cam.device)
                        if ground_truth is not None
                        else None
                    )
                    overlay, raw_heatmap, predictions, _ = cam.process(
                        str(image_path), return_details=True
                    )
                    overlay.save(class_overlay_dir / f"{stem}.jpg")
                    save_heatmap(raw_heatmap, class_raw_dir / f"{stem}.png")

                    prediction_array = (
                        predictions.detach().cpu().numpy()
                        if hasattr(predictions, "detach")
                        else np.asarray(predictions)
                    )
                    class_predictions = [
                        row for row in prediction_array if int(row[5]) == class_id
                    ]
                    matched_iou = 0.0
                    confidence = 0.0
                    if class_predictions and ground_truth is not None:
                        best = max(
                            class_predictions,
                            key=lambda row: box_iou(row[:4], ground_truth),
                        )
                        matched_iou = box_iou(best[:4], ground_truth)
                        confidence = float(best[4])

                    records.append(
                        {
                            "image": str(image_path.relative_to(PROJECT_ROOT)),
                            "class_id": class_id,
                            "class_name": class_name,
                            "overlay": str(
                                (class_overlay_dir / f"{stem}.jpg").relative_to(PROJECT_ROOT)
                            ),
                            "raw_heatmap": str(
                                (class_raw_dir / f"{stem}.npy").relative_to(PROJECT_ROOT)
                            ),
                            "prediction_count": len(class_predictions),
                            "confidence": confidence,
                            "matched_iou": matched_iou,
                        }
                    )
                except Exception as error:  # Keep the batch running for other images.
                    records.append(
                        {
                            "image": str(image_path.relative_to(PROJECT_ROOT)),
                            "class_id": class_id,
                            "class_name": class_name,
                            "error": str(error),
                        }
                    )
                    print(f"  failed {image_path.name}: {error}")
        finally:
            cam.method.activations_and_grads.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2))
    successful = sum("error" not in record for record in records)
    print(f"Saved {successful}/{len(records)} class-image Grad-CAM results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--images", type=Path, default=INPUT_DIR)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--conf-threshold", type=float, default=0.15)
    args = parser.parse_args()
    run(args.model, args.images, args.labels, args.output_dir, args.conf_threshold)
