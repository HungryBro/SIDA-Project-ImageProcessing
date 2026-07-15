"""Evaluate class-specific shape Grad-CAM against generated rationale masks."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAM_DIR = PROJECT_ROOT / "outputs" / "shapes_v2" / "baseline_v2" / "cam" / "gradcam"
MASK_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "masks"
RATIONALE_DIR = PROJECT_ROOT / "data" / "shapes_v2" / "locked_original" / "rationale_masks"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shapes_v2" / "baseline_v2" / "evaluation" / "cam_metrics.csv"
CLASS_NAMES = ("circle", "triangle", "square")


def energy_in(heatmap: np.ndarray, mask: np.ndarray) -> float:
    total = float(heatmap.sum())
    return float((heatmap * (mask > 0)).sum() / total) if total else 0.0


def top_fraction_mask(heatmap: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    threshold = float(np.quantile(heatmap, 1.0 - fraction))
    return heatmap >= threshold


def evaluate(cam_dir: Path = CAM_DIR, mask_dir: Path = MASK_DIR, rationale_dir: Path = RATIONALE_DIR, output_path: Path = OUTPUT_PATH) -> None:
    rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        heatmap_paths = sorted((cam_dir / "raw_heatmaps" / class_name).glob("*.npy"))
        for heatmap_path in heatmap_paths:
            stem = heatmap_path.stem
            heatmap = np.load(heatmap_path).astype(np.float32)
            object_mask = cv2.imread(
                str(mask_dir / f"{stem}_{class_name}.png"), cv2.IMREAD_GRAYSCALE
            )
            rationale_mask = cv2.imread(
                str(rationale_dir / f"{stem}_{class_name}.png"), cv2.IMREAD_GRAYSCALE
            )
            combined_mask = cv2.imread(
                str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE
            )
            if object_mask is None or rationale_mask is None or combined_mask is None:
                continue

            other_mask = np.logical_and(combined_mask > 0, object_mask == 0)
            top_mask = top_fraction_mask(heatmap)
            rationale = rationale_mask > 0
            top_pixels = int(top_mask.sum())
            rationale_pixels = int(rationale.sum())
            intersection = int(np.logical_and(top_mask, rationale).sum())
            union = int(np.logical_or(top_mask, rationale).sum())
            rationale_area_fraction = rationale_pixels / heatmap.size if heatmap.size else 0.0
            rationale_energy = energy_in(heatmap, rationale_mask)

            rows.append(
                {
                    "image": stem,
                    "class_name": class_name,
                    "object_energy": round(energy_in(heatmap, object_mask), 6),
                    "rationale_energy": round(energy_in(heatmap, rationale_mask), 6),
                    "rationale_lift": round(
                        rationale_energy / rationale_area_fraction,
                        6,
                    )
                    if rationale_area_fraction
                    else 0.0,
                    "other_object_energy": round(energy_in(heatmap, other_mask), 6),
                    "top20_rationale_precision": round(
                        intersection / top_pixels if top_pixels else 0.0, 6
                    ),
                    "top20_rationale_recall": round(
                        intersection / rationale_pixels if rationale_pixels else 0.0, 6
                    ),
                    "top20_rationale_iou": round(
                        intersection / union if union else 0.0, 6
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} CAM metric rows to {output_path}")
    for class_name in CLASS_NAMES:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        if not class_rows:
            continue
        print(
            f"{class_name}: object_energy={np.mean([r['object_energy'] for r in class_rows]):.3f}, "
            f"rationale_energy={np.mean([r['rationale_energy'] for r in class_rows]):.3f}, "
            f"other_object_energy={np.mean([r['other_object_energy'] for r in class_rows]):.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-dir", type=Path, default=CAM_DIR)
    parser.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    parser.add_argument("--rationale-dir", type=Path, default=RATIONALE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    evaluate(args.cam_dir, args.mask_dir, args.rationale_dir, args.output)
