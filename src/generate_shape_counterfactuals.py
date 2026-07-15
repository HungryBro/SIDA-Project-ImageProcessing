"""Generate paired geometry counterfactuals for the three shape classes."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


SHAPES = ("circle", "triangle", "square")


def polygon(center: tuple[int, int], radius: int, sides: int, angle: float = -90.0) -> np.ndarray:
    points = []
    for index in range(sides):
        theta = math.radians(angle + index * 360 / sides)
        points.append([center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)])
    return np.asarray(points, dtype=np.int32)


def render(shape: str, variant: str, image_size: int, color: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((image_size, image_size, 3), background, dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    center = (image_size // 2, image_size // 2)
    radius = 145
    if shape == "circle":
        if variant == "normal":
            cv2.circle(mask, center, radius, 255, -1)
        else:
            sides = {"polygon_8": 8, "polygon_6": 6, "polygon_4": 4}[variant]
            cv2.fillPoly(mask, [polygon(center, radius, sides)], 255)
    elif shape == "triangle":
        if variant == "normal":
            points = polygon(center, radius, 3)
        else:
            points = np.asarray(
                [[center[0] - radius, center[1] + radius // 2],
                 [center[0] + radius, center[1] + radius // 2],
                 [center[0] + radius // 3, center[1] - radius // 3]],
                dtype=np.int32,
            )
        cv2.fillPoly(mask, [points], 255)
    elif shape == "square":
        if variant == "normal":
            points = polygon(center, radius, 4, -45)
            cv2.fillPoly(mask, [points], 255)
        else:
            half = radius
            corner = 55
            x1, y1 = center[0] - half, center[1] - half
            x2, y2 = center[0] + half, center[1] + half
            cv2.rectangle(mask, (x1 + corner, y1), (x2 - corner, y2), 255, -1)
            cv2.rectangle(mask, (x1, y1 + corner), (x2, y2 - corner), 255, -1)
            for point in ((x1 + corner, y1 + corner), (x2 - corner, y1 + corner),
                          (x1 + corner, y2 - corner), (x2 - corner, y2 - corner)):
                cv2.circle(mask, point, corner, 255, -1)
    else:
        raise ValueError(shape)
    image[mask > 0] = color
    return image, mask


def yolo_label(class_id: int, mask: np.ndarray, image_size: int) -> str:
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = ((x_min + x_max) / 2) / image_size
    cy = ((y_min + y_max) / 2) / image_size
    width = (x_max - x_min + 1) / image_size
    height = (y_max - y_min + 1) / image_size
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"


def generate(output_dir: Path, pairs_per_case: int, image_size: int, seed: int) -> None:
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = [
        ("circle", "normal", "polygon_8", "decrease"),
        ("circle", "normal", "polygon_6", "decrease"),
        ("triangle", "normal", "clipped_apex", "decrease"),
        ("square", "normal", "rounded", "decrease"),
        ("circle", "normal", "normal", "stable"),
        ("triangle", "normal", "normal", "stable"),
        ("square", "normal", "normal", "stable"),
    ]
    rows = []
    for case_index, (shape, before_variant, after_variant, expected) in enumerate(cases):
        for index in range(pairs_per_case):
            color = rng.integers(15, 100, size=3, dtype=np.uint8)
            background = rng.integers(205, 246, size=3, dtype=np.uint8)
            before, before_mask = render(shape, before_variant, image_size, color, background)
            after, after_mask = render(shape, after_variant, image_size, color, background)
            pair_id = f"{shape}_{case_index:02d}_{index:03d}"
            before_name = f"{pair_id}_before.jpg"
            after_name = f"{pair_id}_after.jpg"
            cv2.imwrite(str(image_dir / before_name), before)
            cv2.imwrite(str(image_dir / after_name), after)
            class_id = SHAPES.index(shape)
            (label_dir / f"{before_name[:-4]}.txt").write_text(yolo_label(class_id, before_mask, image_size))
            (label_dir / f"{after_name[:-4]}.txt").write_text(yolo_label(class_id, after_mask, image_size))
            rows.append(
                {
                    "pair_id": pair_id,
                    "case_type": f"{shape}_{after_variant}",
                    "class_id": class_id,
                    "before": before_name,
                    "after": after_name,
                    "expected": expected,
                }
            )
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} counterfactual pairs in {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/shapes_v2/counterfactual"))
    parser.add_argument("--pairs-per-case", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=4404)
    args = parser.parse_args()
    generate(args.output_dir, args.pairs_per_case, args.image_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
