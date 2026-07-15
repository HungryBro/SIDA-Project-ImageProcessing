"""Validate a generated YOLO shape dataset before training.

The validator treats missing files, malformed labels, invalid boxes, and
cross-split duplicate images as blocking errors.  It writes a machine-readable
report and exits non-zero when the dataset is unsafe to train on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("circle", "triangle", "square")
IMAGE_SPLITS = ("train", "val", "test", "multi_test")


def image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_label(path: Path) -> tuple[list[dict[str, float | int]], list[str]]:
    errors: list[str] = []
    labels: list[dict[str, float | int]] = []
    text = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric label")
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            errors.append(f"{path}:{line_number}: invalid class id {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{path}:{line_number}: normalized coordinate outside [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{path}:{line_number}: non-positive box size")
        labels.append(
            {
                "class_id": class_id,
                "center_x": values[0],
                "center_y": values[1],
                "width": values[2],
                "height": values[3],
            }
        )
    return labels, errors


def validate(dataset_dir: Path, report_path: Path) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    label_counts: Counter[str] = Counter()

    for split in IMAGE_SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.exists():
            errors.append(f"Missing image split directory: {image_dir}")
            continue
        if not label_dir.exists():
            errors.append(f"Missing label split directory: {label_dir}")
            continue

        for image_path in sorted(image_dir.glob("*.jpg")):
            relative = image_path.relative_to(dataset_dir).as_posix()
            split_image_counts[split] += 1
            digest = image_digest(image_path)
            hashes[digest].append(relative)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"Unreadable image: {relative}")
                continue
            if image.shape[0] != image.shape[1]:
                errors.append(f"Non-square image: {relative} shape={image.shape[:2]}")

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Missing label: {label_path.relative_to(dataset_dir)}")
                continue
            labels, label_errors = parse_label(label_path)
            errors.extend(label_errors)
            if image_path.stem.startswith("empty_") and labels:
                errors.append(f"Empty image has labels: {relative}")
            if not image_path.stem.startswith("empty_") and not labels:
                errors.append(f"Non-empty image has no labels: {relative}")

            for label in labels:
                class_id = int(label["class_id"])
                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                    counts[f"{split}:{class_name}"] += 1
                    label_counts[class_name] += 1
                cx = float(label["center_x"])
                cy = float(label["center_y"])
                width = float(label["width"])
                height = float(label["height"])
                if cx - width / 2 < -1e-6 or cx + width / 2 > 1.000001:
                    errors.append(f"Box exceeds x bounds: {relative}")
                if cy - height / 2 < -1e-6 or cy + height / 2 > 1.000001:
                    errors.append(f"Box exceeds y bounds: {relative}")

            for folder in ("masks", "rationale_masks", "boundary_masks"):
                combined_mask = dataset_dir / folder / split / f"{image_path.stem}.png"
                if not combined_mask.exists():
                    errors.append(f"Missing {folder} file: {combined_mask.relative_to(dataset_dir)}")
                elif cv2.imread(str(combined_mask), cv2.IMREAD_GRAYSCALE) is None:
                    errors.append(f"Unreadable {folder} file: {combined_mask.relative_to(dataset_dir)}")

        image_stems = {path.stem for path in image_dir.glob("*.jpg")}
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        if missing_labels:
            errors.append(f"{split}: missing labels for {missing_labels[:5]}")
        if extra_labels:
            warnings.append(f"{split}: {len(extra_labels)} labels have no image")

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_duplicates = [
        paths
        for paths in duplicate_groups
        if len({path.split("/")[1] for path in paths}) > 1
    ]
    if cross_split_duplicates:
        errors.append(
            f"Cross-split duplicate image groups: {len(cross_split_duplicates)} "
            f"example={cross_split_duplicates[0]}"
        )

    if not any(split_image_counts.values()):
        errors.append("Dataset contains no images")

    report: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "image_counts": dict(split_image_counts),
        "label_counts": dict(label_counts),
        "split_class_counts": dict(counts),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_duplicates,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/shapes_v2"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/shapes_v2/dataset_audit/validation_report.json"),
    )
    args = parser.parse_args()
    report = validate(args.dataset_dir, args.report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
