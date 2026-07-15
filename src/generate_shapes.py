"""Generate a controlled multi-object geometric-shape detection dataset.

The original experiment used only one object per image for training and kept
multi-object images as a stress test.  This generator creates a second,
isolated dataset version that includes single-object, multi-object, repeated
class, hard-case, empty-background, and locked multi-object splits.

Every object receives a YOLO label, object mask, rationale mask, boundary mask,
and metadata row.  The generator is deterministic for a given seed and never
creates the v2 dataset under ``data/shapes_v2`` unless another output directory
is explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SHAPES = ("circle", "triangle", "square")
CLASS_IDS = {name: index for index, name in enumerate(SHAPES)}
IMAGE_SIZE_DEFAULT = 640


def rotate_points(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    radians = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    return (points - center) @ rotation.T + center


def _safe_center(
    rng: np.random.Generator,
    image_size: int,
    extent: int,
    near_edge: bool = False,
) -> np.ndarray:
    margin = max(8, extent + 8)
    if near_edge:
        side = int(rng.integers(0, 4))
        if side == 0:
            x = int(rng.integers(margin, image_size - margin))
            y = int(rng.integers(margin, min(image_size - margin, margin + 50)))
        elif side == 1:
            x = int(rng.integers(margin, image_size - margin))
            y = int(rng.integers(max(margin, image_size - margin - 50), image_size - margin))
        elif side == 2:
            x = int(rng.integers(margin, min(image_size - margin, margin + 50)))
            y = int(rng.integers(margin, image_size - margin))
        else:
            x = int(rng.integers(max(margin, image_size - margin - 50), image_size - margin))
            y = int(rng.integers(margin, image_size - margin))
        return np.array([x, y], dtype=np.float32)
    return np.array(
        [
            int(rng.integers(margin, image_size - margin)),
            int(rng.integers(margin, image_size - margin)),
        ],
        dtype=np.float32,
    )


def _shape_points(shape_name: str, shape_size: int) -> np.ndarray:
    if shape_name == "square":
        half = shape_size / 2
        return np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float32,
        )
    if shape_name == "triangle":
        radius = shape_size * 0.58
        return np.array(
            [
                [0, -radius],
                [radius * np.sin(np.deg2rad(60)), radius * 0.5],
                [-radius * np.sin(np.deg2rad(60)), radius * 0.5],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported polygon shape: {shape_name}")


def make_shape(
    shape_name: str,
    rng: np.random.Generator,
    image_size: int,
    *,
    hard_case: bool = False,
    force_near_edge: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return image, object mask, rationale mask, boundary mask, metadata."""

    background = rng.integers(205, 246, size=3, dtype=np.uint8)
    if hard_case and rng.random() < 0.35:
        shape_color = np.clip(background.astype(int) - rng.integers(35, 70), 15, 120).astype(
            np.uint8
        )
    else:
        shape_color = rng.integers(20, 105, size=3, dtype=np.uint8)

    if hard_case:
        shape_size = int(rng.integers(105, 205))
        angle_choices = [0, 30, 45, 60, 90, 120, 150, 180, 225, 270, 315]
        angle = float(rng.choice(angle_choices))
    else:
        shape_size = int(rng.integers(150, 241))
        angle = float(rng.integers(0, 360))

    image = np.empty((image_size, image_size, 3), dtype=np.uint8)
    image[:] = background
    object_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    rationale_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    boundary_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    if shape_name == "circle":
        radius = shape_size // 2
        center = _safe_center(rng, image_size, radius, force_near_edge)
        center_xy = tuple(center.astype(int))
        cv2.circle(object_mask, center_xy, radius, 255, -1)
        cv2.circle(boundary_mask, center_xy, radius, 255, max(5, radius // 15))
        cv2.circle(rationale_mask, center_xy, radius, 255, max(8, radius // 8))
        geometry_points = []
        shape_extent = radius
    else:
        base_points = _shape_points(shape_name, shape_size)
        # Rotation can increase the axis-aligned extent beyond the unrotated
        # max(|x|, |y|), especially for a square at 45 degrees.  Use the
        # circumradius so the generated polygon remains inside the image.
        extent = int(np.ceil(np.max(np.linalg.norm(base_points, axis=1)))) + 25
        center = _safe_center(rng, image_size, extent, force_near_edge)
        points = rotate_points(base_points + center, center, angle).round().astype(int)
        cv2.fillPoly(object_mask, [points], 255)
        cv2.polylines(boundary_mask, [points], True, 255, max(5, shape_size // 35))
        corner_radius = max(9, int(shape_size * 0.08))
        for point in points:
            cv2.circle(rationale_mask, tuple(point), corner_radius, 255, -1)
        geometry_points = points.tolist()
        shape_extent = extent

    image[object_mask > 0] = shape_color
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError(f"Generated an empty mask for {shape_name}")

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    metadata: dict[str, object] = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "size": int(shape_size),
        "angle": float(angle),
        "geometry_points": json.dumps(geometry_points),
        "shape_extent": int(shape_extent),
        "background_b": int(background[0]),
        "background_g": int(background[1]),
        "background_r": int(background[2]),
        "shape_b": int(shape_color[0]),
        "shape_g": int(shape_color[1]),
        "shape_r": int(shape_color[2]),
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "hard_case": int(hard_case),
        "near_edge": int(force_near_edge),
    }
    return image, object_mask, rationale_mask, boundary_mask, metadata


def yolo_label(class_id: int, metadata: dict[str, object], image_size: int) -> str:
    x_min = int(metadata["x_min"])
    y_min = int(metadata["y_min"])
    x_max = int(metadata["x_max"])
    y_max = int(metadata["y_max"])
    center_x = ((x_min + x_max) / 2) / image_size
    center_y = ((y_min + y_max) / 2) / image_size
    width = (x_max - x_min + 1) / image_size
    height = (y_max - y_min + 1) / image_size
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def _scene_overlaps(candidate: np.ndarray, existing: Iterable[np.ndarray]) -> bool:
    return any(np.logical_and(candidate > 0, mask > 0).any() for mask in existing)


def make_scene(
    shape_names: list[str],
    rng: np.random.Generator,
    image_size: int,
    *,
    hard_case: bool = False,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]]:
    """Create one scene with non-overlapping objects and per-object artifacts."""

    background = rng.integers(205, 246, size=3, dtype=np.uint8)
    image = np.empty((image_size, image_size, 3), dtype=np.uint8)
    image[:] = background
    objects: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
    masks: list[np.ndarray] = []

    # Try larger objects first so a later small object does not make placement
    # impossible.  The class order is shuffled so position is not a class cue.
    order = list(shape_names)
    rng.shuffle(order)
    for shape_name in order:
        for _ in range(200):
            candidate = make_shape(
                shape_name,
                rng,
                image_size,
                # Multi-object scenes use a smaller size range so three
                # non-overlapping objects can fit reliably in a 640x640 frame.
                hard_case=hard_case or len(shape_names) > 1,
                force_near_edge=hard_case and rng.random() < 0.30,
            )
            if not _scene_overlaps(candidate[1], masks):
                break
        else:
            raise RuntimeError(f"Could not place non-overlapping {shape_name} scene")

        candidate_image, object_mask, rationale, boundary, metadata = candidate
        image[object_mask > 0] = candidate_image[object_mask > 0]
        masks.append(object_mask)
        objects.append((CLASS_IDS[shape_name], object_mask, rationale, boundary, metadata))

    return image, objects


def _ensure_dirs(output_dir: Path, split: str) -> None:
    for relative in (
        f"images/{split}",
        f"labels/{split}",
        f"masks/{split}",
        f"rationale_masks/{split}",
        f"boundary_masks/{split}",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def write_scene(
    output_dir: Path,
    split: str,
    stem: str,
    image: np.ndarray,
    objects: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]],
    image_size: int,
    case_type: str,
) -> list[dict[str, object]]:
    _ensure_dirs(output_dir, split)
    image_path = output_dir / "images" / split / f"{stem}.jpg"
    label_path = output_dir / "labels" / split / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), image):
        raise IOError(f"Could not write image: {image_path}")

    labels: list[str] = []
    rows: list[dict[str, object]] = []
    combined_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    combined_rationale = np.zeros((image_size, image_size), dtype=np.uint8)
    combined_boundary = np.zeros((image_size, image_size), dtype=np.uint8)

    for object_index, (class_id, object_mask, rationale, boundary, metadata) in enumerate(objects):
        shape_name = SHAPES[class_id]
        labels.append(yolo_label(class_id, metadata, image_size))
        combined_mask = np.maximum(combined_mask, object_mask)
        combined_rationale = np.maximum(combined_rationale, rationale)
        combined_boundary = np.maximum(combined_boundary, boundary)
        cv2.imwrite(str(output_dir / "masks" / split / f"{stem}_{object_index}_{shape_name}.png"), object_mask)
        cv2.imwrite(
            str(output_dir / "rationale_masks" / split / f"{stem}_{object_index}_{shape_name}.png"),
            rationale,
        )
        cv2.imwrite(
            str(output_dir / "boundary_masks" / split / f"{stem}_{object_index}_{shape_name}.png"),
            boundary,
        )
        rows.append(
            {
                "split": split,
                "filename": image_path.relative_to(output_dir).as_posix(),
                "class_id": class_id,
                "class_name": shape_name,
                "object_index": object_index,
                "case_type": case_type,
                "image_size": image_size,
                **metadata,
            }
        )

    cv2.imwrite(str(output_dir / "masks" / split / f"{stem}.png"), combined_mask)
    cv2.imwrite(str(output_dir / "rationale_masks" / split / f"{stem}.png"), combined_rationale)
    cv2.imwrite(str(output_dir / "boundary_masks" / split / f"{stem}.png"), combined_boundary)
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return rows


def write_empty_scene(output_dir: Path, split: str, stem: str, image_size: int) -> None:
    _ensure_dirs(output_dir, split)
    background = np.full((image_size, image_size, 3), 225, dtype=np.uint8)
    stable_seed = sum((index + 1) * ord(char) for index, char in enumerate(f"{split}:{stem}"))
    noise = np.random.default_rng(stable_seed).integers(
        -4, 5, size=background.shape, dtype=np.int16
    )
    image = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image_path = output_dir / "images" / split / f"{stem}.jpg"
    cv2.imwrite(str(image_path), image)
    (output_dir / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
    for folder in ("masks", "rationale_masks", "boundary_masks"):
        cv2.imwrite(
            str(output_dir / folder / split / f"{stem}.png"),
            np.zeros((image_size, image_size), dtype=np.uint8),
        )


def write_single_split(
    output_dir: Path,
    split: str,
    per_class: int,
    image_size: int,
    rng: np.random.Generator,
    *,
    hard_case: bool = False,
    stem_prefix: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id, shape_name in enumerate(SHAPES):
        for index in range(per_class):
            image, object_mask, rationale, boundary, metadata = make_shape(
                shape_name,
                rng,
                image_size,
                hard_case=hard_case,
                force_near_edge=hard_case and index % 3 == 0,
            )
            rows.extend(
                write_scene(
                    output_dir,
                    split,
                    f"{stem_prefix + '_' if stem_prefix else ''}{shape_name}_{index:03d}",
                    image,
                    [(class_id, object_mask, rationale, boundary, metadata)],
                    image_size,
                    "hard_single" if hard_case else "single",
                )
            )
    return rows


def write_multi_split(
    output_dir: Path,
    split: str,
    count: int,
    image_size: int,
    rng: np.random.Generator,
    *,
    locked_style: bool = False,
) -> list[dict[str, object]]:
    combinations = [
        ["circle", "triangle"],
        ["circle", "square"],
        ["triangle", "square"],
        ["circle", "triangle", "square"],
    ]
    rows: list[dict[str, object]] = []
    for index in range(count):
        if locked_style:
            shape_names = ["circle", "triangle", "square"]
            case_type = "locked_multi"
        else:
            shape_names = combinations[index % len(combinations)]
            case_type = "multi_three" if len(shape_names) == 3 else "multi_two"
        image, objects = make_scene(shape_names, rng, image_size)
        rows.extend(
            write_scene(
                output_dir,
                split,
                f"multi_{index:03d}",
                image,
                objects,
                image_size,
                case_type,
            )
        )
    return rows


def write_repeated_split(
    output_dir: Path,
    split: str,
    count: int,
    image_size: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        shape_name = SHAPES[index % len(SHAPES)]
        image, objects = make_scene([shape_name, shape_name], rng, image_size, hard_case=True)
        rows.extend(
            write_scene(
                output_dir,
                split,
                f"repeated_{index:03d}",
                image,
                objects,
                image_size,
                "repeated_class",
            )
        )
    return rows


def write_dataset(
    output_dir: Path,
    train_single_per_class: int,
    val_single_per_class: int,
    test_single_per_class: int,
    train_multi_count: int,
    val_multi_count: int,
    test_multi_count: int,
    hard_count: int,
    empty_count: int,
    repeated_count: int,
    image_size: int,
    seed: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty dataset directory: {output_dir}. "
            "Choose a new output directory or remove it intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    rows.extend(write_single_split(output_dir, "train", train_single_per_class, image_size, rng))
    rows.extend(write_single_split(output_dir, "val", val_single_per_class, image_size, rng))
    rows.extend(write_single_split(output_dir, "test", test_single_per_class, image_size, rng))
    rows.extend(write_multi_split(output_dir, "train", train_multi_count, image_size, rng))
    rows.extend(write_multi_split(output_dir, "val", val_multi_count, image_size, rng))
    rows.extend(
        write_multi_split(output_dir, "multi_test", test_multi_count, image_size, rng, locked_style=True)
    )
    rows.extend(write_repeated_split(output_dir, "train", repeated_count, image_size, rng))
    rows.extend(
        write_single_split(
            output_dir,
            "train",
            hard_count // 3,
            image_size,
            rng,
            hard_case=True,
            stem_prefix="hard",
        )
    )
    rows.extend(write_empty_split(output_dir, "train", empty_count, image_size))
    rows.extend(write_empty_split(output_dir, "val", max(1, empty_count // 3), image_size))

    data_yaml = (
        "path: data/shapes_v2\n"
        "train:\n"
        "  - images/train\n"
        "val:\n"
        "  - images/val\n"
        "test: images/test\n"
        "multi_test: images/multi_test\n"
        "names:\n"
        "  0: circle\n"
        "  1: triangle\n"
        "  2: square\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    config = {
        "seed": seed,
        "image_size": image_size,
        "train_single_per_class": train_single_per_class,
        "val_single_per_class": val_single_per_class,
        "test_single_per_class": test_single_per_class,
        "train_multi_count": train_multi_count,
        "val_multi_count": val_multi_count,
        "test_multi_count": test_multi_count,
        "hard_count": hard_count,
        "empty_count": empty_count,
        "repeated_count": repeated_count,
        "classes": list(SHAPES),
    }
    (output_dir / "generation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        if rows:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / "README.md").write_text(
        "# Synthetic Shapes Dataset v2\n\n"
        "This dataset contains single-object and multi-object training data, "
        "hard cases, repeated classes, empty backgrounds, and a locked-style "
        "multi-object split. `masks/` stores object masks, `boundary_masks/` "
        "stores contours, and `rationale_masks/` stores expected diagnostic "
        "regions: circle boundary or polygon corners.\n",
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} object annotations in {output_dir}")
    print(json.dumps(config, indent=2))


def write_empty_split(output_dir: Path, split: str, count: int, image_size: int) -> list[dict[str, object]]:
    for index in range(count):
        write_empty_scene(output_dir, split, f"empty_{index:03d}", image_size)
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/shapes_v2"))
    parser.add_argument("--train-single-per-class", type=int, default=60)
    parser.add_argument("--val-single-per-class", type=int, default=10)
    parser.add_argument("--test-single-per-class", type=int, default=10)
    parser.add_argument("--train-multi-count", type=int, default=60)
    parser.add_argument("--val-multi-count", type=int, default=15)
    parser.add_argument("--test-multi-count", type=int, default=30)
    parser.add_argument("--hard-count", type=int, default=30)
    parser.add_argument("--empty-count", type=int, default=10)
    parser.add_argument("--repeated-count", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=1101)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_dataset(
        output_dir=args.output_dir,
        train_single_per_class=args.train_single_per_class,
        val_single_per_class=args.val_single_per_class,
        test_single_per_class=args.test_single_per_class,
        train_multi_count=args.train_multi_count,
        val_multi_count=args.val_multi_count,
        test_multi_count=args.test_multi_count,
        hard_count=args.hard_count,
        empty_count=args.empty_count,
        repeated_count=args.repeated_count,
        image_size=args.image_size,
        seed=args.seed,
    )
