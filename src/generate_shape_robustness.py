"""Create paired robustness variants from the held-out single-object test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def generate(source_dir: Path, output_dir: Path, seed: int) -> None:
    image_dir = source_dir / "images" / "test"
    label_dir = source_dir / "labels" / "test"
    mask_dir = source_dir / "masks" / "test"
    output_image_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for image_path in sorted(image_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(
            str(mask_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE
        )
        label_path = label_dir / f"{image_path.stem}.txt"
        if image is None or mask is None or not label_path.exists():
            raise FileNotFoundError(f"Missing source artifact for {image_path}")

        object_pixels = mask > 0
        background_pixels = ~object_pixels
        variants = {
            "color": image.copy(),
            "background": image.copy(),
            "color_background": image.copy(),
        }
        new_color = rng.integers(10, 120, size=3, dtype=np.uint8)
        new_background = rng.integers(185, 250, size=3, dtype=np.uint8)
        variants["color"][object_pixels] = new_color
        variants["background"][background_pixels] = new_background
        variants["color_background"][object_pixels] = new_color
        variants["color_background"][background_pixels] = new_background

        for variant_name, variant_image in variants.items():
            stem = f"{image_path.stem}_{variant_name}"
            cv2.imwrite(str(output_image_dir / f"{stem}.jpg"), variant_image)
            (output_label_dir / f"{stem}.txt").write_text(
                label_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            rows.append(
                {
                    "source": image_path.name,
                    "variant": variant_name,
                    "filename": f"{stem}.jpg",
                }
            )

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "variant", "filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} robustness images in {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/shapes_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/shapes_v2/robustness"))
    parser.add_argument("--seed", type=int, default=3303)
    args = parser.parse_args()
    generate(args.source_dir, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
