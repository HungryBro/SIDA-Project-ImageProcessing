"""Sweep confidence/NMS settings on validation data and select a config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_shape_detection import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/shapes_v2/baseline_v2/weights/best.pt"))
    parser.add_argument("--images", type=Path, default=Path("data/shapes_v2/images/val"))
    parser.add_argument("--labels", type=Path, default=Path("data/shapes_v2/labels/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shapes_v2/baseline_v2/calibration"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for confidence in (0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        for nms_iou in (0.50, 0.70):
            output_path = args.output_dir / f"tmp_conf_{confidence:.2f}_iou_{nms_iou:.2f}.json"
            result = evaluate(
                args.model,
                args.images,
                args.labels,
                output_path,
                conf=confidence,
                nms_iou=nms_iou,
                match_iou=0.5,
                device=args.device,
            )
            overall = result["overall"]
            errors = result["error_counts"]
            rows.append(
                {
                    "confidence": confidence,
                    "nms_iou": nms_iou,
                    "precision": overall["precision"],
                    "recall": overall["recall"],
                    "exact_count_accuracy": overall["exact_count_accuracy"],
                    "duplicate_count": errors["duplicate_same_class"] + errors["duplicate_cross_class"],
                    "wrong_class": errors["wrong_class"],
                    "missed_detection": errors["missed_detection"],
                }
            )

    # Maximize exact-count accuracy, then precision, then recall.  Ties prefer
    # the lower confidence threshold to avoid hiding hard true positives.
    selected = sorted(
        rows,
        key=lambda row: (
            row["exact_count_accuracy"],
            row["precision"],
            row["recall"],
            -row["confidence"],
        ),
        reverse=True,
    )[0]
    csv_path = args.output_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_path = args.output_dir / "selected_config.json"
    selected_path.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "dataset": str(args.images),
                "confidence": selected["confidence"],
                "nms_iou": selected["nms_iou"],
                "device": args.device,
                "selection_split": "validation",
                "selection_metrics": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "output": str(selected_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
