"""Evaluate expected confidence changes on paired shape counterfactuals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


def target_confidence(model: YOLO, image_path: Path, class_id: int, device: str) -> float:
    result = model.predict(source=str(image_path), conf=0.0, device=device, verbose=False)[0]
    if result.boxes is None:
        return 0.0
    scores = [
        float(confidence)
        for predicted_class, confidence in zip(result.boxes.cls, result.boxes.conf)
        if int(predicted_class) == class_id
    ]
    return max(scores, default=0.0)


def evaluate(metadata_path: Path, image_dir: Path, model_path: Path, output_path: Path, device: str) -> dict:
    model = YOLO(str(model_path))
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    results = []
    for row in rows:
        class_id = int(row["class_id"])
        before = target_confidence(model, image_dir / row["before"], class_id, device)
        after = target_confidence(model, image_dir / row["after"], class_id, device)
        delta = after - before
        relation = row["expected"]
        if relation == "stable":
            passed = abs(delta) <= 0.20
        else:
            passed = delta <= -0.05
        results.append(
            {
                **row,
                "before_confidence": round(before, 6),
                "after_confidence": round(after, 6),
                "delta": round(delta, 6),
                "passed": passed,
            }
        )

    grouped = defaultdict(list)
    for row in results:
        grouped[row["case_type"]].append(row)
    summary = {
        case_type: {
            "pairs": len(case_rows),
            "passed": sum(bool(row["passed"]) for row in case_rows),
            "pass_rate": round(
                sum(bool(row["passed"]) for row in case_rows) / len(case_rows), 6
            ),
            "mean_delta": round(
                sum(float(row["delta"]) for row in case_rows) / len(case_rows), 6
            ),
        }
        for case_type, case_rows in grouped.items()
    }
    output = {
        "model": str(model_path),
        "summary": summary,
        "overall_pass_rate": round(
            sum(bool(row["passed"]) for row in results) / len(results), 6
        )
        if results
        else 0.0,
        "pairs": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/shapes_v2/counterfactual/metadata.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/shapes_v2/counterfactual/images"))
    parser.add_argument("--model", type=Path, default=Path("outputs/shapes/baseline_v2/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shapes/baseline_v2/evaluation/counterfactual_metrics.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = evaluate(args.metadata, args.image_dir, args.model, args.output, args.device)
    print(json.dumps({"summary": output["summary"], "overall_pass_rate": output["overall_pass_rate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
