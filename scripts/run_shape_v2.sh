#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source .venv/bin/activate

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sida-mplconfig}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/sida-pycache}"

if [[ ! -f data/shapes_v2/generation_config.json ]]; then
  python src/generate_shapes.py --output-dir data/shapes_v2
else
  echo "Reusing existing validated data/shapes_v2"
fi
python src/validate_shape_dataset.py --dataset-dir data/shapes_v2

yolo detect train \
  model=models/yolo11n.pt \
  data=data/shapes_v2/data.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  workers=0 \
  device=cpu \
  seed=42 \
  project="$ROOT_DIR/outputs/shapes" \
  name=baseline_v2 \
  exist_ok=True

python src/evaluate_shape_detection.py \
  --model outputs/shapes/baseline_v2/weights/best.pt \
  --images data/shapes_v2/images/multi_test \
  --labels data/shapes_v2/labels/multi_test \
  --output outputs/shapes/baseline_v2/evaluation/v2_multi_test_metrics.json

python src/generate_shape_robustness.py
python src/evaluate_shape_detection.py \
  --model outputs/shapes/baseline_v2/weights/best.pt \
  --images data/shapes_v2/robustness/images \
  --labels data/shapes_v2/robustness/labels \
  --output outputs/shapes/baseline_v2/evaluation/robustness_metrics.json

python src/generate_shape_counterfactuals.py
python src/evaluate_shape_counterfactuals.py

python src/shape_gradcam.py \
  --model outputs/shapes/baseline_v2/weights/best.pt \
  --images data/shapes_v2/locked_original/images \
  --labels data/shapes_v2/locked_original/labels \
  --output-dir outputs/shapes/baseline_v2/cam/gradcam

python src/evaluate_shape_cam.py \
  --cam-dir outputs/shapes/baseline_v2/cam/gradcam \
  --mask-dir data/shapes_v2/locked_original/masks \
  --rationale-dir data/shapes_v2/locked_original/rationale_masks \
  --output outputs/shapes/baseline_v2/evaluation/cam_metrics.csv

python src/evaluate_shape_faithfulness.py \
  --model outputs/shapes/baseline_v2/weights/best.pt \
  --images data/shapes_v2/locked_original/images \
  --labels data/shapes_v2/locked_original/labels \
  --cam-dir outputs/shapes/baseline_v2/cam/gradcam/raw_heatmaps \
  --mask-dir data/shapes_v2/locked_original/masks \
  --output outputs/shapes/baseline_v2/evaluation/faithfulness_metrics.csv
