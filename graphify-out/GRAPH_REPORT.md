# Graph Report - src  (2026-07-15)

## Corpus Check
- Corpus is ~7,665 words - fits in a single context window. You may not need a graph.

## Summary
- 140 nodes · 301 edges · 10 communities
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]
- [[_COMMUNITY_Code Pipeline|Code Pipeline]]

## God Nodes (most connected - your core abstractions)
1. `yolov8_heatmap` - 14 edges
2. `make_shape()` - 13 edges
3. `evaluate()` - 12 edges
4. `int` - 12 edges
5. `str` - 11 edges
6. `make_scene()` - 11 edges
7. `write_scene()` - 11 edges
8. `validate()` - 10 edges
9. `evaluate()` - 9 edges
10. `Path` - 9 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `YOLO`  [INFERRED]
  evaluate_shape_faithfulness.py → evaluate_shape_counterfactuals.py
- `evaluate()` --calls--> `YOLO`  [INFERRED]
  evaluate_shape_detection.py → evaluate_shape_counterfactuals.py
- `evaluate()` --calls--> `bool`  [INFERRED]
  evaluate_shape_counterfactuals.py → generate_shapes.py
- `_scene_overlaps()` --calls--> `Any`  [INFERRED]
  generate_shapes.py → evaluate_shape_detection.py
- `write_dataset()` --calls--> `Any`  [INFERRED]
  generate_shapes.py → evaluate_shape_detection.py

## Communities (10 total, 0 thin omitted)

### Community 0 - "Code Pipeline"
Cohesion: 0.25
Nodes (27): bool, float, int, ndarray, object, Path, str, Namespace (+19 more)

### Community 1 - "Code Pipeline"
Cohesion: 0.15
Nodes (18): ndarray, str, float, int, ndarray, Path, box_iou_xywh(), download_sample_image() (+10 more)

### Community 2 - "Code Pipeline"
Cohesion: 0.23
Nodes (14): int, float, int, ndarray, Path, str, main(), Sweep confidence/NMS settings on validation data and select a config. (+6 more)

### Community 3 - "Code Pipeline"
Cohesion: 0.36
Nodes (11): float, int, ndarray, Path, str, generate(), main(), polygon() (+3 more)

### Community 4 - "Code Pipeline"
Cohesion: 0.35
Nodes (11): Any, image_digest(), main(), parse_label(), Validate a generated YOLO shape dataset before training.  The validator treats m, validate(), float, int (+3 more)

### Community 5 - "Code Pipeline"
Cohesion: 0.38
Nodes (10): float, int, ndarray, Path, confidence_for_target(), iou(), masked_image(), Run a simple deletion test: CAM-selected pixels vs random pixels. (+2 more)

### Community 6 - "Code Pipeline"
Cohesion: 0.40
Nodes (9): float, int, Path, str, evaluate(), main(), Evaluate expected confidence changes on paired shape counterfactuals., target_confidence() (+1 more)

### Community 7 - "Code Pipeline"
Cohesion: 0.22
Nodes (3): Module, ActivationsAndGradients, Class for extracting activations and registering gradients from targeted interme

### Community 8 - "Code Pipeline"
Cohesion: 0.39
Nodes (7): float, ndarray, Path, energy_in(), evaluate(), Evaluate class-specific shape Grad-CAM against generated rationale masks., top_fraction_mask()

### Community 9 - "Code Pipeline"
Cohesion: 0.53
Nodes (5): int, Path, generate(), main(), Create paired robustness variants from the held-out single-object test set.

## Knowledge Gaps
- **7 isolated node(s):** `int`, `Namespace`, `float`, `Path`, `ndarray` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `YOLO` connect `Code Pipeline` to `Code Pipeline`, `Code Pipeline`, `Code Pipeline`?**
  _High betweenness centrality (0.392) - this node is a cross-community bridge._
- **Why does `evaluate()` connect `Code Pipeline` to `Code Pipeline`, `Code Pipeline`?**
  _High betweenness centrality (0.239) - this node is a cross-community bridge._
- **Why does `Any` connect `Code Pipeline` to `Code Pipeline`, `Code Pipeline`?**
  _High betweenness centrality (0.189) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `yolov8_heatmap` (e.g. with `Path` and `int`) actually correct?**
  _`yolov8_heatmap` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Evaluate expected confidence changes on paired shape counterfactuals.`, `Run a simple deletion test: CAM-selected pixels vs random pixels.`, `Create paired robustness variants from the held-out single-object test set.` to the rest of the system?**
  _21 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Code Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.14814814814814814 - nodes in this community are weakly interconnected._