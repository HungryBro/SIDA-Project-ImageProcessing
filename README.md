# SIDA Project — Image Processing

โปรเจกต์ทดลอง Explainable AI สำหรับ YOLOv8 และ YOLO11 โดยใช้ Grad-CAM, Grad-CAM++,
Eigen-CAM และ Layer-CAM

## เริ่มต้นใช้งาน

```powershell
.\scripts\setup_env.ps1
python src\generate_shapes.py
python src\gradcam_yolo.py
```

Shape experiment commands after training:

```bash
python src/shape_gradcam.py
python src/evaluate_shape_cam.py
python src/evaluate_shape_detection.py
python src/evaluate_shape_faithfulness.py
```

Shape experiment v2:

```bash
bash scripts/run_shape_v2.sh
```

The v2 pipeline creates and validates `data/shapes_v2/`, trains
`outputs/shapes_v2/baseline_v2/weights/best.pt`, evaluates multi-object and
robustness performance, runs geometry counterfactuals, and generates a
separate Grad-CAM report. The current results are summarized in
[outputs/shapes_v2/SHAPE_REASONING_V2_REPORT.md](outputs/shapes_v2/SHAPE_REASONING_V2_REPORT.md).

โครงสร้างหลัก:

- `src/gradcam_yolo.py` — สคริปต์หลักสำหรับสร้าง heatmap
- `src/generate_shapes.py` — สร้างภาพพื้นฐาน 150 ภาพ และ multi-object test อีก 30 ภาพ
- `requirements.txt` — รายการ dependencies
- `data/shapes_v2/` — dataset v2 วงกลม สามเหลี่ยม และสี่เหลี่ยม พร้อม labels และ masks
- `data/shapes_v2/images/multi_test/` — ภาพที่มีหลายรูปทรงรวมกันสำหรับ stress test
- `models/` — น้ำหนักโมเดล YOLO
- `assets/input/` — ภาพ input สำหรับการทดลอง
- `outputs/shapes_v2/` — ผลการทดลอง Shape Reasoning v2 ทั้งหมด
- `outputs/archive/legacy_cam/` — ผล CAM จากการทดลองรุ่นเก่า
- `docs/` — เอกสารหลักและรายงานประกอบ
- `docs/reports/` — รายงาน HTML และสไลด์รุ่นเก่า

รายละเอียดทฤษฎีและผลการทดลองอยู่ที่ [docs/README_Week1.md](docs/README_Week1.md)
และ [docs/reports/GradCAM_Report.html](docs/reports/GradCAM_Report.html)
ผลการทดลองรูปทรงอยู่ที่ [outputs/shapes_v2/SHAPE_REASONING_V2_REPORT.md](outputs/shapes_v2/SHAPE_REASONING_V2_REPORT.md)
