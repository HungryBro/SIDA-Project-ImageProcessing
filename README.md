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
`outputs/shapes/baseline_v2/weights/best.pt`, evaluates multi-object and
robustness performance, runs geometry counterfactuals, and generates a
separate Grad-CAM report. The current results are summarized in
[outputs/shapes/SHAPE_REASONING_V2_REPORT.md](outputs/shapes/SHAPE_REASONING_V2_REPORT.md).

โครงสร้างหลัก:

- `src/gradcam_yolo.py` — สคริปต์หลักสำหรับสร้าง heatmap
- `src/generate_shapes.py` — สร้างภาพพื้นฐาน 150 ภาพ และ multi-object test อีก 30 ภาพ
- `requirements.txt` — รายการ dependencies
- `data/shapes_v2/` — dataset v2 วงกลม สามเหลี่ยม และสี่เหลี่ยม พร้อม labels และ masks
- `data/shapes_v2/images/multi_test/` — ภาพที่มีหลายรูปทรงรวมกันสำหรับ stress test
- `models/` — น้ำหนักโมเดล YOLO
- `assets/input/` — ภาพ input สำหรับการทดลอง
- `outputs/standard/` — ผลลัพธ์การรันปกติ
- `outputs/comparison/` — ผลลัพธ์สำหรับเปรียบเทียบวิธี CAM
- `docs/` — รายงาน สไลด์ และเอกสารประกอบ

รายละเอียดทฤษฎีและผลการทดลองอยู่ที่ [docs/README_Week1.md](docs/README_Week1.md)
และ [docs/GradCAM_Report.html](docs/GradCAM_Report.html)
ผลการทดลองรูปทรงอยู่ที่ [outputs/shapes/SHAPE_REASONING_V2_REPORT.md](outputs/shapes/SHAPE_REASONING_V2_REPORT.md)
