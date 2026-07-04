# 📘 คู่มือการศึกษาและทดลอง YOLO Explainable AI (XAI) - Week 1

ยินดีต้อนรับสู่ **Week 1** ของการศึกษาการทำงานภายในของ Deep Learning Model สำหรับการตรวจจับวัตถุ (Object Detection) โดยเน้นไปที่ตระกูล **YOLO (You Only Look Once)** เช่น YOLOv8 หรือ YOLOv11 และการใช้เทคนิค **Grad-CAM (Gradient-weighted Class Activation Mapping)** และ CAM รูปแบบอื่นๆ

---

## 🧠 1. ทำความเข้าใจ Explainable AI (XAI) สำหรับ YOLO

ปกติแล้ว Deep Learning เปรียบเสมือน **"กล่องดำ" (Black Box)** ที่รับภาพเข้าไปแล้วพ่นกรอบทำนายวัตถุออกมา XAI จะช่วย **"เปิดกล่องดำ"** นี้เพื่อให้เราเห็นว่า โมเดลใช้ส่วนไหนของภาพในการตัดสินใจว่ามีวัตถุนั้นอยู่จริง

ในโมเดลตรวจจับวัตถุ เช่น YOLO การทำ XAI จะมีความซับซ้อนกว่างานจำแนกภาพ (Image Classification) ทั่วไป เนื่องจาก:
1. **Multi-task Output:** YOLO ทำนายทั้งประเภทวัตถุ (Classification) และ พิกัดกล่อง (Bounding Box Regression) ในเวลาเดียวกัน
2. **Anchor-based/Anchor-free Grid:** โมเดลประเมินค่านับพันตำแหน่งบนภาพ ทำให้การดึงค่า Gradient ย้อนกลับ (Backpropagation) ต้องทำผ่านกลไก Wrapper เพื่อเจาะจงเฉพาะกล่องทำนายที่เราสนใจ

### อัลกอริทึมที่สำคัญในการทดลองนี้:
*   **Grad-CAM:** ใช้ผลคูณระหว่าง Feature Map (ค่าคุณลักษณะในเลเยอร์ที่คัดเลือก) กับค่าเฉลี่ยสเปเชียลของเกรเดียนต์ เพื่อแสดงจุดที่โมเดลสนใจมากที่สุดสำหรับวัตถุคลาสนั้นๆ
*   **Grad-CAM++:** พัฒนาต่อยอดจาก Grad-CAM โดยคำนวณเกรเดียนต์อันดับสอง (Second-order gradients) ช่วยให้แสดงพื้นที่การตอบสนองได้สมบูรณ์ขึ้นเมื่อมีวัตถุประเภทเดียวกันปรากฏขึ้นหลายชิ้นในภาพเดียว (Multiple Instances)
*   **Eigen-CAM:** เป็นวิธีแบบไร้เกรเดียนต์ (Gradient-free) โดยการหา Principal Component (PCA) ชิ้นแรกของเลเยอร์ฟีเจอร์ เหมาะสำหรับการดูโครงสร้างทางกายภาพที่โมเดลดึงออกมาโดยไม่ต้องอ้างอิงประเภทคลาส (Class-agnostic) และทำงานได้รวดเร็วมาก

---

## 🛠️ 2. ขั้นตอนการตั้งค่าสภาพแวดล้อมเพื่อทดลองรันโค้ดจริง

เพื่อรันโค้ด Python ในการดึงค่าจากโมเดล YOLO แนะนำให้ตั้งค่าตามขั้นตอนด้านล่างนี้:

### ขั้นตอนที่ 1: ตรวจสอบและติดตั้ง Python
หากเครื่องคอมพิวเตอร์ของคุณยังไม่มี Python สามารถดาวน์โหลดและติดตั้งได้จาก [Python.org](https://www.python.org/downloads/) (แนะนำเวอร์ชัน 3.10 หรือ 3.11) *อย่าลืมติ๊กเลือก "Add Python to PATH" ตอนติดตั้ง*

### ขั้นตอนที่ 2: สร้างสภาพแวดล้อมเสมือน (Virtual Environment)
เปิด PowerShell/Terminal แล้วย้ายเข้าไปในไดเรกทอรีนี้ จากนั้นรันคำสั่ง:
```bash
# 1. ย้ายเข้าโฟลเดอร์ Week1
cd c:\Users\Acer\Desktop\Project\Week1

# 2. สร้าง Virtual Environment ชื่อ venv
python -m venv venv

# 3. เปิดใช้งาน (Activate) Virtual Environment
# สำหรับ Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# (หากพบ Permission Error ให้รัน Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process ก่อน)
```

### ขั้นตอนที่ 3: ติดตั้งไลบรารีที่จำเป็น
เมื่ออยู่ในสถานะ `(venv)` แล้ว ให้รันคำสั่งติดตั้งตัวแปรหลัก:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics grad-cam opencv-python matplotlib
```
*(หมายเหตุ: หากเครื่องมีการ์ดจอ NVIDIA และต้องการใช้ GPU ให้ติดตั้ง PyTorch รุ่น CUDA)*

---

## 🚀 3. ไฟล์การทดลองและวิธีการใช้งาน

ในไดเรกทอรีนี้ประกอบด้วย 3 รูปแบบการทดลองหลัก:

### 🌐 รูปแบบที่ A: สนามทดลองอินเตอร์แอคทีฟ (ไม่ต้องใช้ Python)
*   **ไฟล์:** [yolo_xai.html](file:///c:/Users/Acer/Desktop/Project/Week1/yolo_xai.html)
*   **วิธีใช้:** เพียงแค่ดับเบิลคลิกไฟล์นี้เพื่อเปิดบน Web Browser (Chrome, Edge)
*   **สิ่งที่ทำได้:** 
    *   ทดลองเปลี่ยน XAI Algorithm (Grad-CAM, Grad-CAM++, Eigen-CAM)
    *   ทดลองเปลี่ยน Target Layer เพื่อดูระดับการประมวลผล (Backbone ละเอียดน้อย-โครงสร้างหลัก, Head ละเอียดสูง-การจำแนกประเภท)
    *   คลิกที่แต่ละ Bounding Box เพื่อคำนวณ Heatmap ของวัตถุชิ้นนั้นๆ แบบ Real-time
    *   ลากวาดกรอบตรวจจับเองเพื่อดูบริเวณรอบขอบเขต
    *   ดูแชนเนลโครงข่ายย่อย (Feature Channels 0-15) ว่ามองเห็นรูปภาพเป็นอย่างไร

### 🐍 รูปแบบที่ B: โค้ดสคริปต์ Python สำหรับรันจริง
*   **ไฟล์:** [yolo_xai_demo.py](file:///c:/Users/Acer/Desktop/Project/Week1/yolo_xai_demo.py)
*   **วิธีใช้:** รันด้วยคำสั่ง:
    ```bash
    python yolo_xai_demo.py
    ```
*   **ผลลัพธ์:** สคริปต์จะใช้โมเดล YOLOv8n และสร้างแผนภาพความร้อน (Heatmap) ทับลงบนรูปภาพตัวอย่าง `seaweed_insect_detection.png` และเซฟออกมาเป็นภาพเปรียบเทียบในไดเรกทอรีนี้

### 📓 รูปแบบที่ C: สมุดโน้ตอธิบายทีละขั้นตอน (Interactive Notebook)
*   **ไฟล์:** [yolo_xai_demo.ipynb](file:///c:/Users/Acer/Desktop/Project/Week1/yolo_xai_demo.ipynb)
*   **วิธีใช้:** เปิดไฟล์นี้ใน VS Code (ที่ติดตั้ง Extension Jupyter แล้ว) หรือผ่าน `jupyter notebook`
*   **ประโยชน์:** เหมาะสำหรับเปิดอ่านทฤษฎีควบคู่ไปกับการรันภาพเพื่อดูผลลัพธ์การเรียนรู้ไปทีละบล็อก

---

## 📊 4. สรุปผลการเปรียบเทียบแต่ละเลเยอร์ (สำหรับรายงานอาจารย์)

เมื่อคุณทำการทดลองวิเคราะห์ผ่านเลเยอร์ต่างๆ ของ YOLO จะพบข้อสังเกตดังนี้:
1.  **Backbone (เช่น SPPF หรือ Conv เลเยอร์ลึกๆ):** Heatmap มักจะฟุ้งและครอบคลุมบริเวณกว้างรอบๆ วัตถุ เนื่องจากเป็นฟีเจอร์ระดับสูง (High-level semantic features) ที่เน้นบริบทและรูปร่างโดยรวม
2.  **Neck (เช่น C2f หรือ Concat ใน Path-Aggregation Network):** ความร้อนจะเริ่มบีบแคบลงตามพิกัดของวัตถุอย่างชัดเจน
3.  **Head - Class Branch (cv3):** ความร้อนพุ่งตรงไปยังจุดเด่นเฉพาะตัวที่ระบุชนิดวัตถุ (เช่น หัวแมลง, ใบหูสุนัข, ลวดลายบนตัว)
4.  **Head - Box Branch (cv2):** ความร้อนจะกระจายตัวอยู่ตามขอบเขตและมุมขอบของวัตถุ เพื่อช่วยในการทำนายขอบเขตพิกัดกล่อง (Bounding Box Regression)
