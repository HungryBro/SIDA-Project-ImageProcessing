import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ลองนำเข้า pytorch_grad_cam หากติดตั้งแล้ว หากยังไม่ติดตั้งสคริปต์จะแนะนำวิธีติดตั้ง
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("\n[!] Error: ไม่พบไลบรารี 'grad-cam' หรือ 'pytorch-grad-cam'")
    print("กรุณาติดตั้งด้วยคำสั่ง: pip install grad-cam\n")
    exit(1)

# ==========================================
# 1. นิยาม Wrapper สำหรับ YOLO
# wrapper นี้ช่วยปรับรูปแบบ outputs ของ YOLOv8/v11 ให้เข้ากับ pytorch-grad-cam
# ==========================================
class YOLOv8XAIWrapper(torch.nn.Module):
    def __init__(self, yolomodel):
        super().__init__()
        self.model = yolomodel.model

    def forward(self, x):
        # รันการทำนายผลของโมเดล PyTorch หลัก
        # YOLO จะคืนค่าผลลัพธ์เป็น Tuple/List โดย outputs[0] คือผลลัพธ์การทำนาย
        return self.model(x)

# ==========================================
# 2. นิยาม Class Target สำหรับคำนวณ Gradient
# ระบุว่าจะดึงคะแนนของ Bounding Box ชิ้นไหนและคลาสประเภทใด
# ==========================================
class YOLOBoxTarget:
    def __init__(self, target_class_id, target_box_idx):
        self.target_class_id = target_class_id
        self.target_box_idx = target_box_idx

    def __call__(self, outputs):
        # outputs[0] คือ Tensor ผลลัพธ์: [batch, dimensions, anchors]
        predictions = outputs[0]
        # ดึงคะแนนผลลัพธ์สำหรับ Bounding Box ที่คัดเลือก
        # สำหรับ YOLOv8: มิติแรกๆ คือพิกัดกล่อง มิติถัดไปคือคะแนนของแต่ละคลาส (class scores)
        # 4 + target_class_id คือตำแหน่งคลาสนั้นๆ ในมิติคะแนนทำนาย
        class_score = predictions[0, 4 + self.target_class_id, self.target_box_idx]
        return class_score

def main():
    print("=== เริ่มการวิเคราะห์ YOLO Explainable AI (XAI) ===")
    
    # กำหนดพาธรูปภาพตัวอย่าง
    # คาดหวังภาพจากโฟลเดอร์หลัก: ../images/seaweed_insect_detection.png
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "..", "images", "seaweed_insect_detection.png")
    
    if not os.path.exists(img_path):
        print(f"[!] Warning: ไม่พบภาพตัวอย่างที่ '{img_path}'")
        # สร้างภาพจำลองขึ้นมาแทนเพื่อให้รันโค้ดได้โดยไม่พัง
        print("-> กำลังสร้างรูปภาพจำลองสำหรับการทดสอบ...")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (0, 0, 255), -1) # วาดวงกลมแดงตรงกลางแทนแมลง
        cv2.putText(img, "Simulation Target", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        print(f"[+] โหลดภาพตัวอย่างสำเร็จ: {img_path}")
        img = cv2.imread(img_path)
    
    # ปรับขนาดภาพเข้าสู่โมเดล (640x640)
    img_resized = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = np.float32(rgb_img) / 255.0
    
    # แปลงภาพเป็น Tensor สำหรับ PyTorch [Batch=1, Channels=3, H=640, W=640]
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # 2. โหลดโมเดล YOLOv8n (หากไม่มี จะถูกดาวน์โหลดจาก ultralytics โดยอัตโนมัติ)
    print("-> กำลังโหลดโมเดล YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # รันการทำนายผลเพื่อหาดัชนีกล่องและคลาสในภาพต้นฉบับ
    results = model(img_resized)
    
    # ตรวจสอบว่าโมเดลทำนายพบวัตถุอะไรบ้าง
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("[!] Warning: โมเดลไม่พบวัตถุในภาพนี้ จะวิเคราะห์โดยอ้างอิงตำแหน่งกล่องสมมติ (กล่องที่ 0 คลาสที่ 0)")
        target_class_id = 0
        target_box_idx = 0
    else:
        print(f"[+] โมเดลตรวจพบวัตถุ {len(boxes)} ชิ้น:")
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            name = model.names[cls_id]
            print(f"  - ชิ้นที่ {idx}: {name} (Class ID: {cls_id}) ความมั่นใจ: {conf:.2f}")
        
        # เลือกวัตถุชิ้นแรกสุดที่ตรวจพบมาวิเคราะห์
        target_class_id = int(boxes[0].cls[0].item())
        target_box_idx = 0
        print(f"-> คัดเลือกวัตถุชิ้นแรกสำหรับการทำ XAI: คลาส ID {target_class_id} (ดัชนีกล่อง: {target_box_idx})")

    # 3. เตรียม Wrapped Model และตั้งค่า Target Layer
    wrapped_model = YOLOv8XAIWrapper(model)
    wrapped_model.eval()
    
    # เลเยอร์หลักที่เราสนใจวิเคราะห์:
    # 1) Backbone P5 (โมเดลลำดับเลเยอร์ที่ 9 ใน YOLOv8-nano)
    backbone_layer = model.model.model[9]
    # 2) Head Classification Head (สำหรับ Classifiers ใน YOLOv8)
    head_layer = model.model.model[22].cv3[0]
    
    targets = [YOLOBoxTarget(target_class_id=target_class_id, target_box_idx=target_box_idx)]
    
    # 4. วิเคราะห์ด้วย Grad-CAM บน Backbone Layer
    print("-> กำลังวิเคราะห์ Grad-CAM สำหรับ Backbone Layer [Layer 9]...")
    cam_backbone = GradCAM(model=wrapped_model, target_layers=[backbone_layer])
    grayscale_cam_backbone = cam_backbone(input_tensor=input_tensor, targets=targets)[0, :]
    vis_backbone = show_cam_on_image(img_normalized, grayscale_cam_backbone, use_rgb=True)
    
    # 5. วิเคราะห์ด้วย Grad-CAM บน Classification Head
    print("-> กำลังวิเคราะห์ Grad-CAM สำหรับ Classification Head [Layer 22]...")
    cam_head = GradCAM(model=wrapped_model, target_layers=[head_layer])
    grayscale_cam_head = cam_head(input_tensor=input_tensor, targets=targets)[0, :]
    vis_head = show_cam_on_image(img_normalized, grayscale_cam_head, use_rgb=True)
    
    # 6. วิเคราะห์ด้วย Eigen-CAM (Gradient-free PCA) เพื่อเปรียบเทียบ
    print("-> กำลังวิเคราะห์ Eigen-CAM สำหรับ Backbone Layer...")
    eigencam = EigenCAM(model=wrapped_model, target_layers=[backbone_layer])
    grayscale_eigen = eigencam(input_tensor=input_tensor, targets=targets)[0, :]
    vis_eigen = show_cam_on_image(img_normalized, grayscale_eigen, use_rgb=True)
    
    # ==========================================
    # 7. สร้างภาพเปรียบเทียบและบันทึก
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("1. Original Image (Resized)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(vis_backbone)
    axes[0, 1].set_title("2. Grad-CAM on Backbone (P5)")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(vis_head)
    axes[1, 0].set_title("3. Grad-CAM on Classification Head")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(vis_eigen)
    axes[1, 1].set_title("4. Eigen-CAM on Backbone (Class-Agnostic)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    # บันทึกภาพผลลัพธ์
    output_path = os.path.join(base_dir, "yolo_xai_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n[+] วิเคราะห์ผลและประมวลผลสำเร็จ!")
    print(f"[+] บันทึกภาพผลการเปรียบเทียบใน: {output_path}")
    print("คุณสามารถเปิดไฟล์ภาพ yolo_xai_comparison.png เพื่อเปรียบเทียบเลเยอร์และสถาปัตยกรรม XAI ได้ทันที")

if __name__ == "__main__":
    main()
