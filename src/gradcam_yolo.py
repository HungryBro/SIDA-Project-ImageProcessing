import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Import pytorch-grad-cam components
from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus,
                              HiResCAM, LayerCAM, XGradCAM)
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "assets" / "input"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Copy of standard letterbox function for YOLO input resizing
def letterbox(
    im: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return im, ratio, (dw, dh)

# Custom ActivationsAndGradients for registering hooks
class ActivationsAndGradients:
    """ Class for extracting activations and registering gradients from targeted intermediate layers """
    def __init__(self, model: torch.nn.Module, target_layers, reshape_transform) -> None:
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output) -> None:
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output) -> None:
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

# Target module to compute custom score for gradients backpropagation
class yolov8_target(torch.nn.Module):
    def __init__(
        self, output_type, conf, ratio, target_class_ids=None, target_box=None
    ) -> None:
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio
        self.target_class_ids = target_class_ids  # List of allowed class IDs, or None for all
        self.target_box = target_box

    @staticmethod
    def box_iou_xywh(box, target_box):
        box_x1 = box[0] - box[2] / 2
        box_y1 = box[1] - box[3] / 2
        box_x2 = box[0] + box[2] / 2
        box_y2 = box[1] + box[3] / 2
        target_x1, target_y1, target_x2, target_y2 = target_box
        intersection = torch.clamp(
            torch.minimum(box_x2, target_x2) - torch.maximum(box_x1, target_x1),
            min=0,
        ) * torch.clamp(
            torch.minimum(box_y2, target_y2) - torch.maximum(box_y1, target_y1),
            min=0,
        )
        box_area = torch.clamp(box_x2 - box_x1, min=0) * torch.clamp(
            box_y2 - box_y1, min=0
        )
        target_area = torch.clamp(target_x2 - target_x1, min=0) * torch.clamp(
            target_y2 - target_y1, min=0
        )
        union = box_area + target_area - intersection
        return intersection / (union + 1e-6)

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        candidates = []
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max().detach()) < self.conf:
                break
            
            class_id = int(post_result[i].argmax())
            if self.target_class_ids is not None and class_id not in self.target_class_ids:
                continue

            if self.target_box is not None:
                candidates.append(
                    (self.box_iou_xywh(pre_post_boxes[i], self.target_box), i)
                )
                continue
                
            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())
            elif self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
                    
        if self.target_box is not None and candidates:
            _, selected_index = max(candidates, key=lambda item: float(item[0]))
            if self.output_type == "class" or self.output_type == "all":
                result.append(post_result[selected_index].max())
            if self.output_type == "box" or self.output_type == "all":
                for j in range(4):
                    result.append(pre_post_boxes[selected_index, j])

        if len(result) == 0:
            return torch.tensor(0.0, device=post_result.device, requires_grad=True)
            
        return sum(result)

# Main heatmap generator class
class yolov8_heatmap:
    def __init__(
        self,
        weight: str,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        method="EigenGradCAM",
        layer=[12, 17, 21],
        conf_threshold=0.2,
        ratio=0.02,
        show_box=True,
        renormalize=False,
        target_class_ids=None,
        target_box=None,
        target_output_type="all",
    ) -> None:
        self.device = device
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_box = show_box
        self.renormalize = renormalize
        self.target_class_ids = target_class_ids

        # Load YOLO model
        print(f"[*] Loading model weight: {weight} using ultralytics YOLO...")
        yolo_model = YOLO(weight)
        model = yolo_model.model
        model.to(device)
        
        self.model = model
        self.model_names = yolo_model.names
        
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.eval()

        self.target = yolov8_target(
            target_output_type, conf_threshold, ratio, target_class_ids, target_box
        )
        self.target_layers = [self.model.model[l] for l in layer]

        cam_class = globals()[method]
        self.method = cam_class(self.model, self.target_layers, use_cuda=device.type == 'cuda')
        self.method.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, None)

        self.colors = np.random.uniform(0, 255, size=(len(self.model_names), 3)).astype(int)

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.80)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = scale_cam_image(grayscale_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, return_details=False):
        img = cv2.imread(str(img_path))
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0

        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except Exception as e:
            print(f"[-] Error generating CAM: {e}")
            return None
            
        grayscale_cam = grayscale_cam[0, :]

        with torch.no_grad():
            pred1 = self.model(tensor)[0]
        pred = self.post_process(pred1)
        
        if self.renormalize:
            cam_image = self.renormalize_cam(
                pred[:, :4].cpu().detach().numpy().astype(np.int32), img, grayscale_cam
            )
        else:
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                conf = float(data[4])
                class_id = int(data[5])
                
                if self.target_class_ids is not None and class_id not in self.target_class_ids:
                    continue
                    
                cam_image = self.draw_detections(
                    data[:4],
                    self.colors[class_id],
                    f"{self.model_names[class_id]} {conf:.2f}",
                    cam_image,
                )

        output_image = Image.fromarray(cam_image)
        if return_details:
            return output_image, grayscale_cam, pred, img
        return output_image

    def __call__(self, img_path):
        if os.path.isdir(img_path):
            image_list = []
            for img_path_ in os.listdir(img_path):
                img_pil = self.process(os.path.join(img_path, img_path_))
                if img_pil is not None:
                    image_list.append(img_pil)
            return image_list
        else:
            res = self.process(img_path)
            return [res] if res is not None else []

def download_sample_image(url, save_path):
    """Downloads a sample image if it doesn't exist already."""
    if not os.path.exists(save_path):
        print(f"[*] Downloading sample image from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[+] Downloaded sample image to: {save_path}")
        except Exception as e:
            print(f"[-] Failed to download image: {e}")
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Sample Image", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(str(save_path), placeholder)
            print(f"[+] Created placeholder image at: {save_path}")

def run_gradcam_experiment():
    output_dir = OUTPUT_DIR / "standard"
    comparison_dir = OUTPUT_DIR / "comparison"
    
    # Create subdirectories for YOLOv8 and YOLO11
    for d in [output_dir, comparison_dir]:
        os.makedirs(os.path.join(d, "yolov8"), exist_ok=True)
        os.makedirs(os.path.join(d, "yolo11"), exist_ok=True)
    
    # Download sample images
    img_a_path = INPUT_DIR / "sample_cat_dog.jpg"
    img_a_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
    download_sample_image(img_a_url, img_a_path)
    
    img_b_path = INPUT_DIR / "test_bus.jpg"
    img_b_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
    download_sample_image(img_b_url, img_b_path)
    
    # CAM methods
    cam_methods = {
        "GradCAM": "GradCAM",
        "GradCAMPlusPlus": "GradCAMPlusPlus",
        "EigenCAM": "EigenCAM",
        "LayerCAM": "LayerCAM"
    }
    
    # Define models to compare
    models_to_test = {
        "yolov8": {
            "weight": MODEL_DIR / "yolov8n.pt",
            "layers": [12, 15, 16, 18, 21]
        },
        "yolo11": {
            "weight": MODEL_DIR / "yolo11n.pt",
            "layers": [13, 16, 19, 22]
        }
    }
    
    print("\n[*] Initializing YOLO Grad-CAM Folder-Separated Experiment...")
    
    for model_key, model_info in models_to_test.items():
        print(f"\n==========================================")
        print(f"[*] Processing Model: {model_key.upper()}")
        print(f"==========================================")
        
        weight_file = str(model_info["weight"])
        target_layers = model_info["layers"]
        
        # 1. RUN STANDARD EXPERIMENT (outputs/standard/yolov8/ or outputs/standard/yolo11/)
        # Targets all classes on sample_cat_dog.jpg
        print(f"\n--- [Standard Run] Target: All Classes (sample_cat_dog.jpg) ---")
        for label, method_class_name in cam_methods.items():
            print(f"[*] Generating standard heatmap using {label}...")
            try:
                model = yolov8_heatmap(
                    weight=weight_file,
                    method=method_class_name,
                    layer=target_layers,
                    conf_threshold=0.25,
                    show_box=True,
                    target_class_ids=None,
                )
                imagelist = model(img_path=img_a_path)
                if imagelist and len(imagelist) > 0:
                    out_filename = os.path.join(output_dir, model_key, f"{label.lower()}_result_0.jpg")
                    imagelist[0].save(out_filename)
                    print(f"    Saved: {out_filename}")
            except Exception as e:
                print(f"[-] Error: {e}")
        
        # 2. RUN COMPARISON EVALUATION (outputs/comparison/yolov8/ or outputs/comparison/yolo11/)
        # Test Case A: Dog Only (shows EigenCAM weakness)
        print(f"\n--- [Evaluation A] Class-Specific: Dog Only (sample_cat_dog.jpg) ---")
        for label, method_class_name in cam_methods.items():
            print(f"[*] Generating dog-only heatmap using {label}...")
            try:
                model = yolov8_heatmap(
                    weight=weight_file,
                    method=method_class_name,
                    layer=target_layers,
                    conf_threshold=0.25,
                    show_box=True,
                    target_class_ids=[16],  # Dog class only
                )
                imagelist = model(img_path=img_a_path)
                if imagelist and len(imagelist) > 0:
                    out_filename = os.path.join(comparison_dir, model_key, f"dog_only_{label.lower()}.jpg")
                    imagelist[0].save(out_filename)
                    print(f"    Saved: {out_filename}")
            except Exception as e:
                print(f"[-] Error: {e}")
                
        # Test Case B: Multi-Object (shows Grad-CAM weakness)
        print(f"\n--- [Evaluation B] Multi-Object: All Classes (test_bus.jpg) ---")
        for label, method_class_name in cam_methods.items():
            print(f"[*] Generating multi-object heatmap using {label}...")
            try:
                model = yolov8_heatmap(
                    weight=weight_file,
                    method=method_class_name,
                    layer=target_layers,
                    conf_threshold=0.25,
                    show_box=True,
                    target_class_ids=None,
                )
                imagelist = model(img_path=img_b_path)
                if imagelist and len(imagelist) > 0:
                    out_filename = os.path.join(comparison_dir, model_key, f"multi_object_{label.lower()}.jpg")
                    imagelist[0].save(out_filename)
                    print(f"    Saved: {out_filename}")
            except Exception as e:
                print(f"[-] Error: {e}")

    print("\n[+] Folder-separated experiment completed successfully!")
    print(f"[+] Outputs saved to '{output_dir}' and '{comparison_dir}' subfolders.")

if __name__ == "__main__":
    run_gradcam_experiment()
