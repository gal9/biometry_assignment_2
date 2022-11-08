from src.metrics import yolo_iou
import torch

image_codes = ["0501", "0502", "0503", "0504", "0505", "0506", "0507", "0508"]

model = torch.hub.load('./yolov5/yolov5', 'custom', path='./yolo5s.pt', source="local")

for image_code in image_codes:

    img_path = f"data/test/{image_code}.png"

    print(yolo_iou(image_code, model))
