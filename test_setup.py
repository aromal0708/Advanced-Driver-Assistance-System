import torch
import cv2
import numpy as np

# Load YOLOv5 model from Torch Hub (automatically downloads if not available)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Check if model is loaded
if model:
    print("✅ YOLOv5 Model Loaded Successfully!")
else:
    print("❌ Error Loading YOLOv5 Model")

# Check OpenCV installation
print("OpenCV Version:", cv2.__version__)
