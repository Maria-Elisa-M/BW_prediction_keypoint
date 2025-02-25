import os
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

os.chdir(r'/home/lobatomeneze@ad.wisc.edu/Yolo/keypoint')

# Use the model
model.train(data="config.yaml", epochs=200, imgsz=644, project='stream3', batch=128)
