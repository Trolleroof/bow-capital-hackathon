"""
Export YOLO model to TensorRT for Jetson Nano.
Run this ONCE on the Jetson after setup:

    python export_trt.py

Output: yolo11n.engine  (or whatever YOLO_MODEL is set to minus .pt)
"""
from ultralytics import YOLO
import config

model = YOLO(config.YOLO_MODEL)
model.export(format="engine", device=0, half=True)
print("[export] Done. Set YOLO_MODEL env var to the .engine path.")
