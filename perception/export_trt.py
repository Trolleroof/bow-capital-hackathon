"""
Export YOLO model to TensorRT for Jetson Nano.
Run this ONCE on the Jetson after setup:

    python export_trt.py

Reads YOLO_MODEL and YOLO_IMGSZ from config (.env).
Always loads the .pt source -- if YOLO_MODEL points at a .engine that
doesn't exist yet, the script finds the matching .pt automatically.

Output: <model_stem>.engine  (e.g. yolo11n.engine)
"""
import os
import sys
from pathlib import Path

import config
from ultralytics import YOLO

# Resolve source .pt -- handle the case where YOLO_MODEL is already .engine
model_path = Path(config.YOLO_MODEL)
if model_path.suffix == ".engine":
    pt_path = model_path.with_suffix(".pt")
else:
    pt_path = model_path

if not pt_path.exists():
    sys.exit(f"[export] Source model not found: {pt_path}\n"
             f"         Download it first: yolo export model={pt_path} format=pt")

engine_path = pt_path.with_suffix(".engine")
print(f"[export] Source : {pt_path}")
print(f"[export] Output : {engine_path}")
print(f"[export] imgsz  : {config.YOLO_IMGSZ}")
print(f"[export] half   : True (FP16)")
print()

model = YOLO(str(pt_path))
model.export(
    format="engine",
    device=0,
    half=True,
    imgsz=config.YOLO_IMGSZ,
)

print()
print(f"[export] Done → {engine_path}")
print(f"         YOLO_MODEL={engine_path} is already set in .env.jetson")
