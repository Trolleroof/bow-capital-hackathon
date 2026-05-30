"""Central config — override via environment variables or .env file."""
import os
from dotenv import load_dotenv

load_dotenv()

# Model
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")          # swap for .engine on Jetson
YOLO_CONF  = float(os.getenv("YOLO_CONF", "0.40"))
YOLO_IOU   = float(os.getenv("YOLO_IOU",  "0.45"))
DEVICE     = os.getenv("DEVICE", "cpu")                      # "0" for GPU/TRT on Jetson

# Face detection
FACE_CONF  = float(os.getenv("FACE_CONF", "0.60"))

# Tracker
MAX_DISTANCE     = float(os.getenv("MAX_DISTANCE", "150"))
MAX_LOST_FRAMES  = int(os.getenv("MAX_LOST_FRAMES", "30"))

# Bus
WS_HOST = os.getenv("WS_HOST", "localhost")
WS_PORT = int(os.getenv("WS_PORT", "8765"))
WS_TOPIC = "detections"

# COCO → battlefield label remap
BATTLEFIELD_LABELS: dict[str, str] = {
    "person":     "troop",
    "car":        "vehicle",
    "truck":      "vehicle",
    "bus":        "vehicle",
    "motorcycle": "ugv",
    "bicycle":    "ugv",
    "airplane":   "aerial",
    "helicopter": "aerial",
}

# Priority weights per battlefield label (higher = higher priority)
CLASS_WEIGHTS: dict[str, float] = {
    "troop":   1.0,
    "vehicle": 0.8,
    "ugv":     0.9,
    "aerial":  0.7,
    "unknown": 0.4,
}

# Input source: int for camera index, str for video file / RTSP URL
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
