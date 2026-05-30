"""Central config -- override via environment variables or .env file."""
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

# Candidate stability buffer -- number of frames to average before switching proposed target
CANDIDATE_BUFFER_FRAMES = int(os.getenv("CANDIDATE_BUFFER_FRAMES", "30"))

# ReID -- cross-frame identity preservation
REID_THRESHOLD        = float(os.getenv("REID_THRESHOLD",        "0.88"))  # cosine sim to accept a re-id
REID_CONSECUTIVE      = int(os.getenv("REID_CONSECUTIVE",        "3"))     # frames a match must hold before reassigning
REID_GALLERY_SIZE     = int(os.getenv("REID_GALLERY_SIZE",       "12"))    # max embeddings in confirmed gallery
REID_PREBUFFER_SIZE   = int(os.getenv("REID_PREBUFFER_SIZE",     "8"))     # max embeddings in each track's pre-buffer
REID_SAMPLE_INTERVAL  = int(os.getenv("REID_SAMPLE_INTERVAL",    "30"))    # frames between samples (~1 s at 30 fps)
REID_MIN_CROP_PX      = int(os.getenv("REID_MIN_CROP_PX",        "32"))    # ignore crops smaller than this (px)
REID_PART_MIN_H       = int(os.getenv("REID_PART_MIN_H",         "48"))    # min crop height for part-based embedding
# Part weights [head/shoulders, torso/loadout, legs] -- must sum to 1.0
REID_PART_WEIGHTS: list[float] = [0.20, 0.50, 0.30]

# Input source: int for camera index, str for video file / RTSP URL
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
