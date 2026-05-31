"""Central config -- all values overridable via environment variables or .env file.

Tuning quick-reference
----------------------
Detection sensitivity  -- lower YOLO_CONF catches more targets but raises false positives.
Tracker stickiness     -- raise MAX_LOST_FRAMES to hold IDs through longer occlusions;
                          raise MAX_DISTANCE if targets move fast between frames.
ReID aggressiveness    -- lower REID_THRESHOLD to re-link more liberally (more false
                          re-links); raise REID_CONSECUTIVE to require a longer match
                          streak before committing (slower but more confident).
Candidate switching    -- raise CANDIDATE_BUFFER_FRAMES to make the proposed target
                          stickier; lower it to react faster to a higher-priority target.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Detection model
# ---------------------------------------------------------------------------

# Ultralytics model to load.  Use a .pt file for CPU/GPU PyTorch inference,
# a .onnx file for ONNX Runtime, or a .engine file for TensorRT on Jetson.
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")

# Minimum per-box confidence score; boxes below this are discarded outright.
YOLO_CONF  = float(os.getenv("YOLO_CONF", "0.4"))

# IoU threshold for non-maximum suppression; lower = more aggressive merging
# of overlapping boxes (useful when targets are close together).
YOLO_IOU   = float(os.getenv("YOLO_IOU",  "0.5"))

# Inference device: "cpu", a CUDA index string like "0", or "cuda:0".
# On Jetson use "0" to hit the TensorRT engine via the CUDA provider.
DEVICE     = os.getenv("DEVICE", "cpu")

# YOLO inference image size (pixels, square). Smaller = faster + less VRAM.
# 640 is the default; on Jetson Nano try 320 or 416 if hitting OOM.
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))

# Device for the ReID embedding extractor.
# CPU is preferred on memory-constrained hardware: CUDA allocations are pinned
# in physical RAM and cannot be swapped out, while CPU allocations can spill to
# swap. ReID runs on small crops every ~30 frames so swap latency is acceptable.
REID_DEVICE = os.getenv("REID_DEVICE", "cpu")

# Set to "0" to skip loading the ReID model entirely (~150 MB saved).
# The gallery will silently no-op; operator confirm/follow still works,
# but identity is not preserved across tracker ID changes.
REID_ENABLED = os.getenv("REID_ENABLED", "1") == "1"

# ---------------------------------------------------------------------------
# Face detection (Haar cascade run inside each troop bounding box)
# ---------------------------------------------------------------------------

# Minimum cascade detector confidence to count a face hit.  Higher values
# reduce false positives at the cost of missing partially-occluded faces.
FACE_CONF  = float(os.getenv("FACE_CONF", "0.60"))

# ---------------------------------------------------------------------------
# Norfair multi-object tracker
# ---------------------------------------------------------------------------

# Maximum centroid-to-centroid distance (pixels) allowed when associating a
# new detection with an existing track.  Raise for fast-moving or high-res sources.
MAX_DISTANCE     = float(os.getenv("MAX_DISTANCE", "150"))

# Frames a track survives without a matching detection before being deleted.
# At 30 fps, the default of 30 gives ~1 s of occlusion tolerance.
MAX_LOST_FRAMES  = int(os.getenv("MAX_LOST_FRAMES", "30"))

# Consecutive frames a new detection must be seen before it becomes a visible track.
# Eliminates single-frame ghost detections without delaying real targets noticeably.
TRACK_INIT_DELAY = int(os.getenv("TRACK_INIT_DELAY", "3"))

# EMA factor for bbox smoothing (0-1). Lower = smoother but more lag.
# 0 disables smoothing; 0.35 is a good balance at 30 fps.
BBOX_SMOOTH_ALPHA = float(os.getenv("BBOX_SMOOTH_ALPHA", "0.35"))

# ---------------------------------------------------------------------------
# WebSocket event bus
# ---------------------------------------------------------------------------

# Hostname of the downstream consumer (dashboard / swarm integrator).
WS_HOST  = os.getenv("WS_HOST", "localhost")

# Port the perception node publishes to. This should be the CombatOS
# orchestrator control bus, not the old standalone perception bus.
WS_PORT  = int(os.getenv("WS_PORT", "8000"))
IMAGE_WS_PORT = int(os.getenv("IMAGE_WS_PORT", "8001"))

# Message topic written into every published payload.
WS_TOPIC = "detections"

# ---------------------------------------------------------------------------
# COCO label -> battlefield label remap
#
# Any COCO class not listed here is published as "unknown" and receives the
# lowest priority weight.  Add entries to expose additional COCO classes.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Target priority
# ---------------------------------------------------------------------------

# Scalar weight per battlefield label used by CandidateBuffer to rank visible
# targets.  Higher = more likely to be proposed as the follow candidate.
CLASS_WEIGHTS: dict[str, float] = {
    "troop":   1.0,
    "vehicle": 0.8,
    "ugv":     0.9,
    "aerial":  0.7,
    "unknown": 0.4,
}

# Number of frames over which per-track priority scores are averaged before
# the system proposes a new candidate.  Smooths out single-frame detections.
CANDIDATE_BUFFER_FRAMES = int(os.getenv("CANDIDATE_BUFFER_FRAMES", "30"))

# ---------------------------------------------------------------------------
# Re-identification (ReID)
#
# ReID preserves a target's identity across tracker ID changes caused by
# occlusion, re-entry, or tracker reset.  Embeddings are extracted passively
# into per-track pre-buffers and promoted to a confirmed gallery on operator
# confirm.  When the confirmed track ID goes missing, every other visible
# track is scored against the gallery; a streak of high-similarity frames
# triggers re-assignment.
# ---------------------------------------------------------------------------

# Minimum cosine similarity (0-1) between a candidate crop and the confirmed
# gallery to count as a match frame.  0.88 works well for OSNet; may need
# lowering to ~0.75 when using the MobileNetV3 fallback.
REID_THRESHOLD        = float(os.getenv("REID_THRESHOLD",        "0.88"))

# Number of consecutive frames a candidate must exceed REID_THRESHOLD before
# the confirmed ID is reassigned to it.  Guards against a single misleading frame.
REID_CONSECUTIVE      = int(os.getenv("REID_CONSECUTIVE",        "3"))

# Maximum embeddings stored in the confirmed gallery (ring buffer).  More
# embeddings = better coverage of pose/lighting variation; higher CPU cost
# per match frame (linear scan).
REID_GALLERY_SIZE     = int(os.getenv("REID_GALLERY_SIZE",       "12"))

# Maximum embeddings per unconfirmed track's pre-buffer.  Pre-buffers are
# promoted to the gallery on confirm and demoted back on release.
REID_PREBUFFER_SIZE   = int(os.getenv("REID_PREBUFFER_SIZE",     "8"))

# Frames between embedding samples for any given track.  At 30 fps the
# default of 30 gives ~1 sample/second, balancing diversity vs. CPU load.
REID_SAMPLE_INTERVAL  = int(os.getenv("REID_SAMPLE_INTERVAL",    "30"))

# Run YOLO detection only every Nth frame; tracker interpolates in between.
# 1 = detect every frame (slowest, most accurate).
# 2 = detect every other frame (~1.4x throughput gain with minimal tracking error).
# 3 = every 3rd frame (~1.8x gain, good for slow-moving targets).
DETECT_EVERY = int(os.getenv("DETECT_EVERY", "1"))

# Crops narrower or shorter than this (pixels) are skipped entirely; too
# small to yield a reliable embedding.
REID_MIN_CROP_PX      = int(os.getenv("REID_MIN_CROP_PX",        "32"))

# Minimum crop height (pixels) to attempt part-based embedding.  Below this,
# the extractor falls back to a single global embedding of the full crop.
REID_PART_MIN_H       = int(os.getenv("REID_PART_MIN_H",         "48"))

# Weights for the three horizontal body strips [head/shoulders, torso/loadout, legs].
# Torso is weighted highest because loadout and clothing are most discriminative
# for uniformed targets.  Must sum to 1.0.
REID_PART_WEIGHTS: list[float] = [0.20, 0.50, 0.30]

# ---------------------------------------------------------------------------
# Video input
# ---------------------------------------------------------------------------

# Integer string → webcam index (e.g. "0").
# Absolute path   → video file (MP4, AVI, etc.).
# RTSP/HTTP URL   → network stream.
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

# ROS2 topic names (used by camera_node.py and rosmain.py).
# camera_node.py publishes here; rosmain.py subscribes here.
ROS_CAMERA_TOPIC = os.getenv("ROS_CAMERA_TOPIC", "/camera/image_raw")

# Frames are downscaled to this width (pixels) before detection and tracking;
# height is derived automatically to preserve the source aspect ratio.
# Smaller values improve throughput at the cost of small-target recall.
# Set to 0 to disable rescaling and process at native resolution.
PROC_WIDTH = int(os.getenv("PROC_WIDTH", "0"))

# ---------------------------------------------------------------------------
# FPV stream (HUD frame broadcast over WebSocket)
# ---------------------------------------------------------------------------

# JPEG encode quality for the FPV stream (0-100). Lower = smaller payload.
FPV_QUALITY = int(os.getenv("FPV_QUALITY", "60"))

# Send every Nth processed frame. At 30 fps, 3 → ~10 fps stream.
FPV_INTERVAL = int(os.getenv("FPV_INTERVAL", "3"))

# Convert FPV stream frames to grayscale before encoding.
# Reduces transmitted data to ~1 byte/pixel (single-channel) vs 3 bytes/pixel
# for RGB.  Detection pipeline is unaffected and continues to run on color.
# Set to "1" to enable.
GRAYSCALE_STREAM = os.getenv("GRAYSCALE_STREAM", "1") == "1"
