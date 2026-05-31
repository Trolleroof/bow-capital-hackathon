# perception/ -- CombatOS Targeting Module

**Owner:** Matthieu Fuller (Targeting Lead)

---

## Modes

| Mode | Entry point | Frame source |
|------|-------------|--------------|
| Standalone | `main.py` | `cv2.VideoCapture` (webcam / file / RTSP) |
| ROS2 | `camera_node.py` + `rosmain.py` | `/camera/image_raw` ROS2 topic |

Both modes run the same YOLO → tracker → ReID → bus pipeline.  `rosmain.py`
is a thin adapter that swaps the frame source; every processing object
(`Detector`, `TargetTracker`, `ReIDGallery`, `BusPublisher`, …) is identical.

---

## Quick start — standalone

```bash
cd perception
cp .env.example .env        # edit VIDEO_SOURCE, WS_HOST, etc.
pip install -r requirements.txt
python main.py
```

---

## Quick start — ROS2 (WSL / Windows 11)

### 1. Install ROS2

Run once inside your WSL terminal:

```bash
chmod +x perception/setup_ros2.sh
./perception/setup_ros2.sh
source ~/.bashrc
```

This installs ROS2 Jazzy, `cv_bridge`, and `image_transport` on Ubuntu 22.04+ WSL2.

### 2. Attach the USB camera to WSL

From a Windows PowerShell (Administrator), using [usbipd-win](https://github.com/dorssel/usbipd-win/releases):

```powershell
usbipd list                        # find camera BUSID, e.g. 2-3
usbipd bind   --busid 2-3
usbipd attach --wsl --busid 2-3
```

Verify in WSL:

```bash
ls /dev/video*                     # should show /dev/video0
```

### 3. Run

Open two WSL terminals, both with ROS2 sourced:

```bash
# Terminal 1 — camera publisher
source /opt/ros/jazzy/setup.bash
cd perception
python camera_node.py

# Terminal 2 — perception node (ROS2 input)
source /opt/ros/jazzy/setup.bash
cd perception
python rosmain.py
```

### 4. Inspect

```bash
ros2 topic list
ros2 topic hz /camera/image_raw           # verify camera is publishing
ros2 topic hz /perception/detections      # verify YOLO is running
ros2 run rqt_image_view rqt_image_view    # view annotated video feed
```

---

## Data flow

```
USB Camera (/dev/video0)
    │
    ▼
camera_node.py
  cv2.VideoCapture → sensor_msgs/Image (BGR8)
    │
    │  /camera/image_raw
    ▼
rosmain.py  (or main.py with VideoCapture directly)
  Detector  → YOLO inference
  TargetTracker → Norfair multi-object tracking
  ReIDGallery   → identity-preserving re-identification
  CandidateBuffer → priority scoring
    │
    ├── /perception/detections (std_msgs/String JSON)
    │       ↓
    │   ros_perception_module.py (orchestrator)
    │       ↓ WebSocket bus "detections" topic
    │       ↓ React dashboard — detections panel
    │
    ├── BusPublisher → ws://WS_HOST:WS_PORT  (JSON detections)
    │
    └── BusPublisher → ws://WS_HOST:IMAGE_WS_PORT  (JPEG FPV stream)
```

Operator commands flow back:

```
React dashboard → orchestrator WebSocket → BusPublisher._receive_loop()
    → publisher.commands queue (drained each frame)
    → tracker state machine (follow / confirm / release)
```

---

## Jetson setup

```bash
python export_trt.py        # builds yolo11n.engine (run once on Jetson)
# then set in .env:
# YOLO_MODEL=yolo11n.engine
# DEVICE=0
python main.py              # or rosmain.py if Jetson publishes camera via ROS2
```

---

## Controls (debug window)

| Key | Action |
|-----|--------|
| `f` | Follow: lock onto proposed candidate (or re-lock confirmed target if visible) |
| `c` | Confirm: lock in the followed target (requires follow mode) |
| `r` | Release: step back one level (Confirmed → Followed → Proposed) |
| `u` | Reset: clear confirmed target, follow lock, and ReID gallery |
| `Space` | Toggle recording (saves `raw.mp4` + `hud.mp4` to `footage/recordings/`) |
| `q` | Quit |

---

## Bus output

Publishes JSON to WebSocket topic `detections` per §5 of TEAM_PLAN.md:

```json
{
  "topic": "detections",
  "t": 1234.56,
  "objects": [
    {
      "id": 7,
      "cls": "troop",
      "conf": 0.91,
      "bbox": [0.12, 0.34, 0.08, 0.21],
      "has_face": true,
      "is_primary": false,
      "is_candidate": false,
      "confirmed": true
    }
  ]
}
```

`bbox` values are normalized to `[0, 1]` as `[x, y, w, h]`.

---

## Configuration

All values are set via environment variables or `.env` (loaded by `python-dotenv`).

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `0` | Webcam index, file path, or RTSP URL (standalone mode) |
| `ROS_CAMERA_TOPIC` | `/camera/image_raw` | ROS2 topic camera_node publishes to / rosmain subscribes from |
| `YOLO_MODEL` | `yolo11n.pt` | `.pt` (PyTorch), `.onnx`, or `.engine` (TensorRT) |
| `YOLO_CONF` | `0.4` | Per-box confidence threshold |
| `YOLO_IMGSZ` | `640` | YOLO inference resolution (px). Try `416` or `320` on Jetson Nano to reduce VRAM |
| `DEVICE` | `cpu` | YOLO device: `cpu`, `0`, or `cuda:0` |
| `REID_DEVICE` | `cpu` | ReID extractor device. Keep `cpu` on Jetson so YOLO gets the full GPU budget |
| `WS_HOST` | `localhost` | Orchestrator hostname |
| `WS_PORT` | `8000` | Orchestrator control bus port |
| `IMAGE_WS_PORT` | `8001` | Orchestrator image bus port |
| `PROC_WIDTH` | `0` | Downscale width before inference (0 = native) |
| `DETECT_EVERY` | `1` | Run YOLO every N frames; tracker interpolates between |
| `FPV_INTERVAL` | `3` | Publish raw frame every N frames (~10 fps at 30 fps source) |
| `FPV_QUALITY` | `60` | JPEG quality for FPV stream (0–100) |
| `GRAYSCALE_STREAM` | `1` | Convert FPV frames to grayscale before transmitting |

See `config.py` for the full list including tracker, ReID, and face detection tuning knobs.

---

## File map

| File | Purpose |
|------|---------|
| `main.py` | Standalone entry point: VideoCapture → full pipeline |
| `rosmain.py` | ROS2 entry point: `/camera/image_raw` → same full pipeline |
| `camera_node.py` | ROS2 camera publisher: VideoCapture → `/camera/image_raw` |
| `setup_ros2.sh` | One-shot ROS2 Jazzy install for Ubuntu 22.04+ WSL2 |
| `config.py` | All tunable parameters + env overrides |
| `detector.py` | YOLO inference + Haar cascade face detection |
| `tracker.py` | Norfair tracker, follow-lock, operator confirm state machine |
| `priority.py` | Priority scoring, top-candidate selection |
| `reid.py` | ReID gallery: passive sampling + identity-preserving matching |
| `bus.py` | WebSocket publisher (detections + FPV) + operator command receiver |
| `visualizer.py` | OpenCV HUD overlay |
| `export_trt.py` | One-shot TensorRT export for Jetson |
