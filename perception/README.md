# perception/ -- Outcast Virus Targeting Module

**Owner:** Matthieu Fuller (Targeting Lead)

---

## Modes

| Mode | Entry point | Frame source |
|------|-------------|--------------|
| Standalone (YOLO + inline vSLAM) | `main.py` | `cv2.VideoCapture` (webcam / file / RTSP) |
| ROS2 | `camera_node.py` + `rosmain.py` | `/camera/image_raw` ROS2 topic |
| SLAM-only fallback | `slam_sim.py` | `cv2.VideoCapture` (any clip, no YOLO) |

`main.py` / `rosmain.py` run the same YOLO → tracker → ReID → bus pipeline.
`rosmain.py` is a thin adapter that swaps the frame source; every processing
object (`Detector`, `TargetTracker`, `ReIDGallery`, `BusPublisher`, …) is
identical.

`main.py` also drives the monocular vSLAM in `vo.py` inline on the same frames,
so one local video produces both the targeting feed and the SLAM panels in sync
— see **Local full-stack test** below.

---

## Quick start — standalone

```bash
cd perception
cp .env.example .env        # edit VIDEO_SOURCE, WS_HOST, etc.
pip install -r requirements.txt
python main.py
```

---

## Local full-stack test — one video drives every panel

On the Jetson a single ROS camera node feeds **both** YOLO and ORB-SLAM3, so the
dashboard's targeting feed *and* its two SLAM panels (the 3D
`NAVIGATION · STEREO VSLAM` scene and the `SLAM KEYFRAME` feed) come alive off
one image stream. Locally there was no SLAM producer, so those two panels stayed
dark.

`main.py` now reproduces that single-source-of-truth: each frame it reads is fed
to **both** YOLO **and** an inline monocular visual-odometry stage (`vo.py` —
ORB features → essential-matrix `recoverPose` → triangulated sparse cloud). One
capture, one timestamp, so the targeting feed and the SLAM feed are literally the
same frame and stay in lockstep — no second decoder, no drift.

Run **two** processes, both pointed at the same `VIDEO_SOURCE`:

```bash
# 1. orchestrator — control bus (8000) + image bus (8001)
python -m combatos.orchestrator

# 2. YOLO + inline vSLAM on one video → detections, fpv_raw, AND the slam_* topics
cd perception && VIDEO_SOURCE=footage/plane.mp4 python main.py
```

`main.py` publishes the SLAM half via `vo.py`:

| Bus | Topics |
|-----|--------|
| control (8000) | `pose`, `slam_odometry`, `slam_path`, `slam_point_cloud`, `slam_status`, `slam_diagnostics` |
| image (8001) | `slam_frame` (ORB features overlaid); `camera_frame` only when `SLAM_SIM_PUBLISH_CAMERA=1` |

The annotated SLAM frame goes out as `slam_frame` (the `SLAM KEYFRAME` panel);
the raw targeting frame keeps going out as `fpv_raw`. `camera_frame` stays **off**
by default so it does not fight `fpv_raw` for the main targeting panel — enable it
only when exercising the standalone `SlamTestPanel`.

### `slam_sim.py` — SLAM-only fallback

`slam_sim.py` runs the identical `vo.py` VO **without** YOLO, against any clip.
Because it opens its own capture it is *not* frame-synced with a separately
running `main.py`; use it to test SLAM in isolation (e.g. a EuRoC sequence), not
alongside `main.py`.

```bash
cd perception && python slam_sim.py
```

### Notes

Coordinates are converted from the OpenCV optical frame to the ROS REP-103 frame
the 3D scene expects, so the trajectory sits upright on the grid. Monocular VO is
scale-ambiguous and will drift — that is expected for a local test rig; tune
`SLAM_SIM_TRANSLATION_SCALE` for a comfortable trajectory size.

Lock-on is footage-dependent: clips with parallax lock in a few frames
(`plane.mp4` ≈ 0.1 s), while a shaky, smoke-filled intro can read
`INITIALIZING` for a few seconds until there is something to track
(`firefight.mp4` ≈ 3.6 s) — that is honest VO behavior, not a stall. Stability
knobs (`SLAM_SIM_POSE_SMOOTH`, `SLAM_SIM_LOST_GRACE`, `SLAM_SIM_POINTS_PER_FRAME`)
keep the panels from strobing when tracking briefly drops.

### Tuning (env / `.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `SLAM_SIM_VIDEO_FPS` | `10` | Rate frames are pushed to the image bus |
| `SLAM_SIM_PROC_FPS` | `15` | Cap on VO frames/sec (`0` = source speed) |
| `SLAM_SIM_LOOP` | `1` | Restart the clip at EOF so panels keep streaming |
| `SLAM_SIM_PUBLISH_CAMERA` | `0` | Also publish `camera_frame` (SlamTestPanel) |
| `SLAM_SIM_SHOW` | `0` | Open a local OpenCV preview window |
| `SLAM_SIM_ORB_FEATURES` | `1500` | ORB features per frame |
| `SLAM_SIM_MAX_PATH` | `240` | Max trajectory poses sent |
| `SLAM_SIM_MAX_POINTS` | `2500` | Max point-cloud points sent |
| `SLAM_SIM_JPEG_QUALITY` | `70` | `slam_frame` / `camera_frame` JPEG quality |
| `SLAM_SIM_FOCAL_RATIO` | `0.9` | Assumed focal length as a fraction of width |
| `SLAM_SIM_TRANSLATION_SCALE` | `1.0` | Scales the unit VO translation |

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
    ├── BusPublisher → ws://WS_HOST:WS_PORT  (JSON detections)
    │       ↓ WebSocket bus "detections" topic
    │       ↓ React dashboard — detections panel
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
| `vo.py` | Monocular visual odometry + SLAM bus publishing (driven by `main.py`) |
| `slam_sim.py` | SLAM-only fallback: runs `vo.py` standalone against any clip |
| `setup_ros2.sh` | One-shot ROS2 Jazzy install for Ubuntu 22.04+ WSL2 |
| `config.py` | All tunable parameters + env overrides |
| `detector.py` | YOLO inference + Haar cascade face detection |
| `tracker.py` | Norfair tracker, follow-lock, operator confirm state machine |
| `priority.py` | Priority scoring, top-candidate selection |
| `reid.py` | ReID gallery: passive sampling + identity-preserving matching |
| `bus.py` | WebSocket publisher (detections + FPV) + operator command receiver |
| `visualizer.py` | OpenCV HUD overlay |
| `export_trt.py` | One-shot TensorRT export for Jetson |
