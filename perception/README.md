# perception/ -- CombatOS Targeting Module

**Owner:** Matthieu Fuller (Targeting Lead)

## Quick start

```bash
cd perception
cp .env.example .env        # edit VIDEO_SOURCE, WS_HOST, etc.
pip install -r requirements.txt
python main.py
```

## Jetson setup

```bash
python export_trt.py        # builds yolo11n.engine (run once on Jetson)
# then set YOLO_MODEL=yolo11n.engine DEVICE=0 in .env
python main.py
```

## Controls (debug window)

| Key | Action |
|-----|--------|
| `c` | Confirm proposed candidate as target |
| `f` | Lock follow mode on proposed candidate |
| `r` | Release follow mode |
| `q` | Quit |

## Bus output

Publishes to WebSocket topic `detections` per §5 of TEAM_PLAN.md:

```json
{
  "topic": "detections",
  "data": {
    "t": 1234.56,
    "objects": [
      { "id": 7, "cls": "troop", "conf": 0.91, "bbox": [x,y,w,h],
        "has_face": true, "is_target": true, "confirmed": false }
    ]
  }
}
```

## File map

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters + env overrides |
| `detector.py` | YOLO inference + MediaPipe face overlay |
| `tracker.py` | Norfair tracker, follow-lock, operator confirm |
| `priority.py` | Priority scoring, top-candidate selection |
| `bus.py` | WebSocket publisher |
| `visualizer.py` | OpenCV debug overlay |
| `main.py` | Main loop |
| `export_trt.py` | One-shot TensorRT export for Jetson |
