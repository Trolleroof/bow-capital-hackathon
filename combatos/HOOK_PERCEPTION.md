# HOOK_PERCEPTION.md — How to connect perception (⓶) to the CombatOS orchestrator

**Owner:** Matthieu Fuller · **Module:** `perception/` · **Runs on:** Jetson Nano

Same pattern as nav: connect to the orchestrator on port **8000** and publish
JSON frames.  The orchestrator relays everything to the dashboard detections panel.

---

## 1. What you need to send

One frame per YOLO inference pass.  Send even if the frame has zero detections —
it keeps the health heartbeat alive and the dashboard panel live.

```json
{
  "topic": "detections",
  "t": 1234.567,
  "objects": [
    {
      "id": 7,
      "cls": "person",
      "conf": 0.91,
      "bbox": [0.42, 0.31, 0.18, 0.24],
      "is_target": true,
      "confirmed": false
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `topic` | `"detections"` | **must be the literal string `"detections"`** |
| `t` | float | timestamp in seconds (same clock as nav if possible) |
| `objects` | array | empty list `[]` is valid and expected between targets |
| `id` | int | stable track ID from ByteTrack / Norfair across frames |
| `cls` | string | YOLO class label, e.g. `"person"`, `"vehicle"` |
| `conf` | float | 0–1 YOLO confidence score |
| `bbox` | `[x, y, w, h]` | **normalised 0–1**, top-left origin, relative to frame |
| `is_target` | bool | `true` when this object is the highest-priority candidate |
| `confirmed` | bool | `false` until the operator clicks CONFIRM in the dashboard |

---

## 2. Minimal Python client (drop this into `perception/`)

```python
# perception/bus_client.py
import asyncio, json, websockets

ORCH_WS = "ws://ORCH_IP:8000"   # ← replace ORCH_IP

async def publish_detections(t: float, objects: list[dict]):
    msg = json.dumps({"topic": "detections", "t": t, "objects": objects})
    await _ws.send(msg)

_ws = None

async def connect(detection_generator):
    global _ws
    while True:
        try:
            async with websockets.connect(ORCH_WS) as ws:
                _ws = ws
                async for frame in detection_generator:
                    await publish_detections(frame["t"], frame["objects"])
        except Exception as e:
            print(f"[bus] reconnecting: {e}")
            await asyncio.sleep(1.0)
```

---

## 3. Building the `objects` list from YOLO + ByteTrack output

```python
import time

def yolo_to_bus(results, tracker, frame_w, frame_h, target_id=None):
    """Convert ultralytics Results + tracker output to the bus schema."""
    objects = []
    tracks = tracker.update(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls_id = track
        objects.append({
            "id": int(track_id),
            "cls": results.names[int(cls_id)],
            "conf": round(float(conf), 3),
            # normalise bbox to [0-1] xywh
            "bbox": [
                round((x1) / frame_w, 4),
                round((y1) / frame_h, 4),
                round((x2 - x1) / frame_w, 4),
                round((y2 - y1) / frame_h, 4),
            ],
            "is_target": int(track_id) == target_id,
            "confirmed": False,  # operator sets this via dashboard
        })
    return {"t": round(time.monotonic(), 3), "objects": objects}
```

---

## 4. Operator confirm gate

`confirmed` starts as `false`.  The dashboard will have a "CONFIRM TARGET" button.
When the operator clicks it, the dashboard sends back a control message:

```json
{ "topic": "confirm_target", "id": 7 }
```

To receive this, subscribe after connecting:

```python
# Right after connecting, send a subscribe message to receive confirm events:
await ws.send(json.dumps({"type": "subscribe", "topics": ["confirm_target"]}))

# Then in your receive loop:
async for raw in ws:
    msg = json.loads(raw)
    if msg.get("topic") == "confirm_target":
        confirmed_id = msg["id"]
        # set confirmed=True on your next detections frame for that track_id
```

Once confirmed, set `"confirmed": true` for that `id` in subsequent frames.

---

## 5. What the orchestrator does with your messages

- **Relays** every detections frame to the dashboard detections panel.
- **Beats** `system_state("perception")` on every received frame.
- **Emits** empty `detections` stubs at 1 Hz when you're offline so the panel
  shows a live empty feed rather than freezing on the last frame.

---

## 6. Nano resource sharing with nav (coordinate with Vikram)

From §7 of TEAM_PLAN: if VSLAM + YOLO can't share the Nano comfortably:
- **VSLAM stays on-device** — GPS-denied localisation is the hero message.
- **Move YOLO to laptop** — same `bus_client.py`, just connect from laptop instead.
- Decide in Phase 1.  The bus contract doesn't change either way.

---

## 7. Quick smoke test

```bash
# Sends 20 fake detection frames to the orchestrator
python - <<'EOF'
import asyncio, json, time, websockets

async def smoke():
    async with websockets.connect("ws://localhost:8000") as ws:
        for i in range(20):
            msg = json.dumps({
                "topic": "detections",
                "t": round(time.monotonic(), 3),
                "objects": [{
                    "id": 1, "cls": "person", "conf": 0.88,
                    "bbox": [0.4, 0.3, 0.15, 0.3],
                    "is_target": True, "confirmed": False,
                }],
            })
            await ws.send(msg)
            await asyncio.sleep(0.1)
        print("done — check dashboard detections panel")

asyncio.run(smoke())
EOF
```
