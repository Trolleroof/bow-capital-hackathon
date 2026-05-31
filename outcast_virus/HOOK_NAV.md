# HOOK_NAV.md — How to connect nav (⓵) to the Outcast Virus orchestrator

**Owner:** Vikram · **Module:** `nav/` · **Runs on:** Jetson Nano

The orchestrator runs on the laptop/Mac at some IP we'll call `ORCH_IP`.
It listens on port **8000**.  Your job: connect to it as a WebSocket client and
stream pose messages.  That's it.  The orchestrator handles everything else.

---

## 1. What you need to send

Every time ORB-SLAM3 produces a new pose, encode it as JSON and send it over the
WebSocket connection.  Match this schema **exactly** (from `outcast_virus/bus/schema.py`):

```json
{
  "topic": "pose",
  "t": 1234.567,
  "x": 0.123,
  "y": -0.456,
  "z": 1.500,
  "qx": 0.0,
  "qy": 0.0,
  "qz": 0.707,
  "qw": 0.707,
  "gps": false,
  "tracking": "OK"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `topic` | `"pose"` | **must be the literal string `"pose"`** |
| `t` | float | timestamp in seconds (monotonic or epoch, consistent) |
| `x`, `y`, `z` | float | position in metres, camera frame |
| `qx`, `qy`, `qz`, `qw` | float | rotation quaternion (unit) |
| `gps` | `false` | **always false** — that is the whole point |
| `tracking` | string | `"OK"` \| `"NO_LOCK"` \| `"LOST"` from ORB-SLAM3 state |

Target rate: **~30 Hz**.  The dashboard trajectory panel updates at each frame.

---

## 2. Minimal Python client (drop this into `nav/`)

```python
# nav/bus_client.py
import asyncio, json, websockets

ORCH_WS = "ws://ORCH_IP:8000"   # ← replace ORCH_IP

async def publish_pose(t, x, y, z, qx, qy, qz, qw, tracking="OK"):
    msg = json.dumps({
        "topic": "pose",
        "t": t, "x": x, "y": y, "z": z,
        "qx": qx, "qy": qy, "qz": qz, "qw": qw,
        "gps": False, "tracking": tracking,
    })
    await _ws.send(msg)

_ws = None

async def connect_and_stream(pose_generator):
    global _ws
    async with websockets.connect(ORCH_WS) as ws:
        _ws = ws
        async for pose in pose_generator:
            await publish_pose(**pose)
```

Call `connect_and_stream` from your ORB-SLAM3 wrapper.  Reconnect on disconnect
— websockets raises `ConnectionClosed` which you can catch and retry.

---

## 3. What the orchestrator does with your messages

- **Relays** every pose frame to the React dashboard in real time (no code needed from you).
- **Calls** `system_state.beat("nav")` on each received frame → the hero banner
  flips from `LOCALIZED: false` to `LOCALIZED: true`.
- **Stops emitting** the mock circular stub pose (which runs at 1 Hz when you're offline).

---

## 4. Coordinate frame & units (agree Day 1)

- **Units:** metres (not millimetres, not pixels).
- **Frame:** camera frame, right-hand.  X = right, Y = down, Z = forward — match EuRoC.
- **Quaternion convention:** `[qx, qy, qz, qw]` (Hamilton, scalar last).
- If your SLAM outputs SE3 as a 4×4 matrix, extract: `t = T[:3,3]`, rotation from `R = T[:3,:3]`.
- The recon vertical (⓷) will consume these exact poses to skip COLMAP — verify the frame is consistent with the video before handing off.

---

## 5. VSLAM → 3DGS pose handoff (coordinate with ⓷)

The recon team needs camera poses in a format Gaussian Splatting can ingest.
The simplest handoff:

```python
# Append to a file nav/poses.jsonl after each frame:
import json, time
with open("poses.jsonl", "a") as f:
    f.write(json.dumps({
        "t": t, "x": x, "y": y, "z": z,
        "qx": qx, "qy": qy, "qz": qz, "qw": qw,
        "frame_idx": frame_number,
    }) + "\n")
```

⓷ reads `poses.jsonl` and converts to COLMAP format.  Lock this interface in
**Phase 0** so neither of you is blocked.

---

## 6. Health & fallback

- The orchestrator marks nav **"up"** within one received pose, **"degraded"** after
  6 seconds of silence, **"down"** after a full disconnect.
- While you're degraded/down the orchestrator emits a slow circular stub pose so
  the dashboard panel shows something.  It will not pretend you are localised
  (`localized: false` in the status banner).
- If VSLAM + YOLO share Nano resources and VSLAM lags: publish at whatever rate
  you can sustain — even 5 Hz keeps the trajectory visible.

---

## 7. Quick smoke test (no Jetson needed)

```bash
# From repo root — sends 100 fake poses to the orchestrator
python - <<'EOF'
import asyncio, json, math, websockets, time

async def smoke():
    async with websockets.connect("ws://localhost:8000") as ws:
        for i in range(100):
            angle = i * 0.1
            msg = json.dumps({
                "topic": "pose", "t": round(time.monotonic(), 3),
                "x": round(2*math.cos(angle),3),
                "y": round(2*math.sin(angle),3),
                "z": 1.5,
                "qx":0,"qy":0,"qz":0,"qw":1,
                "gps": False, "tracking": "OK",
            })
            await ws.send(msg)
            await asyncio.sleep(0.033)
        print("done — check dashboard trajectory panel")

asyncio.run(smoke())
EOF
```
