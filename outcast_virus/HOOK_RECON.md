# HOOK_RECON.md — How to connect recon (⓷) to the Outcast Virus orchestrator

**Owner:** _Surveillance Lead_ · **Module:** `recon/` · **Runs on:** Colab GPU (training) + browser (render)

Recon is deliberately **offline** — you don't connect a long-running client to the
bus.  Instead, you drop finished files into a known location and the orchestrator
detects them automatically.  Zero custom networking code required.

---

## 1. Drop two files when training completes

```
recon/assets/field.splat    ← the trained Gaussian Splat file
recon/assets/field.json     ← sidecar metadata (optional but useful)
```

The orchestrator polls every 5 seconds.  The moment `field.splat` appears it
broadcasts:

```json
{
  "topic": "recon",
  "status": "ready",
  "splat_url": "/assets/field.splat",
  "frames_used": 220
}
```

The dashboard recon panel picks this up and loads the splat viewer.

---

## 2. Sidecar JSON schema (`field.json`)

```json
{
  "frames_used": 220,
  "splat_url": "/assets/field.splat",
  "scene": "EuRoC Machine Hall 01",
  "trained_at": "2026-05-30T14:22:00Z"
}
```

All fields are optional — the orchestrator uses `frames_used` for the dashboard
counter and `splat_url` to override the default asset URL.  If the sidecar is
missing, sensible defaults are used.

---

## 3. Serving the splat to the browser

The dashboard will try to `fetch("/assets/field.splat")`.  You need a static file
server that maps `/assets/` to `recon/assets/`.

**Quickest option — serve it from Vite's public folder:**
```bash
# symlink (or copy) the finished splat into the frontend public dir
ln -s ../../recon/assets/field.splat frontend/public/field.splat
# Then /assets/ in the browser maps to /field.splat when Vite dev-serves public/
```

**Or add a FastAPI static mount** alongside the orchestrator (add to `orchestrator.py`):
```python
# In orchestrator.py, if you switch to uvicorn/FastAPI:
from fastapi.staticfiles import StaticFiles
app.mount("/assets", StaticFiles(directory="recon/assets"), name="recon-assets")
```

Coordinate with ⓸ on which approach to use.

---

## 4. Workflow: from Colab training to dashboard

```
Phase 0:  Create  recon/assets/  directory (empty).
          Orchestrator broadcasts status: "training".

Phase 1:  Run training on Colab.
          Input:  video frames from the EuRoC dataset.
          Poses:  load from  nav/poses.jsonl  (see HOOK_NAV.md §5).
          Output: field.splat  (download from Colab).

Phase 2:  Download field.splat → copy to  recon/assets/field.splat.
          Orchestrator detects it → broadcasts status: "ready".
          Dashboard loads the splat viewer automatically.

Phase 3:  Overlay the VSLAM trajectory inside the splat viewer.
          Source: subscribe to the "pose" topic on the bus.
```

---

## 5. Getting poses from nav (skip COLMAP)

You need camera poses to feed the Gaussian Splat trainer.  Nav (⓵) writes them
to `nav/poses.jsonl` — one JSON object per line:

```json
{"t": 1.23, "x": 0.1, "y": -0.3, "z": 1.5, "qx":0,"qy":0,"qz":0.707,"qw":0.707, "frame_idx": 42}
```

Convert to COLMAP `images.txt` format:

```python
# recon/poses_to_colmap.py
import json, pathlib

def jsonl_to_colmap(jsonl_path: str, out_path: str):
    lines = []
    with open(jsonl_path) as f:
        for line in f:
            p = json.loads(line)
            # COLMAP: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            lines.append(
                f"{p['frame_idx']} {p['qw']} {p['qx']} {p['qy']} {p['qz']} "
                f"{p['x']} {p['y']} {p['z']} 1 frame_{p['frame_idx']:06d}.png\n\n"
            )
    pathlib.Path(out_path).write_text("".join(lines))

jsonl_to_colmap("nav/poses.jsonl", "recon/colmap/images.txt")
```

---

## 6. Progress signal (optional: live training status)

If you want the dashboard to show training progress (e.g. "training: 1200/30000 iters"),
you can publish directly to the bus from Colab via a tunnel:

```python
# In your Colab training loop — optional
import asyncio, json, websockets

ORCH_WS = "ws://YOUR_NGROK_OR_ORCH_IP:8000"

async def report_progress(iteration, total):
    async with websockets.connect(ORCH_WS) as ws:
        await ws.send(json.dumps({
            "topic": "recon",
            "status": "training",
            "splat_url": "",
            "frames_used": iteration,   # repurpose this field as iteration count
        }))
```

This is optional — the orchestrator already broadcasts `status: "training"` until the
file appears.

---

## 7. Fallback ladder (from TEAM_PLAN §7)

| Scenario | Action |
|----------|--------|
| Full scene splat done | `recon/assets/field.splat` → status: ready |
| Only a sub-clip finished | Use the shorter splat — dashboard doesn't care about length |
| Training failed / no GPU | Copy the nerfstudio sample splat from their repo — pre-baked is fine for the demo |
| Asset served wrong | Check browser network tab for 404; verify the static-file mount or symlink |

Recon is **GREEN** in TEAM_PLAN.  It must never block the rest of the demo.
