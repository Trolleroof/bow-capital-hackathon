# CombatOS -- Team Execution Plan

**Bow Capital × DS3 × SIC Defense Hackathon · May 29–31, 2026 · UCSD DIB**
**Track:** Autonomous Navigation & Edge AI (Hardware+) · **Team size:** 4

> **One line:** CombatOS is the GPS-denied autonomy OS for unmanned platforms -- it
> localizes from stereo vision alone, identifies and tracks targets at the edge,
> reconstructs the battlefield in 3D, and coordinates a swarm with the network down.
> One OS, swap the body (RC car today → drone tomorrow).

---

## 0. Why this wins (read before building)

The track brief literally lists our whiteboard back to us: *"edge AI, autonomous
navigation without GPS, swarming behavior, and offline coordination… systems that
can operate with little to no internet."* We are not adapting to the rubric -- we
**are** the rubric. Every design choice below ladders up to one message:

**No GPS. No network. Still flies. Still fights.**

Two judge audiences -- tailor every panel and every slide to both:
- **Track judges (Autonomous Nav & Edge AI):** GPS-denied VSLAM running on-device, edge inference, offline operation.
- **Challenge judges (FireStorm = loitering munitions/payloads, Qualcomm = edge silicon, TargetX = unmanned systems):** "CombatOS is a *payload/augment OS* that drops onto any unmanned platform and runs with zero connectivity."

---

## 1. First-principles spine (RED) vs. swappable (GREEN)

This is from the whiteboard and it drives everything. RED = non-negotiable, derived
from first principles. GREEN = an implementation choice we can swap if something better appears.

| # | Capability | Color | First-principles reason it's RED / why it's swappable |
|---|------------|-------|-------------------------------------------------------|
| 1 | **GPS-denied navigation** (stereo VSLAM → live trajectory) | 🔴 RED | In a real defense environment GPS is jammed/spoofed. Self-localization from onboard sensors is the irreducible requirement. |
| 2 | **Autonomous target determination** (identify + track, operator-confirmed) | 🔴 RED | The platform must perceive and prioritize on its own when the link is down. |
| 2a | YOLO object detector / face detector | 🟢 GREEN | Detector model is interchangeable (YOLO11n, RT-DETR, etc.). The *autonomy loop* around it is the red part. |
| 3 | Battlefield surveillance -- **3D Gaussian Splat** of the field | 🟢 GREEN | "Dream" add-on. Reconstruction method is swappable (3DGS ↔ NeRF ↔ photogrammetry). Explicitly **not** real-time. |
| 4 | **Swarm behavior** -- offline / decentralized coordination | 🔴 RED (concept) | Coordinating with comms denied is first-principles defense. |
| 4a | PPO policy in MuJoCo | 🟢 GREEN | The RL algorithm/sim is the swappable implementation of the red concept. |

**Bake the RED framing into the product itself:** the dashboard's hero banner reads
**`GPS: DENIED ✓  ·  LINK: NONE ✓  ·  LOCALIZED`** -- the absence is the feature.

---

## 2. Hardware & compute reality (locked)

| Thing | Decision |
|-------|----------|
| Input | **Real drone footage** (recorded), replayed as the live stereo stream. Same footage feeds perception **and** reconstruction. |
| "Flight controller" | **Spoofed** -- we emulate the FC telemetry (IMU/sensor stream) over a MAVLink-style interface. No live aircraft. |
| Edge compute | **Jetson Nano** runs everything online: stereo VSLAM, YOLO, face detection. |
| Training compute | **Mac (Intel CPU)** runs PPO training only. |
| 3DGS training | ⚠️ Needs CUDA → **train offline on a cloud GPU (Colab)**; render in-browser. (It's GREEN/offline, so this is fine -- see Risks.) |
| Recommended dataset | **EuRoC MAV** -- real micro-aerial-vehicle **stereo + IMU** footage. It literally *is* "real drone footage with FC sensor data," perfect for VSLAM + as 3DGS input. Bring-your-own footage = stretch. |

---

## 3. Architecture at a glance

```
            ┌─────────────────────────────────────────────────────────┐
            │  INPUT: real drone footage  +  SPOOFED flight-controller │
            │         (stereo video)          (IMU/telemetry replay)   │
            └───────────────┬───────────────────────────┬─────────────┘
                            │                           │
        ┌───────────────────▼─────────┐   ┌─────────────▼─────────────┐
   ⓵    │  NAVIGATION (Jetson)        │   │  TARGETING (Jetson)   ⓶    │
        │  Stereo VSLAM → 6-DoF pose  │   │  YOLO + face + target loop │
        │  + FC-spoof harness         │   │  (operator-confirmed)      │
        └───────────────┬─────────────┘   └─────────────┬─────────────┘
                        │  pose stream                 │  detections
                        │   (poses also feed 3DGS ▼)   │
        ┌───────────────▼─────────────┐   ┌─────────────▼─────────────┐
   ⓷    │  SURVEILLANCE (offline GPU) │   │  SWARM (Mac CPU)      ⓸    │
        │  3D Gaussian Splat of field │   │  MuJoCo multi-agent PPO,   │
        │  (rendered in browser)      │   │  decentralized / comms-off │
        └───────────────┬─────────────┘   └─────────────┬─────────────┘
                        │                               │
                        └──────────────┬────────────────┘
                          WebSocket bus │  (JSON topics, defined in §5)
                        ┌───────────────▼───────────────┐
                        │   CombatOS DASHBOARD (React)   │  ← owner ⓸ owns the shell
                        │  Banner: GPS DENIED · LINK NONE│     each owner ships 1 panel
                        │  [traj] [detections] [splat] [swarm]
                        └────────────────────────────────┘
```

**Key synergy:** VSLAM (⓵) produces camera poses → hand them to 3DGS (⓷) so it skips
COLMAP. Wire that interface early; it's a free win and a great slide.

---

## 4. The 4-person split -- scope, ownership, "done"

> Self-assign in the table, then **each person owns their vertical end-to-end,
> including its dashboard panel.** No one is blocked waiting on a "frontend person."
> The only shared surface is the message contract in §5 -- agree on it in Phase 0 and
> nobody steps on anyone.

| Role | Owner | Repo area |
|------|-------|-----------|
| ⓵ Navigation Lead | Vikram (VSLAM experience) | `nav/` |
| ⓶ Targeting Lead | Matthieu Fuller | `perception/` |
| ⓷ Surveillance Lead | _claim me_ | `recon/` |
| ⓸ Swarm + Integration Lead | Nikhil | `swarm/` + `frontend/` |
Alex? Dashboard? Maybe @alexgaoth look here
---

### ⓵ Navigation Lead -- GPS-Denied Stereo VSLAM  🔴 *hero #1*

**Scope (yours alone):** Turn a stereo video stream into a live 6-DoF trajectory with
**zero GPS**, plus the spoofed flight-controller harness that feeds it sensor data.

- Stand up **stereo VSLAM** on the Jetson Nano -- **ORB-SLAM3 (stereo, optionally stereo-inertial)**. Output = **trajectory only** (pose over time). **No dense mapping.**
- Build the **FC-spoof harness**: replay the dataset's IMU/telemetry over a MAVLink-style stream (`pymavlink` or a simple ZMQ feed) so the story is "the flight controller streams sensor data to CombatOS exactly like on a real drone."
- Own the footage ingestion (EuRoC MAV primary).
- Publish `pose` messages to the bus (§5). Ship the **trajectory panel** + the **`GPS: DENIED`** hero banner in the dashboard.
- **Hand your poses to ⓷** so 3DGS skips COLMAP.

**Done =** drive a clip, watch a clean trajectory trace out live in the dashboard with GPS visibly off, running on the Jetson.
**Stack:** ORB-SLAM3, OpenCV, pymavlink/ZMQ, Jetson (JetPack), three.js panel.
**Risk you own:** VSLAM + YOLO sharing the Nano. Coordinate with ⓶ early -- if it's tight, VSLAM is the on-device hero (GPS-denied is THE message); YOLO can move to laptop.

---

### ⓶ Targeting Lead -- Autonomous Target Determination  🔴 *hero #2*

**Scope (yours alone):** From the same video feed, detect → identify → track → **lock a
target with an operator-confirm gate**. Runs on the Jetson.

- **YOLO** object detection (YOLO11n / YOLOv8n via Ultralytics, **export to TensorRT** for the Nano) + a lightweight **face detector** (YOLO-face / RetinaFace-mobile / MediaPipe).
- **Target-selection logic:** pick the highest-priority detection, keep identity across frames with a simple tracker (**ByteTrack / Norfair**), surface it as a candidate.
- **Human-in-the-loop:** the system *proposes*; an operator confirms. This is both the honest version of what's buildable in 36h and the professional/ethical framing judges want. Frame everything as **"target identification & tracking,"** not "kill list." (Also: keep names clean -- drop "Battle propaganda," it'll get flagged under the "professional environment" rule.)
- Publish `detections` to the bus. Ship the **detections/lock panel**.

**Done =** live feed with boxes, a tracked candidate, and a "CONFIRM TARGET" gate that an operator clicks -- all on-device.
**Stack:** Ultralytics YOLO, TensorRT, ByteTrack/Norfair, OpenCV, React panel.
**Risk you own:** Nano FPS. Use the nano models + TensorRT from the start; have a laptop fallback path.

---

### ⓷ Surveillance Lead -- 3D Gaussian Splat Reconstruction  🟢 *dream feature*

**Scope (yours alone):** Reconstruct the battlefield in 3D from the drone footage and
let judges fly through it -- with ⓵'s trajectory overlaid inside the scene.

- **Train a 3D Gaussian Splat** of the scene from the footage. **Reuse ⓵'s VSLAM poses** as camera poses (skip COLMAP) -- coordinate that interface day 1.
- ⚠️ **Train on a cloud GPU (Colab)** -- it's offline/"not real time" by design (matches the whiteboard). Don't try to train on the Nano/Intel Mac.
- Render in-browser (web splat viewer: `gsplat.js` / antimatter15 splat / react-three viewer). Overlay the drone's path through the reconstruction = the killer visual.
- Publish `recon` status/asset to the bus. Ship the **splat-viewer panel**.
- **Secondary (because this is the most "demo-ready" / lowest-coupling vertical):** lead the **5-min demo video** assembly.

**Done =** an interactive splat of the field in the dashboard with the VSLAM trajectory threaded through it.
**Stack:** gaussian-splatting / nerfstudio (gsplat), Colab GPU, web splat viewer.
**Risk you own:** GPU access + train time. Kick training off **Phase 1** (it's long-running). Fallback ladder: full splat → splat of a short sub-clip → pre-baked sample splat. It's GREEN -- never let it block the spine.

---

### ⓸ Swarm + Integration Lead -- Offline Coordination + the CombatOS spine  🔴 *concept*

**Scope (yours alone):** A decentralized multi-agent swarm in sim, **plus** the
connective tissue that makes four verticals look like one OS. (You own integration
because PPO training runs unattended -- you have the spare cycles.)

- **MuJoCo multi-agent PPO**, trained on the Mac Intel CPU. Keep it small (few agents, modest obs/action) so CPU training converges in time. SB3/CleanRL PPO + PettingZoo-style wrapper.
- **Decentralized = the defense angle:** shared policy, *local* observations, **no central server, comms denied** → emergent coordination. That's "offline coordination" made literal.
- Record rollouts → ship the **swarm panel** (top-down agents).
- **Integration spine (own this from Phase 0):**
  - The **WebSocket bus + message contract** (§5) -- define it first so everyone codes to it.
  - The **`frontend/` dashboard shell** (React 19 + Vite + Bun, already scaffolded): layout, the `GPS DENIED · LINK NONE` hero banner, and the 4 panel slots others fill.
  - The **pitch deck + narrative** for both judge audiences. Attend the **Pitching Workshop (Sat 5–6 PM)**.

**Done =** N agents coordinating with comms off in the dashboard, and all four panels live in one CombatOS view streaming over the bus.
**Stack:** MuJoCo, SB3/CleanRL, PettingZoo, Python WS server (`websockets`/FastAPI), React/Vite/Bun, three.js.
**Risk you own:** integration cliff. Enforce a **daily integrated build** -- modules talk over the contract from Phase 1, not Sunday.

---

## 5. Shared contract (the only thing all four touch)

A Python WebSocket server broadcasts JSON on topics; the React app subscribes. **Agree
on these schemas in Phase 0 and freeze them.** Everyone mocks the others' messages to
work in parallel.

```jsonc
// topic: "pose"  (from ⓵, ~30 Hz)
{ "t": 1234.56, "x":0,"y":0,"z":0, "qx":0,"qy":0,"qz":0,"qw":1, "gps": false, "tracking": "OK" }

// topic: "detections"  (from ⓶, per frame)
{ "t": 1234.56, "objects": [
    { "id": 7, "cls": "vehicle", "conf": 0.91, "bbox": [x,y,w,h], "is_target": true, "confirmed": false } ] }

// topic: "recon"  (from ⓷, on update)
{ "status": "ready", "splat_url": "/assets/field.splat", "frames_used": 220 }

// topic: "swarm"  (from ⓸, ~10 Hz)
{ "t": 1234.56, "comms": "denied", "agents": [ { "id":0, "x":1.2,"y":-0.4, "role":"scout" } ] }

// topic: "train"  (from ⓸ trainer or mock publisher, ~1-2 Hz)
{ "topic": "train", "env_id": "search-and-interdict", "profile": "combat",
  "phase": "update", "step": 6400, "reward_mean": 37.42, "coverage": 0.781,
  "losses": { "pg_loss": 0.041, "v_loss": 0.218, "entropy": 0.067, "approx_kl": 0.012 },
  "params_hash": "c19d6d4a9f10" }

// topic: "status"  (system, hero banner)
{ "gps":"DENIED", "link":"NONE", "localized": true, "modules": { "nav":"up","perception":"up","recon":"ready","swarm":"up" } }
```

**Interface decisions to make in the first hour:** coordinate frame & units for pose,
bbox convention (xywh vs xyxy), how VSLAM poses are exported to 3DGS, dataset/clip everyone tests on.

---

## 5a. Orchestrator — CombatOS integration spine

> **Owner: ⓸** · Built Phase 0 (stubs) → Phase 2 (real modules) · Single entry point: `python orchestrator.py`

The orchestrator is the single process that boots all four verticals, runs the WebSocket pub/sub bus, aggregates system state into the `status` topic, and triggers fallbacks when a module fails. It is what makes four repo folders look like one OS.

### Directory layout

```
combatos/
├── orchestrator.py           # entry point — boots everything, runs event loop
├── bus/
│   ├── ws_server.py         # FastAPI WebSocket broker (pub/sub by topic)
│   ├── router.py            # topic → subscriber list; cross-module relay rules
│   └── schema.py            # Pydantic models enforcing the §5 message contract
├── modules/
│   ├── base.py              # AbstractModule: start / stop / health_check / on_message
│   ├── nav_module.py        # spawns nav/ process; relays pose → recon (COLMAP-skip)
│   ├── perception_module.py # spawns perception/ process; relays detections
│   ├── recon_module.py      # offline — polls Colab asset, emits recon status
│   └── swarm_module.py      # spawns swarm/ process; relays swarm msgs
├── state/
│   ├── system_state.py      # GPS / LINK / LOCALIZED flags; module health map
│   └── fallback.py          # fallback ladder logic per §7
└── config.py                # ports, topic names, heartbeat interval, thresholds
```

### Core responsibilities

| Responsibility | Mechanism |
|---|---|
| **Module lifecycle** | `asyncio.create_subprocess_exec` per module; restart on crash up to N retries |
| **WebSocket bus** | FastAPI + uvicorn; React dashboard and modules subscribe by topic name |
| **Status aggregation** | Polls module heartbeats → writes `status` topic at 1 Hz → drives hero banner |
| **Pose relay to recon** | `nav_module` echoes every `pose` msg to `recon_module` internally — VSLAM→3DGS handoff wired here, not ad-hoc |
| **Fallback triggering** | 3 missed heartbeats → mark module degraded, emit updated `status`; never crash the bus |

### System state machine

```
BOOT → INITIALIZING → LOCALIZING → OPERATIONAL
                                   ↓ (module heartbeat loss)
                                DEGRADED → OPERATIONAL (on recovery)
```

`system_state.py` owns transitions. Every module emits an internal heartbeat every ~2 s. Missing 3 consecutive beats flips that module to `"down"` in the `status` topic — the dashboard banner updates automatically.

### Module base interface (`base.py`)

```python
class AbstractModule:
    async def start(self) -> None: ...   # spawn subprocess or connect
    async def stop(self) -> None: ...    # graceful shutdown
    async def health_check(self) -> bool: ...   # liveness probe
    async def on_message(self, topic: str, payload: dict) -> None: ...  # inbound relay
```

Each concrete module implements this; the orchestrator event loop calls them uniformly regardless of where the module physically runs (Jetson, Mac, Colab).

### Stack

- **Python 3.11+**, `asyncio` throughout — no threads
- **FastAPI** + `uvicorn` for the WebSocket server
- **Pydantic v2** for schema validation at the bus boundary (malformed messages → logged warning, not crash)
- **`websockets`** for the React client connection

### Build order (ties to timeline)

| Phase | Orchestrator milestone |
|-------|----------------------|
| **0** | `ws_server` + `schema.py` + stub modules emitting mock data → dashboard develops against real bus immediately |
| **1** | Swap stubs for real subprocess modules one at a time as verticals come online |
| **2** | Wire nav→recon pose relay; enable `status` aggregation from live heartbeats |
| **3** | Fallback logic + graceful degradation; record stable demo run |

### Why this matters for the demo

- The `GPS: DENIED · LINK: NONE` hero banner is driven by `system_state.py` — it reflects actual module liveness, not a hardcoded string.
- If YOLO moves to laptop (§7 fallback), only `perception_module.py` changes — bus contract and dashboard are untouched.
- `python orchestrator.py` is the single command judges see start the whole OS.

---

## 6. Timeline -- you lose both nights, so MVP locks Saturday

DIB closes ~11 PM Fri and ~10 PM Sat; it is **not** overnight. Real working hours ≈ 18.
**Hard deadline: Sun May 31, 11 AM** (soft 10 AM). Plan to it.

| Phase | When | Goal | Per-owner |
|-------|------|------|-----------|
| **0 -- Foundations** | Fri 8–11 PM | Repo skeleton (`nav/ perception/ recon/ swarm/ frontend/`), **freeze the §5 contract**, Jetson flashed, dataset downloaded, dashboard shell + 4 empty panels, MuJoCo + Colab smoke tests. | Everyone unblocks their own pipeline + mocks the bus. |
| **1 -- Verticals solo** | Sat 9 AM–1 PM | Each vertical works in isolation on the dataset. **Kick off the long-runners now:** ⓷ Colab 3DGS training, ⓸ PPO training. | ⓵ trajectory out · ⓶ boxes on feed · ⓷ training + viewer · ⓸ PPO + shell. |
| **2 -- Integrate** | Sat 1–6 PM | Wire all four to the bus → dashboard shows trajectory + detections live together. VSLAM→3DGS pose handoff. ⓸ at Pitch Workshop 5–6. | Daily integrated build #1. |
| **3 -- MVP lock + polish** | Sat 6–10 PM | End-to-end demo runs in one CombatOS view. Splat + trajectory overlay. Swarm viz polished. **Record a backup demo run before the building closes.** | Freeze MVP. |
| **4 -- Ship** | Sun 9–11 AM | Record the **5-min video** (inspiration / development / demo), Devpost, GitHub README, slides final. Submit by 10 AM. | ⓷ leads video, ⓸ leads deck. |

---

## 7. Risks & fallbacks (decide the fallback before you need it)

| Risk | Owner | Fallback |
|------|-------|----------|
| 3DGS needs CUDA; no good local GPU | ⓷ | Train on Colab (planned). Ladder: full splat → sub-clip splat → pre-baked sample. It's GREEN. |
| Jetson can't run VSLAM **and** YOLO at once | ⓵+⓶ | VSLAM stays on-device (it's the message); move YOLO to laptop, or time-slice clips. Decide Phase 1. |
| Stereo "real drone footage" hard to source | ⓵ | **EuRoC MAV** = real drone stereo + IMU, off the shelf. |
| PPO won't converge on CPU in time | ⓸ | Shrink agents/obs; pre-train Phase 1; ship a shorter-horizon policy or scripted-baseline-vs-learned comparison. |
| Integration cliff on Sunday | ⓸ | Contract frozen Phase 0; integrated build every phase; no big-bang merge. |
| Lost nights eat the schedule | all | MVP **must** be demoable Sat ~10 PM; Sunday is video + submit only. |

---

## 8. Demo script (the 5 minutes that win)

1. **Cold open -- the absence is the feature.** Dashboard up, banner pulsing `GPS: DENIED · LINK: NONE`. "Everything you're about to see runs with no GPS and no network." Trajectory traces out live from stereo vision alone.
2. **Autonomy.** Detections light up; a target is proposed and an operator confirms it -- "the platform perceives and prioritizes itself; the human stays in the loop."
3. **Surveillance.** Fly through the 3D Gaussian Splat of the battlefield with the drone's path threaded through it.
4. **Scale.** Cut to the swarm coordinating with comms denied. "One policy, no server, no link."
5. **Close -- the OS thesis.** "CombatOS is a payload OS. RC car today, drone tomorrow -- same stack, swap the body. Built for the day the network goes dark." (Aim this line straight at FireStorm / TargetX / Qualcomm.)

---

## 9. Definition of done

**MVP (must demo):** ⓵ live GPS-denied trajectory on Jetson · ⓶ live detection + operator-confirmed target lock · ⓸ swarm coordinating in sim, all in **one** CombatOS dashboard over the bus.
**Stretch (green):** ⓷ 3DGS field with trajectory overlay · VSLAM→3DGS pose handoff · on-device YOLO+VSLAM simultaneously · bring-your-own drone footage.

**Submission checklist:** GitHub repo (clean READMEs per module) · 5-min demo video · Devpost (team, name "CombatOS", repo link) · slides tailored to track **and** challenge judges. **Submit by Sun 10 AM soft / 11 AM hard -- no late submissions, ever.**
