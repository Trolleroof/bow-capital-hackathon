# Swarm Vertical — Decentralized Multi-Agent RL → Edge Inference

**Owner:** Nikhil (⓸ Swarm + Integration Lead) · **Repo area:** `swarm/`
**Status:** design locked, building.

> One line: a swarm of drones that coordinates with **comms denied** — a real
> learned MAPPO policy with **local-observation-only** actors, trained offline,
> run at the **edge** (browser today, Jetson tomorrow), rendered in Three.js.

This supersedes TEAM_PLAN.md §4 ⓸'s "MuJoCo multi-agent PPO on Mac" note. MuJoCo is
dropped (weak multi-agent fit); the spine below is the implementation.

---

## 0. Why this is the real thing (not the faked demo)

The reference we're beating ("SplatSwarm") is a Three.js scene with **scripted**
movement + LLM narration — no trained policy, no sim. Ours renders just as well but the
motion comes from an **actual neural net**. When a judge asks "real or scripted?": open
devtools, show the `.onnx` running with the network **offline**, show the reward curve.

The decentralized framing is not decoration — it's the defense thesis made literal:
**CTDE** (Centralized Training, Decentralized Execution). Train with a critic that sees
global state; deploy actors that see **only local observations and send zero messages
to each other**. That *is* "offline / comms-denied coordination."

---

## 1. Architecture — train heavy, run at the edge

```
  Mac CPU (or cloud/JAX)          one-time, offline
  ┌────────────────────────┐
  │  MAPPO / CTDE training  │   point-mass swarm env, velocity actions
  │  (PyTorch, CleanRL-ish) │
  └───────────┬────────────┘
              │ export actor net
              ▼
       policy.onnx  ──────────────────────────────────────┐
              │                                            │
   Phase A (do first)                          Phase B (the "later" flex)
  ┌───────────▼─────────────┐               ┌──────────────▼──────────────┐
  │ Browser inference        │               │ Jetson: ONNX → TensorRT      │
  │ onnxruntime-web (WASM/   │               │ run actor on-device,         │
  │ WebGPU), step sim in JS  │               │ comms denied, edge silicon   │
  └───────────┬─────────────┘               └──────────────┬──────────────┘
              │  publishes "swarm" message (§5 contract)    │
              └───────────────────┬─────────────────────────┘
                                  ▼
                    Three.js swarm scene in React dashboard
                 (ideally composited with 3DGS field + VSLAM path)
```

**Key property:** Phase A and Phase B emit the **identical** `swarm` bus message, so the
dashboard is agnostic. If the Jetson is flaky at demo time, flip back to browser
inference — nobody notices. The Jetson is a flex you can fall back from.

**Do NOT train on the Jetson.** It's an inference device, shared with VSLAM + YOLO.
Training stays on the Mac/cloud. A swarm actor is a tiny MLP (~thousands of params,
microseconds/step), so it costs ~nothing next to YOLO/VSLAM on the shared Jetson.

---

## 2. The learning task — coverage/search + attrition resilience

- **N agents** cover an area/volume; **local obs only**, shared policy, no inter-agent
  comms.
- **Money demo:** mid-rollout, "kill" an agent (`alive:false`) — the swarm **re-covers
  the gap with zero communication**. Emergent, because CTDE.
- Coverage/search ties into the **Surveillance** vertical — the swarm searches the same
  field that ⓷ reconstructs in 3DGS. Four verticals, one story.

**Observation (per agent, local):** own position/velocity, relative positions of K
nearest neighbors, local coverage/occupancy patch, goal/role flag. **No global state.**
**Action:** 2D/3D velocity command (continuous). Low-level flight handled by the
(spoofed) flight controller — we learn *coordination*, not PID.

---

## 3. Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| RL algo | **MAPPO + CTDE** | hand-rolled CleanRL-style first (one file, learnable). |
| DL framework | **PyTorch (CPU)** | tiny nets converge on CPU. |
| Env API | **Gymnasium + PettingZoo** | parallel multi-agent API. |
| Physics (core) | **point-mass kinematics** | velocity actions, integrate in code; converges fast. |
| Physics (stretch) | **gym-pybullet-drones / PyBullet** | real quadrotor dynamics → B-roll only. |
| Export | **ONNX** (`torch.onnx`) | export the **actor** only. |
| Edge A | **onnxruntime-web** | browser inference, WASM/WebGPU. |
| Edge B | **TensorRT** on Jetson | `onnx → trt`, same as YOLO path. |
| Render | **Three.js** | in the React dashboard. |
| Bus | Python WS server + JSON | §5 of TEAM_PLAN. |

---

## 4. Bus message (3D — extends TEAM_PLAN §5)

```jsonc
// topic: "swarm"  (~10 Hz)
{ "t": 1234.56, "comms": "denied",
  "agents": [
    { "id": 0, "x": 1.2, "y": -0.4, "z": 2.1, "yaw": 0.3, "role": "scout", "alive": true }
  ] }
```

`alive:false` drives the kill-an-agent demo.

```jsonc
// topic: "train"  (from train.py or mock_train_publisher.py, ~1-2 Hz for UI)
{ "topic": "train",
  "env_id": "search-and-interdict",
  "profile": "combat",
  "phase": "update",              // init | baseline | update | eval | checkpoint | final
  "step": 6400,
  "reward_mean": 37.42,
  "coverage": 0.781,
  "losses": {
    "pg_loss": 0.041,
    "v_loss": 0.218,
    "entropy": 0.067,
    "approx_kl": 0.012
  },
  "params_hash": "c19d6d4a9f10" }
```

Frontend-dev mock:

```bash
uv run --project swarm python -m swarm.mock_train_publisher --env-id drone-vs-drone --profile combat
```

This serves a WebSocket broadcaster on `ws://localhost:8766` that emits the same
`topic: "train"` JSON shape as the real trainer, so frontend work can proceed
without waiting on a live MAPPO run.

---

## 5. Build phases (never get blocked)

Each phase is independently demoable and falls back to the prior one. Ship in order —
don't start a phase until the previous one's **Done** criterion is green.

### Phase 0 — Env + random policy (the pipe)
**Goal:** prove data flows env → bus → Three.js before any learning exists.
- Build the point-mass swarm `gymnasium`/`pettingzoo` env: N agents, local obs (§2),
  velocity actions, coverage reward, `alive` flag, world bounds.
- Drive it with a **random policy**; publish the `swarm` message (§4) over the WS bus.
- Stub Three.js panel: N dots moving in a box.
- **Files:** `swarm/env.py`, `swarm/bus.py`, `frontend/src/panels/SwarmPanel.tsx`.
- **Done:** random agents visibly move in the dashboard, streaming live over the bus.

### Phase 1 — MAPPO training loop (the brain)
**Goal:** a policy that actually coordinates, trained on CPU.
- Hand-rolled CleanRL-style **MAPPO + CTDE**: shared-param actor (local obs), centralized
  critic (global state, train-only). PPO clip, GAE.
- Log to TensorBoard; **save the reward curve** (it's a slide).
- Checkpoint best policy to `swarm/checkpoints/<env_id>/policy.pt`, plus
  `params.json`, `meta.json`, and `train-events.ndjson`.
- **Files:** `swarm/train.py`, `swarm/mappo.py`, `swarm/models.py`.
- **Done:** reward climbs past random baseline; rollout shows agents spreading to cover,
  not clumping. Save a before/after (random vs trained) clip.

### Phase 2 — Export → ONNX (parity)
**Goal:** the trained **actor** runs outside PyTorch, identically.
- `torch.onnx.export` the actor only (plain MLP, standard ops, static input shape).
- Verify parity: same obs → same action within tolerance, Python torch vs `onnxruntime`.
- **Files:** `swarm/export_onnx.py`, output `frontend/public/policies/<env_id>/policy.onnx`.
- **Done:** parity test passes; the env-specific `policy.onnx` sits beside the matching
  checkpoint under both `swarm/checkpoints/<env_id>/` and `frontend/public/policies/<env_id>/`.

### Phase 3 (Edge A) — Browser inference (the guaranteed demo)
**Goal:** the neural net runs **client-side**, no Python in the loop. This is the demo.
- `onnxruntime-web` (WASM, WebGPU if available) loads the env-specific export.
- Port the env **step** to TypeScript (point-mass integration is trivial); each frame:
  build per-agent local obs → ORT actor → velocity → integrate → render.
- **Files:** `frontend/src/swarm/sim.ts`, `frontend/src/swarm/policy.ts`.
- **Done:** open the dashboard offline (network tab killed) and the swarm still
  coordinates — proof it's real and edge-local.

Frontend loader contract:

```ts
await loadPolicy("search-and-interdict")
// resolves to /policies/search-and-interdict/policy.onnx
```

Artifact layout:

```text
swarm/checkpoints/
  drone-vs-drone/
    policy.pt
    policy.onnx
    params.json
    meta.json
    train-events.ndjson
  search-and-interdict/
    policy.pt
    policy.onnx
    params.json
    meta.json
    train-events.ndjson

frontend/public/policies/
  drone-vs-drone/policy.onnx
  search-and-interdict/policy.onnx
```

### Phase 4 — Three.js polish + the money demo
**Goal:** match SplatSwarm's production value, then beat it with the kill demo.
- Drone meshes, ground/field, camera; **coordination minimap**; optional **POV insets**.
- **Kill-an-agent button** → set `alive:false` → swarm re-covers the gap, no comms.
- HUD: `COMMS: DENIED`, agent count, coverage %.
- **Done:** click-kill an agent on stage and the swarm visibly reflows around the loss.

### Phase 5 (Edge B, stretch) — Jetson on-device inference (the flex)
**Goal:** the trained brain runs on real edge silicon, streaming to the laptop.
- Convert `policy.onnx` → **TensorRT** engine on the Jetson (same path as YOLO).
- Run the actor on-device; publish the **identical** `swarm` message over the bus.
- **Done:** dashboard renders a swarm driven by the Jetson; flip back to Edge A if flaky.

### Phase 6 (stretch) — real dynamics + scene compositing
**Goal:** extra credibility and the unified hero shot.
- `uv sync --extra drones` → **gym-pybullet-drones** real-quadrotor **B-roll** clip.
- Composite the swarm into the **same Three.js scene** as ⓷'s 3DGS field + ⓵'s VSLAM
  path — one 3D world, all verticals inside it.
- **Done:** a single 3D view with the swarm flying through the reconstructed field.

**MVP line:** Phases 0→4 are the must-ship demo. Phases 5–6 are pure upside —
never let them block the spine.

---

## 6a. Gym scenario registry

Issue `#7` adds a hard-coded scenario registry shared by the frontend gym page and
the Python training entry point in `swarm/scenarios.py`.

```python
from swarm import make_scenario_env

env = make_scenario_env("search-and-interdict", seed=7)
obs = env.reset()
```

All scenarios currently reuse the same `SwarmEnv` point-mass core, but each one
pins different environment knobs and documents the intended reward/telemetry
shape so we can fork into dedicated env subclasses later without changing ids.

| Scenario id | Operator task | Observation sketch | Action space | Reward sketch |
|---|---|---|---|---|
| `drone-vs-drone` | Contest hostile airspace, survive contact, hold a denial lane | local neighbors, contested-lane occupancy, friendly/alive counts | continuous 2D velocity per drone | reward lane control + survival; penalize blue-on-blue crowding and losses |
| `moving-target-track` | Maintain visual custody on evasive ground movers | target-relative bearings, occlusion bins, wingman offsets | continuous 2D velocity | reward uninterrupted custody and multi-angle coverage; penalize lost track |
| `search-and-interdict` | Sweep cluttered space, find hidden mover, collapse once contact is made | coverage patch, jammer pockets, obstacle slices, last-seen cue | continuous 2D velocity | reward new search coverage pre-contact, then rapid intercept post-contact |
| `defend-asset` | Keep inbound agents outside a protected ring | asset-relative bearings, defended sectors, inbound velocity cues | continuous 2D velocity | reward perimeter integrity and early intercepts; penalize breaches |
| `swarm-vs-swarm-race` | Win contested coverage first under jamming | coverage patch, contested cells, rival offsets, jammer corridors | continuous 2D velocity | reward first-touch coverage and zone control; penalize collisions |

Frontend mapping:
- `frontend/src/gym/scenarios.ts` is the card registry and operator copy.
- `frontend/src/gym/GymScenarioStage.tsx` renders the hard-coded 2D gym floor,
  agents, obstacles/assets, and scenario telemetry for demos.

---

## 6b. Battlefield parameter obs/reward delta (issues #13–#15)

P0 parameters from `swarm/env_config.py` change the env as follows.
See `docs/battlefield-parameters.md` for the full catalog and priority tiers.

| P0 Parameter | Obs delta | Reward delta | Dynamics delta |
|---|---|---|---|
| `wind_speed` + `wind_dir_rad` | None (position obs reflects real position after drift) | Coverage rate drops as wind pushes agents off target cells; bounds penalty increases at high speed | Live agents drift `wind_vector × DT` every step after their command is applied |
| `gps_denial_level` | `obs[0:2]` (own pos) receives Gaussian noise σ = level×0.2 | None directly | None |
| `jam_duty_cycle` | Each of the K neighbor slots in `obs[4:10]` is independently zeroed with probability `jam_duty_cycle` | None directly; indirectly increases crowding because agents fly without neighbor awareness | None |
| `attrition_inject_rate` | Dead agents leave neighbor sets (zero-filled slots) | Team loses coverage contributors; dead agents receive 0 reward | `kill()` called probabilistically; agent freezes |
| `battery_envelope_sec` / `time_limit_sec` | None | Shorter horizon increases urgency | Episode truncates at `min(battery_envelope_sec, time_limit_sec)` |

**CTDE constraint:** all P0 obs deltas affect only per-agent local observations.
The centralized critic additionally sees 4 normalized P0 scalars appended to
`global_state()`: `[wind_speed/15, jam_duty_cycle, gps_denial_level, attrition_rate/0.5]`.
These are **never** in the actor input — pure CTDE.

---

## 7. Risks & fallbacks

| Risk | Fallback |
|------|----------|
| MAPPO won't converge in time | shrink agents/obs; ship shorter-horizon policy or scripted-baseline-vs-learned comparison. |
| Jetson busy / TensorRT flaky | Phase A browser inference is the real demo; Jetson is the flex. |
| ONNX export mismatch | keep actor net simple (plain MLP, standard ops); verify parity early. |
| pybullet eats time | it's stretch B-roll only — never let it block the spine. |

---

## 8. Dev setup

```bash
# Python sim/training env (managed by uv)
uv sync --project swarm  # installs from swarm/pyproject.toml
uv run --project swarm python -m swarm.train --env-id search-and-interdict

# Frontend already has three + onnxruntime-web (see frontend/package.json)
```
