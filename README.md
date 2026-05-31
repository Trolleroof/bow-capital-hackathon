# Outcast Virus

GPS-denied autonomy OS for unmanned platforms -- stereo VSLAM, edge targeting, 3D battlefield reconstruction, offline swarm coordination.

## Modules

| Module | Owner | Dir |
|--------|-------|-----|
| Navigation (stereo VSLAM) | Vikram | `nav/` |
| Targeting (YOLO + tracking) | Matthieu | `perception/` |
| Surveillance (3D Gaussian Splat) | TBD | `recon/` |
| Swarm + Integration | Nikhil | `swarm/` + `frontend/` |

---

## Perception (Targeting)

```bash
cd perception
cp .env.example .env        # set VIDEO_SOURCE, WS_HOST, WS_PORT
pip install -r requirements.txt
python main.py
```

**Jetson Nano -- first-time setup:**

```bash
python export_trt.py        # exports yolo11n.engine (run once)
# then in .env: YOLO_MODEL=yolo11n.engine  DEVICE=0
python main.py
```

See [`perception/README.md`](./perception/README.md) for full docs.

---

## Frontend

React + TypeScript app in [`frontend/`](./frontend), managed with [Bun](https://bun.sh).

```bash
cd frontend
bun install
bun dev
```

Other commands:

```bash
bun run build    # production build → frontend/dist
bun run preview  # preview production build
bun run lint
```

---

## Backend (Swarm API)

From the repo root, start the FastAPI training/sim server on port **8787**:

```bash
uv run --project swarm uvicorn swarm.backend:app --host 127.0.0.1 --port 8787
```

The process stays idle until the frontend calls `POST /api/train/start` or sim endpoints. Requires [uv](https://docs.astral.sh/uv/) and deps in `swarm/` (`uv sync --project swarm` if imports fail).

If port 8787 is already in use:

```bash
lsof -ti :8787 | xargs kill
```

---

## Gym training (MAPPO → ONNX)

Run the **backend** (above), then the frontend. Clicking **Train Policy** starts training only for the selected gym environment.

In a second terminal:

```bash
cd frontend
bun install
bun dev
```

Enter a gym environment, tune battlefield params, and click **Train Policy**. Metrics stream over WebSocket; on completion the service exports `frontend/public/policies/<env_id>/policy.onnx` and unlocks **Mission Sim**.

Optional one-terminal dev mode:

```bash
VITE_AUTO_TRAIN_SERVICE=1 bun dev
```

Optional env overrides:

- `VITE_TRAIN_API_URL` — default uses the Vite `/api` proxy to `http://127.0.0.1:8787`
- `VITE_TRAIN_WS_URL` — default `ws://127.0.0.1:8787/ws`
- `VITE_TRAIN_TIMESTEPS` — default `12000`
- `VITE_ALLOW_HEURISTIC_MISSION=1` — skip policy gate in dev
