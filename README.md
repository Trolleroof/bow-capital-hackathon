# CombatOS

GPS-denied autonomy OS for unmanned platforms — stereo VSLAM, edge targeting, 3D battlefield reconstruction, offline swarm coordination.

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

**Jetson Nano — first-time setup:**

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
