# Bow Capital Hackathon

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

## Gym training (MAPPO → ONNX)

`bun dev` **starts the training bridge automatically** (HTTP `:8787`, WebSocket `:8766`). You need [uv](https://docs.astral.sh/uv/) installed for Python deps.

```bash
cd frontend
bun install
bun dev
```

Enter a gym environment, tune battlefield params, and click **Train Policy**. Metrics stream over WebSocket; on completion the service exports `frontend/public/policies/<env_id>/policy.onnx` and unlocks **Mission Sim**.

To run the bridge manually instead (e.g. `VITE_AUTO_TRAIN_SERVICE=0 bun dev`):

```bash
uv run --project swarm python -m swarm.train_service
```

If you see `Address already in use` on port 8787, a previous train service is still running (that's usually fine — **Train Policy still works**). To stop it:

```bash
lsof -ti :8787 | xargs kill
lsof -ti :8766 | xargs kill
```

Optional env overrides:

- `VITE_TRAIN_WS_URL` — default `ws://127.0.0.1:8766`
- `VITE_TRAIN_TIMESTEPS` — default `12000`
- `VITE_ALLOW_HEURISTIC_MISSION=1` — skip policy gate in dev
