"""Canonical uvicorn entrypoint for the Outcast Virus backend.

Run:
    uv run --project swarm uvicorn swarm.backend:app --host 127.0.0.1 --port 8787

The backend is idle at startup. Training starts only when the frontend calls
``POST /api/train/start`` with the selected gym ``env_id``.
"""

from __future__ import annotations

from swarm.train_service import app, main


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
