"""Persistent backend API for the Outcast Virus gym UI.

Starting this process does not train a policy. It only serves HTTP/WebSocket
endpoints. Training is spawned on demand when the frontend calls
``POST /api/train/start`` for a specific ``env_id``.

HTTP (default :8787):
  POST /api/train/start   {"env_id","profile","timesteps"?}
  POST /api/train/stop    {"env_id"?}
  POST /api/train/export  {"env_id"}
  GET  /api/train/status?env_id=
  GET  /api/sim/status
  POST /api/sim/start     {"env_id","policy","camera_mode"?,"selected_drone"?}
  POST /api/sim/stop

WebSocket (same port as HTTP, :8787):
  ws://127.0.0.1:8787/ws       — Broadcasts `topic: "train"` JSON lines.
  ws://127.0.0.1:8787/sim/ws   — Broadcasts `topic: "pybullet_frame"` / `topic: "swarm"`.

Run:
  uvicorn swarm.train_service:app --host 127.0.0.1 --port 8787
  # compatibility wrapper:
  uv run --project swarm python -m swarm.train_service
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from swarm.train import checkpoint_paths

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)

HOST_HTTP = "127.0.0.1"
PORT_HTTP = 8787

_WS_CLIENTS: set[WebSocket] = set()
_SIM_WS_CLIENTS: set[WebSocket] = set()
_LOCK = threading.Lock()
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None
_JOBS: dict[str, dict[str, Any]] = {}
_SIM_JOB: dict[str, Any] | None = None
_NO_CLIENT_WARNED_AT = 0.0


# ── request/response models ─────────────────────────────────────────────────


class TrainStartRequest(BaseModel):
    env_id: str = "search-and-interdict"
    profile: str = "combat"
    timesteps: int = 100_000


class TrainStopRequest(BaseModel):
    env_id: str | None = None


class TrainExportRequest(BaseModel):
    env_id: str = "search-and-interdict"


class SimStartRequest(BaseModel):
    env_id: str = "search-and-interdict"
    policy: str = "trained"
    camera_mode: str = "observer"
    selected_drone: int = 0


# ── WebSocket helpers ──────────────────────────────────────────────────────


def _schedule_ws_send(ws: WebSocket, line: str) -> None:
    """Send to a WebSocket from a background thread via the main event loop."""
    loop = _MAIN_LOOP
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(_send_to_ws(ws, line), loop)


def _broadcast_sync_line(line: str) -> None:
    """Queue a JSON line for broadcast to train clients (sync context)."""
    global _NO_CLIENT_WARNED_AT
    with _LOCK:
        clients = list(_WS_CLIENTS)
    if not clients:
        # Training still runs; metrics are persisted to train-events.ndjson and
        # GET /api/train/status. Avoid spamming the terminal when the gym UI
        # is not open or the socket has not connected yet.
        now = time.time()
        if now - _NO_CLIENT_WARNED_AT > 120.0:
            print(
                "[train] note: no gym WebSocket clients connected — "
                "open the gym UI or connect to ws://127.0.0.1:8787/ws for live metrics",
                flush=True,
            )
            _NO_CLIENT_WARNED_AT = now
        return
    _NO_CLIENT_WARNED_AT = 0.0
    for ws in clients:
        try:
            _schedule_ws_send(ws, line)
        except Exception as exc:
            print(f"[train] broadcast error: {exc}", flush=True)


def _broadcast_sim_sync_line(line: str) -> None:
    """Queue a JSON line for broadcast to sim clients (sync context)."""
    with _LOCK:
        clients = list(_SIM_WS_CLIENTS)
    for ws in clients:
        try:
            _schedule_ws_send(ws, line)
        except Exception:
            pass


async def _send_to_ws(ws: Any, msg: str) -> None:
    """Send a message to a WebSocket client."""
    try:
        await ws.send_text(msg)
    except Exception:
        pass


# ── training control ───────────────────────────────────────────────────────


def _is_job_running(job: dict[str, Any]) -> bool:
    proc = job.get("proc")
    return proc is not None and proc.poll() is None


def _kill_proc(proc: subprocess.Popen | None, *, wait: bool = True) -> None:
    """Terminate a child process (and its group on Unix) without hanging shutdown."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass

    if not wait:
        return

    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=0.5)
        except Exception:
            pass


def _popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
    """Spawn a child in its own session so Ctrl+C can kill the whole tree."""
    if sys.platform != "win32":
        kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(*args, **kwargs)


def _tail_training(env_id: str, proc: subprocess.Popen) -> None:
    """Read train.py stdout (NDJSON) and fan out to WebSocket clients."""
    assert proc.stdout is not None
    last = {}
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"[train] {line}", flush=True)
            continue
        if event.get("topic") == "train":
            last = event
            with _LOCK:
                job = _JOBS.get(env_id, {})
                job["last"] = event
                _JOBS[env_id] = job
            _broadcast_sync_line(line)
        else:
            print(f"[train] {line}", flush=True)

    code = proc.wait()
    with _LOCK:
        job = _JOBS.get(env_id, {})
        job["running"] = False
        job["last"] = last or job.get("last")
        job["status"] = "completed" if code == 0 else "failed"
        job["returncode"] = code
        _JOBS[env_id] = job

    if code == 0 and os.path.exists(checkpoint_paths(env_id)["policy"]):
        try:
            subprocess.run(
                [sys.executable, "-m", "swarm.export_onnx", "--env-id", env_id],
                cwd=REPO,
                check=True,
            )
            exported = {
                "topic": "train",
                "env_id": env_id,
                "phase": "exported",
                "step": last.get("step", 0),
                "reward_mean": last.get("reward_mean", 0),
                "coverage": last.get("coverage", 0),
                "losses": last.get("losses", {}),
                "params_hash": last.get("params_hash", ""),
            }
            print(f"[train] broadcasting exported event for {env_id}", flush=True)
            _broadcast_sync_line(json.dumps(exported))
            time.sleep(0.1)  # Give WebSocket time to deliver the message
        except Exception as exc:
            print(f"[train] export failed: {exc}", flush=True)
            failed = {
                "topic": "train",
                "env_id": env_id,
                "phase": "export_failed",
                "error": str(exc),
                "step": last.get("step", 0),
                "reward_mean": 0,
                "coverage": 0,
                "losses": {},
                "params_hash": last.get("params_hash", ""),
            }
            _broadcast_sync_line(json.dumps(failed))


def _drive_sim_policy(env_id: str, policy: str, proc: subprocess.Popen) -> None:
    assert proc.stdin is not None
    try:
        from swarm.bus import _random_policy, _trained_policy, swarm_message
        from swarm.scenarios import make_scenario_env
    except Exception as exc:
        print(f"[pybullet-sim] policy driver import failed: {exc}", flush=True)
        return

    # Use the scenario env so spawn areas / agent count match how the policy was
    # trained, instead of the generic point-mass defaults.
    try:
        env = make_scenario_env(env_id, seed=0)
    except KeyError:
        from swarm.env import SwarmEnv
        env = SwarmEnv(seed=0)
    obs = env.reset()
    ckpt = checkpoint_paths(env_id)["policy"]
    if policy == "trained" and os.path.exists(ckpt):
        policy_fn = _trained_policy(ckpt)
        label = "trained"
    else:
        if policy == "trained":
            print(f"[pybullet-sim] no checkpoint at {ckpt}; using random controller", flush=True)
        policy_fn = _random_policy(env)
        label = "random"

    dt = 0.1
    ckpt_mtime: float = os.path.getmtime(ckpt) if os.path.exists(ckpt) else 0.0
    reload_check_interval = 50  # frames
    frame_count = 0
    while proc.poll() is None:
        # Hot-reload checkpoint if training produced a new one since we started.
        frame_count += 1
        if frame_count % reload_check_interval == 0 and policy == "trained" and os.path.exists(ckpt):
            mtime = os.path.getmtime(ckpt)
            if mtime > ckpt_mtime:
                try:
                    policy_fn = _trained_policy(ckpt)
                    ckpt_mtime = mtime
                    label = "trained"
                    print(f"[pybullet-sim] reloaded updated checkpoint for {env_id}", flush=True)
                except Exception as e:
                    print(f"[pybullet-sim] checkpoint reload failed: {e}", flush=True)
        actions = policy_fn(obs)
        obs, _, dones, _ = env.step(actions)
        if dones.all():
            obs = env.reset()
        payload = swarm_message(env)
        payload["env_id"] = env_id
        payload["policy"] = label
        covered = int(env.covered.sum())
        total = int(env.covered.size)
        payload["coverage"] = round(covered / total, 3) if total else 0.0
        line = json.dumps({"topic": "swarm", **payload})
        _broadcast_sim_sync_line(line)
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            break
        time.sleep(dt)


def _start_training(env_id: str, profile: str, timesteps: int) -> dict:
    global _JOBS
    with _LOCK:
        existing = _JOBS.get(env_id, {})
        if _is_job_running(existing):
            return {"ok": False, "error": f"training already running for {env_id}"}
        if existing.get("proc") is not None:
            existing["running"] = False

    cmd = [
        sys.executable,
        "-m",
        "swarm.train",
        "--env-id",
        env_id,
        "--profile",
        profile,
        "--timesteps",
        str(timesteps),
    ]
    proc = _popen(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _LOCK:
        existing = _JOBS.get(env_id, {})
        if _is_job_running(existing):
            _kill_proc(proc, wait=False)
            return {"ok": False, "error": f"training already running for {env_id}"}
        _JOBS[env_id] = {
            "env_id": env_id,
            "profile": profile,
            "running": True,
            "proc": proc,
            "status": "running",
            "started_at": time.time(),
        }

    threading.Thread(target=_tail_training, args=(env_id, proc), daemon=True).start()
    print(
        f"[train-service] started training env_id={env_id} profile={profile} "
        f"timesteps={timesteps} pid={proc.pid}",
        flush=True,
    )
    return {"ok": True, "env_id": env_id, "running": True}


def _stop_training(env_id: str | None) -> dict:
    global _JOBS
    with _LOCK:
        if not env_id:
            env_id = list(_JOBS.keys())[0] if _JOBS else None
        if not env_id or env_id not in _JOBS:
            return {"ok": False, "error": "no training job found"}
        job = _JOBS[env_id]
        proc: subprocess.Popen | None = job.get("proc")
        if not _is_job_running(job):
            job["running"] = False
            return {"ok": True, "env_id": env_id, "stopped": False}
        _kill_proc(proc)
        job["running"] = False
    return {"ok": True, "env_id": env_id, "stopped": True}


def _tail_sim(proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"[pybullet-sim] {line}", flush=True)
            continue
        if event.get("topic") == "pybullet_frame":
            _broadcast_sim_sync_line(line)
        else:
            print(f"[pybullet-sim] {line}", flush=True)

    code = proc.wait()
    with _LOCK:
        global _SIM_JOB
        if _SIM_JOB and _SIM_JOB.get("proc") is proc:
            _SIM_JOB["running"] = False
            _SIM_JOB["returncode"] = code


def _start_sim(env_id: str, policy: str, camera_mode: str, selected_drone: int) -> dict:
    global _SIM_JOB
    with _LOCK:
        existing = _SIM_JOB
        if existing and existing.get("proc") and existing["proc"].poll() is None:
            same_renderer = (
                existing.get("env_id") == env_id
                and existing.get("policy") == policy
                and existing.get("camera_mode") == camera_mode
                and int(existing.get("selected_drone", 0)) == int(selected_drone)
            )
            if same_renderer:
                return {
                    "ok": True,
                    "env_id": env_id,
                    "running": True,
                    "ws_url": f"ws://{HOST_HTTP}:{PORT_HTTP}/sim/ws",
                }
            _kill_proc(existing["proc"])
            existing["running"] = False

    renderer_python = os.environ.get("PYBULLET_PYTHON", "/opt/homebrew/bin/python3")
    cmd = [
        renderer_python,
        "-m",
        "swarm.pybullet_renderer",
        "--env-id",
        env_id,
    ]
    proc = _popen(
        cmd,
        cwd=REPO,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _LOCK:
        _SIM_JOB = {
            "env_id": env_id,
            "policy": policy,
            "camera_mode": camera_mode,
            "selected_drone": selected_drone,
            "running": True,
            "proc": proc,
            "started_at": time.time(),
        }

    threading.Thread(target=_tail_sim, args=(proc,), daemon=True).start()
    threading.Thread(target=_drive_sim_policy, args=(env_id, policy, proc), daemon=True).start()
    time.sleep(0.35)
    code = proc.poll()
    if code is not None:
        with _LOCK:
            if _SIM_JOB and _SIM_JOB.get("proc") is proc:
                _SIM_JOB["running"] = False
                _SIM_JOB["returncode"] = code
        return {
            "ok": False,
            "env_id": env_id,
            "error": f"PyBullet sim exited during startup with code {code}",
        }

    return {
        "ok": True,
        "env_id": env_id,
        "camera_mode": camera_mode,
        "selected_drone": selected_drone,
        "running": True,
        "ws_url": f"ws://{HOST_HTTP}:{PORT_HTTP}/sim/ws",
    }


def _stop_sim() -> dict:
    global _SIM_JOB
    with _LOCK:
        job = _SIM_JOB
        if not job or not job.get("proc") or job["proc"].poll() is not None:
            return {"ok": True, "stopped": False}
        _kill_proc(job["proc"])
        job["running"] = False
        return {"ok": True, "stopped": True, "env_id": job.get("env_id")}


# ── FastAPI app ────────────────────────────────────────────────────────────


def _cleanup_all(*, wait: bool = True) -> None:
    """Terminate all running jobs and subprocesses."""
    procs: list[subprocess.Popen] = []
    with _LOCK:
        for job in _JOBS.values():
            proc = job.get("proc")
            if proc is not None:
                procs.append(proc)
                job["running"] = False
        global _SIM_JOB
        if _SIM_JOB and _SIM_JOB.get("proc"):
            procs.append(_SIM_JOB["proc"])
            _SIM_JOB["running"] = False
    for proc in procs:
        _kill_proc(proc, wait=wait)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown for the FastAPI app."""
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_running_loop()
    print(f"[train-service] HTTP http://{HOST_HTTP}:{PORT_HTTP}")
    print(f"[train-service] WebSocket ws://127.0.0.1:8787/ws  topic=train")
    print(f"[train-service] PyBullet WebSocket ws://127.0.0.1:8787/sim/ws  topic=swarm/pybullet_frame")

    yield

    print("\n[train-service] shutting down...", flush=True)
    _cleanup_all()
    _MAIN_LOOP = None
    print("[train-service] stopped", flush=True)


app = FastAPI(title="Outcast Virus Train Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/train/status")
async def get_train_status(env_id: str = "") -> dict:
    """Get training status for a scenario."""
    with _LOCK:
        job = dict(_JOBS.get(env_id, {}))
    running = _is_job_running(job)
    if job and not running:
        job["running"] = False
        with _LOCK:
            if env_id in _JOBS:
                _JOBS[env_id]["running"] = False
    return {
        "env_id": env_id,
        "running": running,
        "status": job.get("status", "idle") if job else "idle",
        "last": job.get("last"),
    }


@app.get("/api/sim/status")
async def get_sim_status() -> dict:
    """Get PyBullet sim status."""
    with _LOCK:
        job = dict(_SIM_JOB or {})
        proc = job.get("proc")
        running = bool(proc and proc.poll() is None)
    return {
        "env_id": job.get("env_id"),
        "policy": job.get("policy"),
        "camera_mode": job.get("camera_mode", "observer"),
        "selected_drone": job.get("selected_drone", 0),
        "running": running,
        "returncode": job.get("returncode"),
        "ws_url": f"ws://{HOST_HTTP}:{PORT_HTTP}/sim/ws",
    }


@app.post("/api/train/start")
async def post_train_start(req: TrainStartRequest) -> dict:
    """Start training a scenario."""
    result = _start_training(req.env_id, req.profile, req.timesteps)
    if result.get("ok"):
        print(
            f"[train-service] POST /api/train/start accepted env_id={req.env_id} "
            f"profile={req.profile} timesteps={req.timesteps}",
            flush=True,
        )
    else:
        print(
            f"[train-service] POST /api/train/start rejected env_id={req.env_id}: "
            f"{result.get('error')}",
            flush=True,
        )
    return result


@app.post("/api/train/stop")
async def post_train_stop(req: TrainStopRequest) -> dict:
    """Stop training."""
    return _stop_training(req.env_id)


@app.post("/api/train/export")
async def post_train_export(req: TrainExportRequest) -> dict:
    """Export a trained policy to ONNX."""
    try:
        subprocess.run(
            [sys.executable, "-m", "swarm.export_onnx", "--env-id", req.env_id],
            cwd=REPO,
            check=True,
        )
        return {"ok": True, "env_id": req.env_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/sim/start")
async def post_sim_start(req: SimStartRequest) -> dict:
    """Start PyBullet sim."""
    if req.policy not in {"trained", "random"}:
        return {"ok": False, "error": "invalid policy"}
    if req.camera_mode not in {"observer", "chase", "fpv"}:
        return {"ok": False, "error": "invalid camera_mode"}
    return _start_sim(req.env_id, req.policy, req.camera_mode, req.selected_drone)


@app.post("/api/sim/stop")
async def post_sim_stop() -> dict:
    """Stop PyBullet sim."""
    return _stop_sim()


@app.websocket("/ws")
async def websocket_train(websocket: WebSocket):
    """Train WebSocket endpoint."""
    await websocket.accept()
    _WS_CLIENTS.add(websocket)
    global _NO_CLIENT_WARNED_AT
    _NO_CLIENT_WARNED_AT = 0.0
    # Late joiners (e.g. socket opened after POST /train/start) get the latest
    # event per job so the dashboard is not stuck waiting for the next update.
    with _LOCK:
        last_events = [
            job["last"]
            for job in _JOBS.values()
            if job.get("last")
        ]
    for event in last_events:
        try:
            await websocket.send_text(json.dumps(event))
        except Exception:
            break
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        _WS_CLIENTS.discard(websocket)


@app.websocket("/sim/ws")
async def websocket_sim(websocket: WebSocket):
    """Sim WebSocket endpoint."""
    await websocket.accept()
    _SIM_WS_CLIENTS.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        _SIM_WS_CLIENTS.discard(websocket)


def main() -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host=HOST_HTTP,
        port=PORT_HTTP,
        log_level="info",
        timeout_graceful_shutdown=2,
        ws_max_size=16 * 1024 * 1024,  # allow large PyBullet frames (>1 MB default)
    )
    server = uvicorn.Server(config)

    def _handle_exit(signum: int, frame: Any) -> None:
        print("\n[train-service] Ctrl+C — stopping...", flush=True)
        _cleanup_all(wait=False)
        server.handle_exit(signum, frame)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    server.run()


if __name__ == "__main__":
    main()
