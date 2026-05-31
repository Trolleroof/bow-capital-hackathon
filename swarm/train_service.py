"""Training API + WebSocket bridge for the CombatOS gym UI.

HTTP (default :8787):
  POST /api/train/start   {"env_id","profile","timesteps"?}
  POST /api/train/stop    {"env_id"?}
  POST /api/train/export  {"env_id"}
  GET  /api/train/status?env_id=

WebSocket (:8766):
  Broadcasts `topic: "train"` JSON lines (same shape as swarm/train.py stdout).

Run:
  uv run --project swarm python -m swarm.train_service

Then in another terminal:
  cd frontend && bun dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import websockets

from swarm.train import checkpoint_paths

HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)

HOST_HTTP = "127.0.0.1"
PORT_HTTP = 8787
HOST_WS = "0.0.0.0"
PORT_WS = 8766

_WS_CLIENTS: set[Any] = set()
_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_WS_LOOP: asyncio.AbstractEventLoop | None = None
_WS_THREAD: threading.Thread | None = None


def _ensure_ws_loop() -> asyncio.AbstractEventLoop:
    global _WS_LOOP, _WS_THREAD
    if _WS_LOOP is not None:
        return _WS_LOOP

    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _WS_THREAD = threading.Thread(target=_run, daemon=True)
    _WS_THREAD.start()
    _WS_LOOP = loop
    return loop


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


async def _ws_register(ws):
    _WS_CLIENTS.add(ws)
    try:
        await ws.wait_closed()
    finally:
        _WS_CLIENTS.discard(ws)


async def _ws_broadcast_line(line: str) -> None:
    if not _WS_CLIENTS:
        return
    dead = []
    for ws in list(_WS_CLIENTS):
        try:
            await ws.send(line)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _WS_CLIENTS.discard(ws)


def _broadcast_sync_line(line: str) -> None:
    loop = _ensure_ws_loop()
    asyncio.run_coroutine_threadsafe(_ws_broadcast_line(line), loop)


def _tail_training(env_id: str, proc: subprocess.Popen) -> None:
    """Read train.py stdout (NDJSON) and fan out to WebSocket clients."""
    assert proc.stdout is not None
    last: dict[str, Any] = {}
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("topic") != "train":
            continue
        last = event
        _broadcast_sync_line(line)
        with _LOCK:
            if env_id in _JOBS:
                _JOBS[env_id]["last"] = event

    code = proc.wait()
    with _LOCK:
        job = _JOBS.get(env_id, {})
        job["returncode"] = code
        job["running"] = False
        phase = last.get("phase")
        if phase in {"final", "checkpoint"} and code == 0:
            job["status"] = "completed"
        elif code == 0:
            job["status"] = "completed"
        else:
            job["status"] = "failed"
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
            _broadcast_sync_line(json.dumps(exported))
        except Exception as exc:
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


def _start_training(env_id: str, profile: str, timesteps: int) -> dict:
    with _LOCK:
        job = _JOBS.get(env_id)
        if job and job.get("running"):
            return {"ok": False, "error": "training already running for this env_id"}

    paths = checkpoint_paths(env_id)
    os.makedirs(paths["dir"], exist_ok=True)
    open(paths["events"], "w", encoding="utf-8").close()

    cmd = [
        sys.executable,
        "-m",
        "swarm.train",
        "--env-id",
        env_id,
        "--profile",
        profile,
        f"--timesteps={timesteps}",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _LOCK:
        _JOBS[env_id] = {
            "running": True,
            "status": "running",
            "proc": proc,
            "last": None,
            "profile": profile,
            "timesteps": timesteps,
        }

    threading.Thread(target=_tail_training, args=(env_id, proc), daemon=True).start()
    return {"ok": True, "env_id": env_id, "timesteps": timesteps}


def _stop_training(env_id: str | None) -> dict:
    with _LOCK:
        targets = [env_id] if env_id else list(_JOBS.keys())
        stopped = []
        for eid in targets:
            job = _JOBS.get(eid)
            if not job or not job.get("running"):
                continue
            proc: subprocess.Popen = job["proc"]
            proc.terminate()
            job["running"] = False
            job["status"] = "stopped"
            stopped.append(eid)
    return {"ok": True, "stopped": stopped}


class TrainAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: D102
        return

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/train/status":
            _json_response(self, 404, {"error": "not found"})
            return
        qs = parse_qs(parsed.query)
        env_id = (qs.get("env_id") or [""])[0]
        with _LOCK:
            job = _JOBS.get(env_id, {})
        _json_response(
            self,
            200,
            {
                "env_id": env_id,
                "running": bool(job.get("running")),
                "status": job.get("status", "idle"),
                "last": job.get("last"),
            },
        )

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        body = _read_body(self)

        if parsed.path == "/api/train/start":
            env_id = body.get("env_id", "search-and-interdict")
            profile = body.get("profile", "combat")
            timesteps = int(body.get("timesteps", 12_000))
            result = _start_training(env_id, profile, timesteps)
            _json_response(self, 200 if result.get("ok") else 409, result)
            return

        if parsed.path == "/api/train/stop":
            result = _stop_training(body.get("env_id"))
            _json_response(self, 200, result)
            return

        if parsed.path == "/api/train/export":
            env_id = body.get("env_id", "search-and-interdict")
            try:
                subprocess.run(
                    [sys.executable, "-m", "swarm.export_onnx", "--env-id", env_id],
                    cwd=REPO,
                    check=True,
                )
                _json_response(self, 200, {"ok": True, "env_id": env_id})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        _json_response(self, 404, {"error": "not found"})


async def _ws_main(host: str, port: int) -> None:
    async with websockets.serve(_ws_register, host, port):
        print(f"[train-service] WebSocket ws://{host}:{port}  topic=train")
        await asyncio.Future()


def probe_train_api(host: str = HOST_HTTP, port: int = PORT_HTTP) -> bool:
    """True if something is already serving our status endpoint."""
    try:
        with urlopen(f"http://{host}:{port}/api/train/status?env_id=", timeout=1.5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return isinstance(body, dict) and "status" in body
    except (URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="CombatOS training API + WS bridge")
    p.add_argument("--http-host", default=HOST_HTTP)
    p.add_argument("--http-port", type=int, default=PORT_HTTP)
    p.add_argument("--ws-host", default=HOST_WS)
    p.add_argument("--ws-port", type=int, default=PORT_WS)
    args = p.parse_args()

    if probe_train_api(args.http_host, args.http_port):
        print(
            f"[train-service] already running at http://{args.http_host}:{args.http_port} "
            f"(ws://127.0.0.1:{args.ws_port}) — reusing existing process"
        )
        return

    try:
        httpd = ThreadingHTTPServer((args.http_host, args.http_port), TrainAPIHandler)
    except OSError as exc:
        if exc.errno == 48 and probe_train_api(args.http_host, args.http_port):
            print(
                f"[train-service] port {args.http_port} in use by existing train service — ok"
            )
            return
        if exc.errno == 48:
            print(
                f"[train-service] ERROR: port {args.http_port} in use by another program.\n"
                f"  Free it: lsof -ti :{args.http_port} | xargs kill\n"
                f"  Or reuse if it's train_service from an earlier bun dev.",
                file=sys.stderr,
            )
        raise

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[train-service] HTTP http://{args.http_host}:{args.http_port}")

    try:
        asyncio.run(_ws_main(args.ws_host, args.ws_port))
    except OSError as exc:
        if exc.errno == 48:
            print(
                f"[train-service] ERROR: WebSocket port {args.ws_port} in use.\n"
                f"  Free it: lsof -ti :{args.ws_port} | xargs kill",
                file=sys.stderr,
            )
        raise
    except KeyboardInterrupt:
        print("\n[train-service] stopped")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
