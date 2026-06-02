"""
perception/slam_sim.py -- Standalone monocular vSLAM publisher.

The visual-odometry brains live in vo.py. main.py drives them inline on the
SAME frames it feeds to YOLO, which is the synced path you normally want.

This script is the SLAM-only fallback: it runs the identical VO against an
arbitrary clip (e.g. a EuRoC sequence) WITHOUT YOLO. Because it opens its own
VideoCapture it is NOT frame-synced with a separately running main.py -- use
main.py for the synced pipeline; use this to exercise SLAM in isolation.

    python -m combatos.orchestrator         # bus
    cd perception && python slam_sim.py     # SLAM only

Honors the same SLAM_SIM_* knobs in config.py.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from typing import Any

import cv2

import config
import vo


def _dbg(msg: str) -> None:
    print(f"[slam_sim][{time.time():.3f}] {msg}", flush=True)


class SlamBusPublisher:
    """Fire-and-forget WebSocket publisher for control + image bus topics."""

    def __init__(self) -> None:
        self._uri = f"ws://{config.WS_HOST}:{config.WS_PORT}"
        self._image_uri = f"ws://{config.WS_HOST}:{config.IMAGE_WS_PORT}"
        self._loop = asyncio.new_event_loop()
        self._ws = None
        self._image_ws = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="slam-bus-io")

    def connect(self) -> None:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result(timeout=5)

    def close(self) -> None:
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(timeout=2)
        if self._image_ws is not None:
            asyncio.run_coroutine_threadsafe(self._image_ws.close(), self._loop).result(timeout=2)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        import websockets

        self._ws = await websockets.connect(self._uri)
        try:
            self._image_ws = await websockets.connect(self._image_uri)
        except OSError as exc:
            print(f"[slam_sim] image bus unavailable ({exc}); slam topics still publish", flush=True)
            self._image_ws = None

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps({"topic": topic, **payload})), self._loop)

    def publish_image(self, topic: str, payload: dict[str, Any]) -> None:
        if self._image_ws is not None:
            asyncio.run_coroutine_threadsafe(self._image_ws.send(json.dumps({"topic": topic, **payload})), self._loop)


def main() -> None:
    source = config.VIDEO_SOURCE
    _dbg(f"opening video source: {source!r}")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"[slam_sim] Cannot open source: {source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if config.PROC_WIDTH > 0 and config.PROC_WIDTH < src_w:
        proc_w = config.PROC_WIDTH
        proc_h = round(src_h * proc_w / src_w)
    else:
        proc_w, proc_h = src_w, src_h
    _dbg(f"video {src_w}x{src_h} -> processing {proc_w}x{proc_h}")

    # Standalone: there is no YOLO targeting feed, so publish camera_frame too
    # unless the operator explicitly disabled it.
    streamer = vo.SlamStreamer(proc_w, proc_h, publish_camera=True)

    _dbg("connecting to bus ...")
    publisher: SlamBusPublisher | None = SlamBusPublisher()
    try:
        publisher.connect()
        _dbg("connected to bus")
    except Exception as exc:  # noqa: BLE001 -- bus optional
        _dbg(f"bus unavailable ({type(exc).__name__}: {exc!r}); running local-only")
        publisher = None

    proc_period = 1.0 / config.SLAM_SIM_PROC_FPS if config.SLAM_SIM_PROC_FPS > 0 else 0.0
    if config.SLAM_SIM_SHOW:
        cv2.namedWindow("CombatOS - vSLAM sidecar", cv2.WINDOW_NORMAL)

    _dbg("entering main loop")
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if config.SLAM_SIM_LOOP:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                _dbg("end of stream")
                break

            if (proc_w, proc_h) != (src_w, src_h):
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            if publisher:
                streamer.process_and_publish(frame, time.time(), publisher.publish, publisher.publish_image)
            else:
                streamer.vo.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            if config.SLAM_SIM_SHOW:
                cv2.imshow("CombatOS - vSLAM sidecar", vo.annotate_frame(frame, streamer.vo))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if proc_period:
                sleep = proc_period - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        cap.release()
        if config.SLAM_SIM_SHOW:
            cv2.destroyAllWindows()
        if publisher:
            publisher.close()
        _dbg("stopped")


if __name__ == "__main__":
    main()
