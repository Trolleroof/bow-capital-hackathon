"""
perception/rosmain.py -- CombatOS Targeting Loop (ROS2 input)

Drop-in equivalent of main.py.  The only difference is the frame source:
instead of cv2.VideoCapture the frames arrive via a ROS2 subscription to
/camera/image_raw (or whatever ROS_CAMERA_TOPIC is set to).

All processing is identical: YOLO detection, Norfair tracking, ReID,
bus publishing, FPV stream, cv2 display window, recording, and the full
operator key-control state machine.

Usage:
    source /opt/ros/<distro>/setup.bash   # e.g. jazzy or humble
    cd perception
    python rosmain.py           # run after camera_node.py is publishing

Operator flow (state machine) -- same as main.py:
  PROPOSED  --[F]--> FOLLOWED  --[C]--> CONFIRMED
            <--[R]--           <--[R]--

  F  -- follow:   lock onto proposed candidate (or re-lock confirmed target)
  C  -- confirm:  lock in the followed target (requires a followed target)
  R  -- release:  step back one level
  U  -- full reset: clear confirmed + follow + ReID gallery
  Q  -- quit
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("[import] stdlib ok", flush=True)

import cv2
print("[import] cv2 ok", flush=True)

import numpy as np
print("[import] numpy ok", flush=True)

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage, CompressedImage as RosCompressedImage
from cv_bridge import CvBridge
print("[import] rclpy/cv_bridge ok", flush=True)

import config
print("[import] config ok", flush=True)

from bus import BusPublisher
print("[import] bus ok", flush=True)

from detector import Detector
print("[import] detector ok", flush=True)

from priority import CandidateBuffer
print("[import] priority ok", flush=True)

from reid import ReIDGallery
print("[import] reid ok", flush=True)

from tracker import TargetTracker
print("[import] tracker ok", flush=True)

from visualizer import draw
print("[import] visualizer ok", flush=True)


def _dbg(msg: str) -> None:
    print(f"[rosmain][{time.time():.3f}] {msg}", flush=True)

WINDOW = "CombatOS - Targeting"


# ── Helpers (identical to main.py) ───────────────────────────────────────────

def _drain_commands(
    publisher: BusPublisher,
    tracker: TargetTracker,
    gallery: ReIDGallery,
) -> None:
    while True:
        try:
            cmd = publisher.commands.get_nowait()
        except queue.Empty:
            break
        action = cmd.get("action")
        tid    = cmd.get("track_id")
        if action == "confirm" and tid is not None:
            tracker.confirm_target(tid)
            gallery.confirm(tid)
            print(f"[rosmain] CONFIRMED (remote) target {tid}")
        elif action == "follow" and tid is not None:
            tracker.lock_follow(tid)
            print(f"[rosmain] FOLLOW LOCK (remote) → {tid}")
        elif action == "unconfirm":
            tracker.unconfirm_all()
            gallery.clear()
            print("[rosmain] Target unconfirmed (remote)")
        elif action in ("release", "release_follow"):
            released_id = tracker.release()
            if released_id is not None:
                gallery.release_confirm(released_id)
            print("[rosmain] Released (remote)")


def _next_recording_dir() -> str:
    base = os.path.join("footage", "recordings")
    os.makedirs(base, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.isdigit()
    ]
    idx = max(existing, default=0) + 1
    path = os.path.join(base, f"{idx:03d}")
    os.makedirs(path)
    return path


def _make_writer(path: str, name: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(os.path.join(path, name), fourcc, fps, (w, h))


def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    return frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class PerceptionNode(Node):
    """
    Subscribes to /camera/image_raw and runs the full main.py pipeline on
    each frame.  State that main.py kept as local loop variables lives here
    as instance attributes so it persists across callbacks.
    """

    def __init__(
        self,
        publisher: BusPublisher | None,
        detector: Detector,
        tracker: TargetTracker,
        buffer: CandidateBuffer,
        gallery: ReIDGallery,
    ) -> None:
        super().__init__("combatos_perception")
        self._bridge    = CvBridge()
        self._publisher = publisher
        self._detector  = detector
        self._tracker   = tracker
        self._buffer    = buffer
        self._gallery   = gallery

        # Per-frame state (mirrors main.py local vars)
        self._frame_n       = 0
        self._last_dets:list= []
        self._recording     = False
        self._rec_dir: str | None       = None
        self._writer_raw: cv2.VideoWriter | None = None
        self._writer_hud: cv2.VideoWriter | None = None
        self._src_fps       = 30.0          # updated on first frame

        # Objects visible on the last frame -- read by handle_key on main thread
        self._followed_obj  = None
        self._confirmed_obj = None
        self._candidate     = None

        # Latest frames for the cv2 display loop (written in callback, read in main)
        self._display_frame: np.ndarray | None = None
        self._display_lock  = threading.Lock()

        # Window created lazily on first frame so we know the dimensions
        self._window_ready  = False

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._compressed = config.ROS_CAMERA_TOPIC.endswith("/compressed")
        if self._compressed:
            self._sub = self.create_subscription(
                RosCompressedImage, config.ROS_CAMERA_TOPIC, self._on_frame, image_qos
            )
        else:
            self._sub = self.create_subscription(
                RosImage, config.ROS_CAMERA_TOPIC, self._on_frame, image_qos
            )
        self.get_logger().info(f"Subscribing to {config.ROS_CAMERA_TOPIC}")

    # ── Frame callback (runs in ROS executor thread) ─────────────────────────

    def _on_frame(self, msg) -> None:
        if self._compressed:
            frame = self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w  = frame.shape[:2]

        # Lazy window + fps init on first frame
        if not self._window_ready:
            self._src_fps = getattr(msg, "_fps", 30.0)  # best-effort; camera_node sets this
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, w, h)
            self._window_ready = True
            _dbg(f"first frame received: {w}x{h}")

        # ── Detection ────────────────────────────────────────────────────────
        if self._frame_n % config.DETECT_EVERY == 0:
            self._last_dets = self._detector.run(frame)

        # ── Tracking ─────────────────────────────────────────────────────────
        objects    = self._tracker.update(self._last_dets)
        active_ids = {o.id for o in objects}

        # ── Drain dashboard commands (non-blocking) ───────────────────────────
        if self._publisher:
            _drain_commands(self._publisher, self._tracker, self._gallery)
            self._tracker.refresh_flags(objects)

        # ── Passive ReID sampling + active matching ───────────────────────────
        confirmed_visible = any(o.confirmed for o in objects)
        for obj in objects:
            c = _crop(frame, obj.bbox)
            self._gallery.sample(obj.id, c)
            if not confirmed_visible and not obj.confirmed and not obj.is_primary and self._gallery.active:
                matched_id = self._gallery.match(obj.id, c)
                if matched_id is not None:
                    self._tracker.reassign_confirmed(obj.id)
                    self._gallery.reassign(obj.id)
                    break

        self._gallery.prune(active_ids)

        # ── Candidate priority ────────────────────────────────────────────────
        candidate = self._buffer.update(objects)

        # Cache for key handler (main thread reads these)
        self._followed_obj  = next((o for o in objects if o.is_primary), None)
        self._confirmed_obj = next((o for o in objects if o.confirmed),  None)
        self._candidate     = candidate

        # ── Bus publish (detections JSON) ─────────────────────────────────────
        if self._publisher:
            self._publisher.publish(objects, w, h, candidate_id=candidate.id if candidate else None)

        # ── Draw HUD ──────────────────────────────────────────────────────────
        raw_frame = frame.copy()
        annotated = draw(frame, objects, candidate)

        # ── Recording ─────────────────────────────────────────────────────────
        if self._recording:
            self._writer_raw.write(raw_frame)
            self._writer_hud.write(annotated)

        # ── FPV stream ────────────────────────────────────────────────────────
        if self._publisher and self._frame_n % config.FPV_INTERVAL == 0:
            self._publisher.publish_frame(raw_frame, topic="fpv_raw", quality=config.FPV_QUALITY)

        # Hand annotated frame to the display loop
        with self._display_lock:
            self._display_frame = annotated

        self._frame_n += 1

    # ── Public accessors (called from main thread) ────────────────────────────

    def get_display_frame(self) -> np.ndarray | None:
        with self._display_lock:
            return self._display_frame

    def handle_key(self, key: int) -> bool:
        """Process a cv2.waitKey result.  Returns True when the loop should exit."""
        if key == ord("q"):
            return True

        elif key == ord(" "):
            if not self._recording:
                if not self._window_ready:
                    return False
                frame = self.get_display_frame()
                if frame is None:
                    return False
                h, w = frame.shape[:2]
                self._rec_dir    = _next_recording_dir()
                self._writer_raw = _make_writer(self._rec_dir, "raw.mp4", w, h, self._src_fps)
                self._writer_hud = _make_writer(self._rec_dir, "hud.mp4", w, h, self._src_fps)
                self._recording  = True
                _dbg(f"recording started → {self._rec_dir}")
            else:
                self._writer_raw.release()
                self._writer_hud.release()
                self._recording = False
                _dbg(f"recording saved → {self._rec_dir}")

        elif key == ord("f"):
            target = self._confirmed_obj or self._candidate
            if target:
                self._tracker.lock_follow(target.id)
                print(f"[rosmain] FOLLOW LOCK → {target.id}")

        elif key == ord("c"):
            if self._followed_obj:
                self._tracker.confirm_target(self._followed_obj.id)
                self._gallery.confirm(self._followed_obj.id)
                print(f"[rosmain] CONFIRMED target {self._followed_obj.id}"
                      f" -- {self._gallery.debug_info()}")
            else:
                print("[rosmain] Press F first to follow a target before confirming")

        elif key == ord("u"):
            self._tracker.unconfirm_all()
            self._gallery.clear()
            print("[rosmain] Target unconfirmed")

        elif key == ord("r"):
            released_id = self._tracker.release()
            if released_id is not None:
                self._gallery.release_confirm(released_id)
            print("[rosmain] Released")

        return False

    def stop_recording(self) -> None:
        if self._recording:
            self._writer_raw.release()
            self._writer_hud.release()
            _dbg(f"recording saved → {self._rec_dir}")

    def destroy_node(self) -> None:
        self.stop_recording()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _dbg(f"camera topic: {config.ROS_CAMERA_TOPIC}")

    _dbg("loading detector (YOLO) ...")
    detector = Detector()
    _dbg("detector ready")

    _dbg("loading tracker ...")
    tracker = TargetTracker()
    _dbg("tracker ready")

    buffer = CandidateBuffer()

    _dbg("loading ReID gallery (may download weights on first run) ...")
    gallery = ReIDGallery()
    _dbg("ReID gallery ready")

    _dbg("connecting to WebSocket bus ...")
    publisher: BusPublisher | None = BusPublisher()
    try:
        publisher.connect()
        _dbg("connected to bus")
    except Exception as e:
        _dbg(f"bus unavailable ({type(e).__name__}: {e!r}), running in local-only mode")
        publisher = None

    rclpy.init()
    node = PerceptionNode(publisher, detector, tracker, buffer, gallery)
    _dbg("ROS2 node created; waiting for frames on " + config.ROS_CAMERA_TOPIC)

    try:
        while rclpy.ok():
            # Spin long enough to drain pending callbacks, but keep the main
            # thread free to handle the cv2 window every ~10 ms.
            rclpy.spin_once(node, timeout_sec=0.01)

            frame = node.get_display_frame()
            if frame is not None:
                cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF and node.handle_key(key):
                break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        if publisher:
            publisher.close()


if __name__ == "__main__":
    main()
