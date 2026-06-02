"""
perception/main.py -- Outcast Virus Targeting Loop

Operator flow (state machine):
  PROPOSED  --[F]--> FOLLOWED  --[C]--> CONFIRMED
            <--[R]--           <--[R]--

  F  -- follow:   lock onto proposed candidate (or re-lock on confirmed target if visible)
  C  -- confirm:  lock in the followed target (requires a followed target)
  R  -- release:  step back one level (Confirmed→Followed, Followed→Proposed)
  U  -- full reset: clear confirmed + follow + ReID gallery
  Q  -- quit
"""
from __future__ import annotations

import os
import queue
import sys
import time

# Must be set before numpy/scipy/norfair import -- prevents OpenBLAS thread-init
# deadlock on Windows (common with filterpy/scipy pulled in by norfair).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("[import] stdlib ok", flush=True)

import cv2
print("[import] cv2 ok", flush=True)

import numpy as np
print("[import] numpy ok", flush=True)

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

import vo
print("[import] vo ok", flush=True)


def _dbg(msg: str) -> None:
    print(f"[perception][{time.time():.3f}] {msg}", flush=True)

WINDOW = "Outcast Virus - Targeting"


def _drain_commands(
    publisher: BusPublisher,
    tracker: TargetTracker,
    gallery: ReIDGallery,
) -> None:
    """Process all queued dashboard commands without blocking.

    Commands arrive from the WebSocket receive loop and are batched into
    publisher.commands (SimpleQueue) between frames.  Applied in order
    so a rapid follow+confirm sequence from the UI works correctly.
    """
    while True:
        try:
            cmd = publisher.commands.get_nowait()
        except queue.Empty:
            break
        action = cmd.get("action")
        tid    = cmd.get("track_id")
        if action == "confirm" and tid is not None:
            if tracker.confirm_target(tid):
                gallery.confirm(tid)
                print(f"[perception] CONFIRMED (remote) target {tid}")
        elif action == "follow" and tid is not None:
            if tracker.lock_follow(tid):
                print(f"[perception] FOLLOW LOCK (remote) → {tid}")
        elif action == "unconfirm":
            tracker.unconfirm_all()
            gallery.clear()
            print("[perception] Target unconfirmed (remote)")
        elif action in ("release", "release_follow"):
            released_id = tracker.release()
            if released_id is not None:
                gallery.release_confirm(released_id)
            print("[perception] Released (remote)")


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


def main() -> None:
    source = config.VIDEO_SOURCE
    _dbg(f"opening video source: {source!r}")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"[perception] Cannot open source: {source}")
    _dbg(f"video opened  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
         f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
         f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}  "
         f"frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

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
    publisher = BusPublisher()
    try:
        publisher.connect()
        _dbg("connected to bus")
    except Exception as e:
        _dbg(f"bus unavailable ({type(e).__name__}: {e!r}), running in local-only mode")
        publisher = None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if config.PROC_WIDTH > 0 and config.PROC_WIDTH < src_w:
        proc_w = config.PROC_WIDTH
        proc_h = round(src_h * proc_w / src_w)
    else:
        proc_w, proc_h = src_w, src_h
    _dbg(f"processing at {proc_w}x{proc_h} (source {src_w}x{src_h})")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, proc_w, proc_h)

    # Monocular vSLAM runs inline on the same frames YOLO sees, so the targeting
    # feed and the SLAM panels are driven by one capture in lockstep. Needs the
    # bus, so skip it in local-only mode.
    slam_streamer = vo.SlamStreamer(proc_w, proc_h) if publisher else None
    if slam_streamer:
        _dbg("monocular vSLAM enabled — publishing slam_* topics in sync with detections")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    _dbg("entering main loop")
    frame_n        = 0
    last_dets: list = []  # reused on non-detect frames

    recording    = False
    rec_dir      = None
    writer_raw   = None
    writer_hud   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            _dbg(f"cap.read() returned False at frame {frame_n} -- end of stream or read error")
            break

        if src_w != proc_w:
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        if frame_n % config.DETECT_EVERY == 0:
            last_dets = detector.run(frame)
        detections = last_dets

        objects    = tracker.update(detections)
        active_ids = {o.id for o in objects}

        # Drain dashboard commands (non-blocking, batched since last frame)
        if publisher:
            _drain_commands(publisher, tracker, gallery)
            tracker.refresh_flags(objects)

        # ---- passive sampling + ReID pass ----------------------------
        confirmed_visible = any(o.confirmed for o in objects)

        for obj in objects:
            c = _crop(frame, obj.bbox)
            gallery.sample(obj.id, c)

            if not confirmed_visible and not obj.confirmed and not obj.is_primary and gallery.active:
                matched_id = gallery.match(obj.id, c)
                if matched_id is not None:
                    tracker.reassign_confirmed(obj.id)
                    gallery.reassign(obj.id)
                    break

        gallery.prune(active_ids)
        # --------------------------------------------------------------

        candidate = buffer.update(objects)

        followed_obj  = next((o for o in objects if o.is_primary), None)
        confirmed_obj = next((o for o in objects if o.confirmed), None)

        if publisher:
            publisher.publish(objects, proc_w, proc_h,
                              candidate_id=candidate.id if candidate else None)

        raw_frame = frame.copy()

        # Same frame, same instant -> SLAM stays synced with the targeting feed.
        if slam_streamer:
            slam_streamer.process_and_publish(
                raw_frame, time.time(), publisher.publish_topic, publisher.publish_image_topic
            )

        frame = draw(frame, objects, candidate)

        if recording:
            writer_raw.write(raw_frame)
            writer_hud.write(frame)

        if publisher and frame_n % config.FPV_INTERVAL == 0:
            publisher.publish_frame(raw_frame, topic="fpv_raw", quality=config.FPV_QUALITY)

        cv2.imshow(WINDOW, frame)

        frame_n += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not recording:
                rec_dir    = _next_recording_dir()
                writer_raw = _make_writer(rec_dir, "raw.mp4", proc_w, proc_h, src_fps)
                writer_hud = _make_writer(rec_dir, "hud.mp4", proc_w, proc_h, src_fps)
                recording  = True
                _dbg(f"recording started → {rec_dir}")
            else:
                writer_raw.release()
                writer_hud.release()
                recording = False
                _dbg(f"recording saved → {rec_dir}")
        elif key == ord("f"):
            # Follow confirmed target if it's visible; otherwise follow proposed candidate
            target = confirmed_obj or candidate
            if target:
                if tracker.lock_follow(target.id):
                    print(f"[perception] FOLLOW LOCK → {target.id}")
        elif key == ord("c"):
            # Confirm requires a followed target -- C on a bare candidate is not allowed
            if followed_obj:
                if tracker.confirm_target(followed_obj.id):
                    gallery.confirm(followed_obj.id)
                    print(f"[perception] CONFIRMED target {followed_obj.id} -- {gallery.debug_info()}")
            else:
                print("[perception] Press F first to follow a target before confirming")
        elif key == ord("u"):
            tracker.unconfirm_all()
            gallery.clear()
            print("[perception] Target unconfirmed")
        elif key == ord("r"):
            released_id = tracker.release()
            if released_id is not None:
                gallery.release_confirm(released_id)
            print("[perception] Released")

    if recording:
        writer_raw.release()
        writer_hud.release()
        _dbg(f"recording saved → {rec_dir}")

    cap.release()
    cv2.destroyAllWindows()
    if publisher:
        publisher.close()


if __name__ == "__main__":
    main()
