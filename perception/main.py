"""
perception/main.py — CombatOS Targeting Loop

Controls:
  c  — confirm the current proposed candidate
  f  — lock follow mode on the proposed candidate
  r  — release follow mode
  q  — quit
"""
from __future__ import annotations

import sys

import cv2

import config
from bus import BusPublisher
from detector import Detector
from priority import top_candidate
from tracker import TargetTracker
from visualizer import draw


def main() -> None:
    source = config.VIDEO_SOURCE
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"[perception] Cannot open source: {source}")

    detector  = Detector()
    tracker   = TargetTracker()
    publisher = BusPublisher()

    try:
        publisher.connect()
        print("[perception] Connected to bus")
    except Exception as e:
        print(f"[perception] Bus unavailable ({e}), running in local-only mode")
        publisher = None

    print("[perception] Running. Press q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.run(frame)
        objects    = tracker.update(detections)
        candidate  = top_candidate(objects)

        if publisher:
            publisher.publish(objects)

        frame = draw(frame, objects)
        cv2.imshow("CombatOS — Targeting", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and candidate:
            tracker.confirm_target(candidate.id)
            print(f"[perception] CONFIRMED target {candidate.id}")
        elif key == ord("f") and candidate:
            tracker.lock_follow(candidate.id)
            print(f"[perception] FOLLOW LOCK → {candidate.id}")
        elif key == ord("r"):
            tracker.release_follow()
            print("[perception] Follow released")

    cap.release()
    cv2.destroyAllWindows()
    if publisher:
        publisher.close()


if __name__ == "__main__":
    main()
