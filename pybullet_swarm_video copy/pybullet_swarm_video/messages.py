from __future__ import annotations

import base64

import cv2
import numpy as np


def make_frame_id(drone_id: int, seq: int) -> str:
    return f"drone-{drone_id:02d}-frame-{seq:06d}"


def encode_jpeg_from_rgb(frame: np.ndarray, quality: int = 80) -> str:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode RGB frame as JPEG")
    return base64.b64encode(buf).decode("ascii")


def encode_jpeg_from_bgr(frame: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode BGR frame as JPEG")
    return base64.b64encode(buf).decode("ascii")


def decode_jpeg_to_bgr(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode("ascii"))
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("failed to decode JPEG frame")
    return frame


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
