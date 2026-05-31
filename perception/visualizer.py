"""HUD overlay -- corner brackets, scan line, status panel."""
from __future__ import annotations

import time
from typing import Protocol, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from tracker import TrackedObject


class _OverlayObject(Protocol):
    id: int
    cls: str
    conf: float
    bbox: list[int]
    has_face: bool
    confirmed: bool
    is_primary: bool

_C = {
    "troop":     (0,   220, 255),
    "vehicle":   (0,   160, 255),
    "ugv":       (30,  200, 120),
    "aerial":    (255, 240,  50),
    "default":   (140, 140, 140),
    "confirmed": (0,   255,  80),
    "primary":   (60,   60, 255),
    "candidate": (200,  80, 255),
    "white":     (220, 220, 220),
}

_FONT = cv2.FONT_HERSHEY_PLAIN


def draw(
    frame: np.ndarray,
    objects: list[_OverlayObject],
    candidate: _OverlayObject | None,
) -> np.ndarray:
    for obj in objects:
        color = _pick_color(obj, candidate)
        state = _state(obj, candidate)
        _draw_target(frame, obj, color, state)
    _draw_hud(frame, objects, candidate)
    return frame


# ---------------------------------------------------------------------------
# Per-target
# ---------------------------------------------------------------------------

def _draw_target(frame, obj, color, state):
    x, y, w, h = obj.bbox
    cx = x + w // 2

    thick     = 2 if state in ("confirmed", "primary") else 1
    frac      = 0.28 if state == "confirmed" else 0.22
    _corners(frame, obj.bbox, color, thickness=thick, length_frac=frac)

    # centre dot for confirmed/primary
    if state in ("confirmed", "primary"):
        cv2.circle(frame, (x + w//2, y + h//2), 3, color, -1, cv2.LINE_AA)

    # label
    if state == "confirmed":
        tag = f"TGT-{obj.id:02d}  {obj.cls.upper()}  {obj.conf:.0%}"
        if obj.has_face: tag += "  FACE"
    elif state == "primary":
        tag = f"LOCK  {obj.id:02d}  {obj.cls.upper()}"
    elif state == "candidate":
        tag = f"{obj.id:02d}  {obj.cls.upper()}"
    else:
        tag = f"{obj.id:02d} {obj.cls[0].upper()}"

    _label(frame, tag, cx, y - 6, color, bold=(state == "confirmed"))


# ---------------------------------------------------------------------------
# HUD panel
# ---------------------------------------------------------------------------

def _draw_hud(frame, objects, candidate):
    fh, fw = frame.shape[:2]
    t = time.time()

    followed  = next((o for o in objects if o.is_primary and not o.confirmed), None)
    confirmed = next((o for o in objects if o.confirmed), None)

    lines = [
        (f"TRACKS    {len(objects):02d}",                                          _C["white"]),
        (f"CANDIDATE  {'--' if not candidate else f'{candidate.id:02d} {candidate.cls.upper()}'}",
         _C["candidate"] if candidate else _C["white"]),
        (f"FOLLOW    {'--' if not followed  else f'{followed.id:02d}'}  [F]",
         _C["primary"]   if followed  else _C["white"]),
        (f"CONFIRMED  {'--' if not confirmed else f'{confirmed.id:02d} {confirmed.cls.upper()}'}  [C]",
         _C["confirmed"] if confirmed else _C["white"]),
    ]

    pad, lh = 10, 18
    panel_h = pad * 2 + len(lines) * lh
    panel_w = 250
    px, py  = 10, fh - panel_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (px, py), (px + panel_w, py), (0, 200, 80), 1, cv2.LINE_AA)

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (px + pad, py + pad + (i + 1) * lh),
                    _FONT, 0.85, color, 1, cv2.LINE_AA)

    # top-right timestamp
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (fw - 70, 18), _FONT, 0.85, (80, 80, 80), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _corners(frame, bbox, color, thickness=1, length_frac=0.22):
    x, y, w, h = bbox
    lx = max(8, int(w * length_frac))
    ly = max(8, int(h * length_frac))
    for ox, oy, sx, sy in [
        (x,   y,   1,  1),
        (x+w, y,  -1,  1),
        (x,   y+h, 1, -1),
        (x+w, y+h,-1, -1),
    ]:
        cv2.line(frame, (ox, oy), (ox + sx*lx, oy),        color, thickness, cv2.LINE_AA)
        cv2.line(frame, (ox, oy), (ox,         oy + sy*ly), color, thickness, cv2.LINE_AA)


def _label(frame, text, cx, top_y, color, bold=False):
    scale = 0.9 if bold else 0.8
    thick = 2    if bold else 1
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    tx = cx - tw // 2
    pad = 3
    bg = frame.copy()
    cv2.rectangle(bg, (tx-pad, top_y-th-pad), (tx+tw+pad, top_y+pad), (0, 0, 0), -1)
    cv2.addWeighted(bg, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (tx, top_y), _FONT, scale, color, thick, cv2.LINE_AA)


def _pick_color(obj, candidate):
    if obj.confirmed:  return _C["confirmed"]
    if obj.is_primary: return _C["primary"]
    if candidate and obj.id == candidate.id: return _C["candidate"]
    return _C.get(obj.cls, _C["default"])


def _state(obj, candidate):
    if obj.confirmed:  return "confirmed"
    if obj.is_primary: return "primary"
    if candidate and obj.id == candidate.id: return "candidate"
    return "passive"
