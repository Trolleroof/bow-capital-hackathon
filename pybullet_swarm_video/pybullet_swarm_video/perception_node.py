from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.visualizer import draw

from .bus_client import OrchestratorBusClient
from .config import OrchestratorConfig
from .messages import decode_jpeg_to_bgr, encode_jpeg_from_bgr


@dataclass
class OverlayTrack:
    id: int
    cls: str
    conf: float
    bbox: list[int]
    has_face: bool = False
    confirmed: bool = False
    is_primary: bool = False


class GroundTruthPerceptionNode:
    """Prototype perception worker.

    This is intentionally a simulation bridge, not a real detector:
    it receives raw FPV frames plus camera/target world metadata through the
    orchestrator, projects ground-truth troop positions into image space,
    overlays the existing perception HUD, and sends the annotated frame back.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._pending_frames: dict[str, dict[str, Any]] = {}
        self._pending_state: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name="ground-truth-perception")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run(self) -> None:
        bus = OrchestratorBusClient(self.config)
        try:
            await bus.connect(
                control_topics=[self.config.state_topic],
                image_topics=[self.config.raw_topic],
            )
            while True:
                control_task = asyncio.create_task(bus.next_control())
                image_task = asyncio.create_task(bus.next_image())
                done, pending = await asyncio.wait(
                    {control_task, image_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                message = next(iter(done)).result()
                topic = message.get("topic")
                if topic == self.config.state_topic:
                    self._pending_state[message["frame_id"]] = message
                    await self._maybe_process(bus, message["frame_id"])
                elif topic == self.config.raw_topic:
                    self._pending_frames[message["frame_id"]] = message
                    await self._maybe_process(bus, message["frame_id"])
        finally:
            await bus.close()

    async def _maybe_process(self, bus: OrchestratorBusClient, frame_id: str) -> None:
        state = self._pending_state.get(frame_id)
        frame_msg = self._pending_frames.get(frame_id)
        if state is None or frame_msg is None:
            return

        frame_bgr = decode_jpeg_to_bgr(frame_msg["data"])
        detections, candidate = self._project_targets(state)
        annotated = draw(frame_bgr.copy(), detections, candidate)

        image_payload = {
            "t": state["t"],
            "seq": state["seq"],
            "frame_id": frame_id,
            "drone_id": state["drone_id"],
            "source": state["source"],
            "width": state["width"],
            "height": state["height"],
            "encoding": "jpeg",
        }
        encoded = encode_jpeg_from_bgr(annotated)
        await bus.publish_encoded_frame(self.config.hud_topic, image_payload, encoded)
        if state["drone_id"] == self.config.dashboard_drone_id:
            await bus.publish_encoded_frame(
                self.config.dashboard_hud_topic,
                image_payload,
                encoded,
            )

        objects = [
            {
                "id": det.id,
                "cls": det.cls,
                "conf": round(det.conf, 3),
                "bbox": self._normalize_bbox(det.bbox, state["width"], state["height"]),
                "is_target": candidate is not None and det.id == candidate.id,
                "confirmed": False,
                "is_primary": det.is_primary,
                "is_candidate": candidate is not None and det.id == candidate.id,
            }
            for det in detections
        ]
        detection_payload = {
            "t": state["t"],
            "source": state["source"],
            "drone_id": state["drone_id"],
            "objects": objects,
        }
        await bus.publish_control(self.config.detections_topic, detection_payload)
        if state["drone_id"] == self.config.dashboard_drone_id:
            await bus.publish_control("detections", detection_payload)

        self._pending_state.pop(frame_id, None)
        self._pending_frames.pop(frame_id, None)
        self._prune()

    def _project_targets(
        self, state: dict[str, Any]
    ) -> tuple[list[OverlayTrack], OverlayTrack | None]:
        width = int(state["width"])
        height = int(state["height"])
        eye = np.asarray(state["eye"], dtype=np.float32)
        forward = self._unit(np.asarray(state["forward"], dtype=np.float32))
        up = self._unit(np.asarray(state["up"], dtype=np.float32))
        right = self._unit(np.cross(forward, up))
        vfov_rad = math.radians(float(state["fov_deg"]))
        aspect = width / max(height, 1)
        half_v = math.tan(vfov_rad / 2.0)
        half_h = half_v * aspect
        fx = width / max(2.0 * half_h, 1e-6)
        fy = height / max(2.0 * half_v, 1e-6)
        cx = width / 2.0
        cy = height / 2.0

        detections: list[OverlayTrack] = []
        for target in state.get("targets", []):
            center = np.array([target["x"], target["y"], target["z"]], dtype=np.float32)
            rel = center - eye
            z_cam = float(np.dot(rel, forward))
            if z_cam <= 0.2:
                continue
            x_cam = float(np.dot(rel, right))
            y_cam = float(np.dot(rel, up))

            px = cx + (x_cam * fx / z_cam)
            py = cy - (y_cam * fy / z_cam)
            box_h = max(8.0, fy * float(target.get("height_m", 1.75)) / z_cam)
            box_w = max(6.0, fx * float(target.get("width_m", 0.55)) / z_cam)
            left = int(round(px - box_w / 2.0))
            top = int(round(py - box_h / 2.0))
            bw = int(round(box_w))
            bh = int(round(box_h))

            if left + bw < 0 or top + bh < 0 or left >= width or top >= height:
                continue

            left = max(0, min(width - 1, left))
            top = max(0, min(height - 1, top))
            bw = max(1, min(width - left, bw))
            bh = max(1, min(height - top, bh))
            conf = max(0.45, min(0.98, 1.15 - z_cam / 55.0))
            detections.append(
                OverlayTrack(
                    id=int(target["id"]),
                    cls=str(target.get("cls", "troop")),
                    conf=conf,
                    bbox=[left, top, bw, bh],
                )
            )

        if not detections:
            return detections, None

        candidate = min(
            detections,
            key=lambda det: self._candidate_score(det.bbox, width, height),
        )
        candidate.is_primary = True
        return detections, candidate

    def _candidate_score(self, bbox: list[int], width: int, height: int) -> float:
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        dx = (cx - width / 2.0) / max(width, 1)
        dy = (cy - height / 2.0) / max(height, 1)
        area_bonus = (w * h) / max(width * height, 1)
        return dx * dx + dy * dy - area_bonus * 0.2

    def _normalize_bbox(self, bbox: list[int], width: int, height: int) -> list[float]:
        x, y, w, h = bbox
        return [
            round(x / width, 4),
            round(y / height, 4),
            round(w / width, 4),
            round(h / height, 4),
        ]

    def _prune(self) -> None:
        max_pending = 32
        if len(self._pending_state) > max_pending:
            stale = sorted(self._pending_state.keys())[:-max_pending]
            for key in stale:
                self._pending_state.pop(key, None)
                self._pending_frames.pop(key, None)
        if len(self._pending_frames) > max_pending:
            stale = sorted(self._pending_frames.keys())[:-max_pending]
            for key in stale:
                self._pending_frames.pop(key, None)
                self._pending_state.pop(key, None)

    def _unit(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return vec
        return vec / norm
