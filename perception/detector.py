"""YOLO detection + MediaPipe face overlay."""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Detection:
    cls: str           # battlefield label
    conf: float
    bbox: list[int]    # [x, y, w, h]
    has_face: bool = False


class Detector:
    def __init__(self) -> None:
        self.yolo = YOLO(config.YOLO_MODEL)
        self.yolo.to(config.DEVICE)

        self._face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,                         # short-range, lighter
            min_detection_confidence=config.FACE_CONF,
        )

    def run(self, frame: np.ndarray) -> list[Detection]:
        results = self.yolo(
            frame,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            coco_cls = results.names[int(box.cls)]
            bf_label = config.BATTLEFIELD_LABELS.get(coco_cls, "unknown")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det = Detection(
                cls=bf_label,
                conf=float(box.conf),
                bbox=[int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            )
            detections.append(det)

        self._annotate_faces(frame, detections)
        return detections

    def _annotate_faces(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """Run face detection only inside person/troop bounding boxes."""
        h, w = frame.shape[:2]
        for det in detections:
            if det.cls != "troop":
                continue
            x, y, bw, bh = det.bbox
            crop = frame[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            face_result = self._face.process(rgb)
            det.has_face = bool(
                face_result.detections and len(face_result.detections) > 0
            )
