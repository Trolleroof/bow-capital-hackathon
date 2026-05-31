#!/usr/bin/env python3
"""ROS2 detector node for Jetson-hosted OAK-D perception.

The node subscribes to an OAK image topic, runs either YOLOX or Ultralytics
YOLO/TensorRT inference with CUDA when available, and publishes two topics:

* /perception/detections: std_msgs/String containing normalized JSON boxes.
* /perception/annotated_image: sensor_msgs/Image with overlayed boxes.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


DEFAULT_LABELS: dict[str, str] = {
    "person": "troop",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "ugv",
    "bicycle": "ugv",
    "airplane": "aerial",
    "helicopter": "aerial",
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]


class UltralyticsBackend:
    def __init__(self, model_path: str, device: str, conf: float, iou: float, fp16: bool) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.fp16 = fp16 and device not in ("", "cpu")
        if device not in ("", "cpu"):
            self.model.to(device)
            if self.fp16 and hasattr(self.model, "model"):
                self.model.model.half()

    def infer(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device or None,
            half=self.fp16,
            verbose=False,
            show=False,
        )[0]
        out: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls)
            raw = result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else str(class_id)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            out.append(Detection(raw, float(box.conf), (x1, y1, x2, y2)))
        return out


class YoloXBackend:
    def __init__(self, model_path: str, exp_file: str, device: str, conf: float, iou: float, fp16: bool) -> None:
        torch = importlib.import_module("torch")
        exp_mod = importlib.import_module("yolox.exp")
        data_mod = importlib.import_module("yolox.data.data_augment")
        utils_mod = importlib.import_module("yolox.utils")

        self.torch = torch
        self.preproc = data_mod.preproc
        self.postprocess = utils_mod.postprocess
        self.device = "cuda" if device not in ("", "cpu") and torch.cuda.is_available() else "cpu"
        self.conf = conf
        self.iou = iou
        self.fp16 = fp16 and self.device == "cuda"

        exp = exp_mod.get_exp(exp_file if exp_file else None, None)
        model = exp.get_model()
        checkpoint = torch.load(model_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        if self.device == "cuda":
            model.cuda()
            if self.fp16:
                model.half()
        self.model = model
        self.test_size = exp.test_size
        self.num_classes = exp.num_classes
        self.names = getattr(exp, "class_names", COCO_NAMES)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        img, ratio = self.preproc(frame, self.test_size)
        tensor = self.torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "cuda":
            tensor = tensor.cuda()
            if self.fp16:
                tensor = tensor.half()
        with self.torch.no_grad():
            outputs = self.model(tensor)
            outputs = self.postprocess(outputs, self.num_classes, self.conf, self.iou, class_agnostic=True)
        if outputs[0] is None:
            return []
        detections = outputs[0].detach().float().cpu().numpy()
        out: list[Detection] = []
        for row in detections:
            x1, y1, x2, y2 = row[:4] / ratio
            obj_conf, cls_conf, cls_id = float(row[4]), float(row[5]), int(row[6])
            label = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
            out.append(Detection(label, obj_conf * cls_conf, (float(x1), float(y1), float(x2), float(y2))))
        return out


class YoloXNode(Node):
    def __init__(self) -> None:
        super().__init__("combatos_yolox_node")
        self.declare_parameter("image_topic", "/oak/rgb/image_rect")
        self.declare_parameter("detections_topic", "/perception/detections")
        self.declare_parameter("annotated_topic", "/perception/annotated_image")
        self.declare_parameter("model_type", os.getenv("COMBATOS_YOLO_MODEL_TYPE", "ultralytics"))
        self.declare_parameter("model_path", os.getenv("COMBATOS_YOLO_MODEL", "perception/yolo11n.pt"))
        self.declare_parameter("yolox_exp_file", os.getenv("COMBATOS_YOLOX_EXP", ""))
        self.declare_parameter("device", os.getenv("COMBATOS_YOLO_DEVICE", "0"))
        self.declare_parameter("confidence", float(os.getenv("COMBATOS_YOLO_CONF", "0.40")))
        self.declare_parameter("iou", float(os.getenv("COMBATOS_YOLO_IOU", "0.45")))
        self.declare_parameter("fp16", os.getenv("COMBATOS_YOLO_FP16", "1") == "1")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("max_fps", float(os.getenv("COMBATOS_YOLO_MAX_FPS", "15.0")))

        self.bridge = CvBridge()
        self.last_infer = 0.0
        self.seq = 0
        self.frame_times: list[float] = []

        model_type = str(self.get_parameter("model_type").value).lower()
        model_path = self._resolve_model_path(str(self.get_parameter("model_path").value))
        exp_file = str(self.get_parameter("yolox_exp_file").value)
        device = str(self.get_parameter("device").value)
        conf = float(self.get_parameter("confidence").value)
        iou = float(self.get_parameter("iou").value)
        fp16 = bool(self.get_parameter("fp16").value)

        if model_type == "yolox":
            if not exp_file:
                raise RuntimeError("model_type=yolox requires yolox_exp_file")
            self.backend = YoloXBackend(model_path, exp_file, device, conf, iou, fp16)
        else:
            self.backend = UltralyticsBackend(model_path, device, conf, iou, fp16)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.det_pub = self.create_publisher(String, str(self.get_parameter("detections_topic").value), 10)
        self.annotated_pub = self.create_publisher(Image, str(self.get_parameter("annotated_topic").value), image_qos)
        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self._on_image, image_qos)
        self.get_logger().info(
            f"subscribed {self.get_parameter('image_topic').value}; publishing "
            f"{self.get_parameter('detections_topic').value} and {self.get_parameter('annotated_topic').value}"
        )

    def _resolve_model_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if path.exists():
            return str(path)
        repo_relative = Path.cwd() / path
        if repo_relative.exists():
            return str(repo_relative)
        return value

    def _on_image(self, msg: Image) -> None:
        max_fps = float(self.get_parameter("max_fps").value)
        now = time.monotonic()
        if max_fps > 0 and now - self.last_infer < 1.0 / max_fps:
            return
        self.last_infer = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warning(f"cv_bridge decode failed: {exc}")
            return

        start = time.perf_counter()
        detections = self.backend.infer(frame)
        latency_ms = (time.perf_counter() - start) * 1000.0
        h, w = frame.shape[:2]
        self.seq += 1

        objects = [self._serialize_detection(i + 1, det, w, h) for i, det in enumerate(detections)]
        payload: dict[str, Any] = {
            "schema": "combatos.perception.v1",
            "seq": self.seq,
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": msg.header.frame_id,
            "source": str(self.get_parameter("image_topic").value),
            "width": int(w),
            "height": int(h),
            "latency_ms": round(latency_ms, 2),
            "objects": objects,
        }
        out = String()
        out.data = json.dumps(payload, separators=(",", ":"))
        self.det_pub.publish(out)

        if bool(self.get_parameter("publish_annotated").value):
            annotated = frame.copy()
            self._draw(annotated, objects, latency_ms)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

    def _serialize_detection(self, track_id: int, det: Detection, width: int, height: int) -> dict[str, Any]:
        x1, y1, x2, y2 = det.xyxy
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(x1, min(float(width), x2))
        y2 = max(y1, min(float(height), y2))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        label = DEFAULT_LABELS.get(det.label, det.label)
        return {
            "id": track_id,
            "cls": label,
            "raw_cls": det.label,
            "conf": round(float(det.conf), 4),
            "bbox": [x1 / width, y1 / height, bw / width, bh / height],
            "bbox_px": [round(x1), round(y1), round(bw), round(bh)],
            "is_primary": False,
            "is_candidate": track_id == 1,
            "confirmed": False,
        }

    def _draw(self, frame: np.ndarray, objects: list[dict[str, Any]], latency_ms: float) -> None:
        h, w = frame.shape[:2]
        for obj in objects:
            x, y, bw, bh = [int(v) for v in obj["bbox_px"]]
            color = (0, 220, 255) if obj["cls"] == "troop" else (0, 160, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)
            label = f"{obj['id']:02d} {str(obj['cls']).upper()} {obj['conf']:.2f}"
            cv2.putText(frame, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"YOLOX {len(objects)} tracks {latency_ms:.1f} ms",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120),
            1,
            cv2.LINE_AA,
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloXNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
