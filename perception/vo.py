"""
perception/vo.py -- Monocular visual odometry + CombatOS SLAM publishing.

This is the SLAM half of the local pipeline, kept framework-agnostic so it can
be driven two ways:

  * main.py  -- calls SlamStreamer.process_and_publish() on the SAME frame it
                feeds to YOLO, so the targeting feed and the SLAM feed are the
                same frame at the same timestamp (perfectly in sync).
  * slam_sim.py -- runs the same VO standalone against an arbitrary clip.

Pipeline per frame:
    ORB features -> ratio-matched to previous frame -> findEssentialMat +
    recoverPose (relative R, t) -> integrate global trajectory -> triangulate
    inlier matches into a sparse point cloud.

Stability: monocular VO on real footage drops lock constantly. To keep the
panels from strobing we (a) hold the last good pose through a grace window of
failed frames instead of immediately reporting LOST, (b) EMA-smooth the
published pose, and (c) add only a few map points per frame so the cloud does
not shimmer. See the SLAM_SIM_* knobs in config.py.

Coordinates are converted from the OpenCV optical frame (x-right, y-down,
z-forward) to the ROS REP-103 frame (x-forward, y-left, z-up) that the
dashboard's VslamScene expects, so the trajectory sits upright on the grid.
"""
from __future__ import annotations

import base64
import math
import time
from collections import deque

import cv2
import numpy as np

import config


def cam_to_ros(vec: np.ndarray) -> tuple[float, float, float]:
    """OpenCV optical (x-right, y-down, z-fwd) -> ROS REP-103 (x-fwd, y-left, z-up)."""
    cx, cy, cz = float(vec[0]), float(vec[1]), float(vec[2])
    return cz, -cx, -cy


def encode_jpeg(frame: np.ndarray) -> str | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.SLAM_SIM_JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf).decode("ascii")


class MonoVO:
    """Frame-to-frame essential-matrix visual odometry with sparse mapping."""

    def __init__(self, width: int, height: int) -> None:
        focal = config.SLAM_SIM_FOCAL_RATIO * width
        self.K = np.array(
            [[focal, 0.0, width / 2.0],
             [0.0, focal, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._orb = cv2.ORB_create(nfeatures=config.SLAM_SIM_ORB_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Accumulated camera pose in the optical world frame.
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)

        self._prev_kp: list | None = None
        self._prev_des: np.ndarray | None = None

        self.cloud: deque[tuple[float, float, float]] = deque(maxlen=config.SLAM_SIM_MAX_POINTS)
        self.path: deque[dict[str, float]] = deque(maxlen=config.SLAM_SIM_MAX_PATH)
        self.tracking = "INITIALIZING"
        self.last_matches: list[tuple[tuple[int, int], tuple[int, int]]] = []

        self._spos: np.ndarray | None = None     # EMA-smoothed ROS position
        self._lost_streak = 0
        self._grace = max(0, config.SLAM_SIM_LOST_GRACE)
        self._alpha = min(1.0, max(0.01, config.SLAM_SIM_POSE_SMOOTH))

    # -- main entry ----------------------------------------------------------

    def process(self, gray: np.ndarray) -> None:
        kp, des = self._orb.detectAndCompute(gray, None)
        self.last_matches = []
        has_features = des is not None and len(kp) >= 8

        if has_features and self._prev_des is not None and self._step(kp, des):
            self._lost_streak = 0
            self.tracking = "TRACKING"
        elif self._prev_des is None and self._spos is None:
            self.tracking = "INITIALIZING"   # true startup, nothing to match yet
        else:
            self._on_failure()               # match failed / blank frame -> hold

        # Only advance the reference frame when this frame actually has features.
        # A single blank or motion-blurred frame must NOT wipe the keyframe, or
        # the next frame would have nothing to match and falsely re-INITIALIZE.
        if has_features:
            self._remember(kp, des)

    def _step(self, kp, des) -> bool:
        """Estimate relative motion against the previous frame. False = no lock."""
        good = []
        for pair in self._matcher.knnMatch(self._prev_des, des, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 20:
            return False

        pts_prev = np.float32([self._prev_kp[m.queryIdx].pt for m in good])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts_cur, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or E.shape != (3, 3):
            return False

        inliers, R, t, pose_mask = cv2.recoverPose(E, pts_cur, pts_prev, self.K, mask=mask)
        if inliers < 15:
            return False

        R_world_prev = self.R.copy()
        t_world_prev = self.t.copy()

        scale = config.SLAM_SIM_TRANSLATION_SCALE
        self.t = self.t + scale * (self.R @ t)
        self.R = R @ self.R

        self._update_smoothed_pose()
        self._triangulate(pts_prev, pts_cur, R, t, pose_mask, R_world_prev, t_world_prev)
        self._collect_match_lines(pts_prev, pts_cur, pose_mask)
        return True

    def _on_failure(self) -> None:
        """Hold the last good pose; only report LOST after the grace window."""
        self._lost_streak += 1
        if self._spos is None:
            self.tracking = "INITIALIZING"
        elif self._lost_streak > self._grace:
            self.tracking = "LOST"
        else:
            self.tracking = "TRACKING"  # brief dropout -- keep the panels calm

    # -- helpers -------------------------------------------------------------

    def _update_smoothed_pose(self) -> None:
        raw = np.array(cam_to_ros(self.t.ravel()), dtype=np.float64)
        if self._spos is None:
            self._spos = raw
        else:
            self._spos = self._alpha * raw + (1.0 - self._alpha) * self._spos
        p = self._spos
        self.path.append({"t": time.time(), "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})

    def _triangulate(self, pts_prev, pts_cur, R, t, pose_mask, R_world_prev, t_world_prev) -> None:
        inliers = pose_mask.ravel().astype(bool) if pose_mask is not None else np.ones(len(pts_prev), bool)
        if inliers.sum() < 6:
            return
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, t])
        hom = cv2.triangulatePoints(P0, P1, pts_prev[inliers].T, pts_cur[inliers].T)
        w = hom[3]
        w[np.abs(w) < 1e-6] = 1e-6
        pts3d = (hom[:3] / w).T  # previous-camera frame

        added = 0
        cap = max(1, config.SLAM_SIM_POINTS_PER_FRAME)
        for p in pts3d:
            if not np.all(np.isfinite(p)) or p[2] <= 0 or p[2] > 200:
                continue
            world = (R_world_prev @ p.reshape(3, 1) + t_world_prev).ravel()
            self.cloud.append(cam_to_ros(world))
            added += 1
            if added >= cap:
                break

    def _collect_match_lines(self, pts_prev, pts_cur, pose_mask) -> None:
        inliers = pose_mask.ravel().astype(bool) if pose_mask is not None else np.ones(len(pts_prev), bool)
        for a, b, keep in zip(pts_prev, pts_cur, inliers):
            if keep:
                self.last_matches.append(((int(a[0]), int(a[1])), (int(b[0]), int(b[1]))))

    def _remember(self, kp, des) -> None:
        self._prev_kp = kp
        self._prev_des = des

    # -- accessors -----------------------------------------------------------

    def published_pose(self) -> tuple[float, float, float]:
        if self._spos is None:
            return 0.0, 0.0, 0.0
        return float(self._spos[0]), float(self._spos[1]), float(self._spos[2])

    def heading_quat(self) -> tuple[float, float, float, float]:
        """Yaw quaternion from the recent smoothed travel direction."""
        if len(self.path) < 2:
            return 0.0, 0.0, 0.0, 1.0
        a, b = self.path[-2], self.path[-1]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if math.hypot(dx, dy) < 1e-4:
            return 0.0, 0.0, 0.0, 1.0
        yaw = math.atan2(dy, dx)
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def annotate_frame(frame: np.ndarray, vo: MonoVO) -> np.ndarray:
    out = frame.copy()
    for (a, b) in vo.last_matches:
        cv2.circle(out, b, 2, (0, 220, 255), -1)
        cv2.line(out, a, b, (0, 140, 90), 1, cv2.LINE_AA)
    color = (0, 220, 120) if vo.tracking == "TRACKING" else (60, 60, 230)
    cv2.putText(out, f"VSLAM {vo.tracking}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(out, f"map {len(vo.cloud)} pts  path {len(vo.path)}", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


class SlamStreamer:
    """Runs MonoVO on each frame and publishes the CombatOS SLAM bus topics.

    Decoupled from any specific bus client: callers pass two send callables,
    ``send_control(topic, payload)`` and ``send_image(topic, payload)``.
    """

    def __init__(self, width: int, height: int, *, publish_camera: bool | None = None) -> None:
        self.width = width
        self.height = height
        self.vo = MonoVO(width, height)
        self.publish_camera = (
            config.SLAM_SIM_PUBLISH_CAMERA if publish_camera is None else publish_camera
        )
        self._video_period = 1.0 / max(0.1, config.SLAM_SIM_VIDEO_FPS)
        self._cam_seq = 0
        self._slam_seq = 0
        self._last_video_pub = 0.0
        self._prev_pose: tuple[float, ...] | None = None
        self._prev_t: float | None = None

    def process_and_publish(self, frame_bgr: np.ndarray, now: float, send_control, send_image) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.vo.process(gray)

        dt = max(1e-3, now - self._prev_t) if self._prev_t is not None else 1e-3
        self._prev_t = now

        x, y, z = self.vo.published_pose()
        qx, qy, qz, qw = self.vo.heading_quat()

        if self._prev_pose is None:
            vx = vy = vz = wz = 0.0
        else:
            px, py, pz, _, _, pqz, pqw = self._prev_pose
            vx, vy, vz = (x - px) / dt, (y - py) / dt, (z - pz) / dt
            prev_yaw = math.atan2(2 * pqw * pqz, 1 - 2 * pqz * pqz)
            cur_yaw = math.atan2(2 * qw * qz, 1 - 2 * qz * qz)
            wz = (cur_yaw - prev_yaw) / dt
        self._prev_pose = (x, y, z, qx, qy, qz, qw)

        track = self.vo.tracking
        send_control("pose", {
            "t": now, "x": x, "y": y, "z": z,
            "qx": qx, "qy": qy, "qz": qz, "qw": qw, "gps": False, "tracking": track,
        })
        send_control("slam_odometry", {
            "t": now, "frame_id": "map", "child_frame_id": "base_link",
            "x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "vx": vx, "vy": vy, "vz": vz, "wx": 0.0, "wy": 0.0, "wz": wz, "tracking": track,
        })
        send_control("slam_path", {"t": now, "frame_id": "map", "poses": list(self.vo.path)})
        send_control("slam_point_cloud", {
            "t": now, "frame_id": "map",
            "points": [{"x": p[0], "y": p[1], "z": p[2]} for p in self.vo.cloud],
            "total_points": len(self.vo.cloud),
        })
        send_control("slam_status", {
            "t": now, "tracking": track, "connected": True,
            "camera_frames": self._cam_seq, "annotated_frames": self._slam_seq, "dropped_frames": 0,
        })
        send_control("slam_diagnostics", {
            "t": now, "tracking": track, "dropped_frames": 0,
            "camera_frames": self._cam_seq, "annotated_frames": self._slam_seq, "queue_depth": 0,
        })

        if now - self._last_video_pub >= self._video_period:
            self._last_video_pub = now
            data = encode_jpeg(annotate_frame(frame_bgr, self.vo))
            if data is not None:
                self._slam_seq += 1
                send_image("slam_frame", {
                    "t": now, "frame_id": "map", "source": "vo", "encoding": "jpeg",
                    "width": self.width, "height": self.height, "seq": self._slam_seq, "data": data,
                })
            if self.publish_camera:
                raw = encode_jpeg(frame_bgr)
                if raw is not None:
                    self._cam_seq += 1
                    send_image("camera_frame", {
                        "t": now, "frame_id": "camera", "source": "vo", "encoding": "jpeg",
                        "width": self.width, "height": self.height, "seq": self._cam_seq, "data": raw,
                    })
