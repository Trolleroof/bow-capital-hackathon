from __future__ import annotations

import shutil
import zipfile
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Sequence

import numpy as np

from .config import SimulationConfig
from .policies import ScriptedSurveillancePolicy

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - import guard only
    raise RuntimeError(
        "pybullet is required for the PyBullet swarm video prototype. "
        "Install dependencies with `uv sync --project pybullet_swarm_video`."
    ) from exc


@dataclass
class SimulationSnapshot:
    sim_time: float
    drone_positions: np.ndarray
    troop_positions: np.ndarray


@dataclass
class CameraPose:
    eye: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    width: int
    height: int
    fov_deg: float


@dataclass(frozen=True)
class MeshAsset:
    path: Path
    scale_xyz: tuple[float, float, float]
    roll_offset: float = 0.0
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    vertical_offset: float = 0.0
    texture_path: Path | None = None
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


class SimulationDisconnectedError(RuntimeError):
    """Raised when the PyBullet client has disconnected during a running sim."""


TROOP_COLLIDER_RADIUS_M = 0.11
TROOP_COLLIDER_HEIGHT_M = 1.55
TROOP_MASS_KG = 80.0
TROOP_DETECTION_WIDTH_M = 0.36
TROOP_DETECTION_HEIGHT_M = 1.75


class DroneSurveillanceSimulation:
    def __init__(self, config: SimulationConfig, gui: bool = False) -> None:
        self.config = config
        self.gui = gui
        self.client_id: int | None = None
        self.rng = np.random.default_rng(7)

        self.drone_ids: list[int] = []
        self.troop_ids: list[int] = []
        self._drone_marker_ids: list[int] = []
        self._troop_marker_ids: list[int] = []
        self._drone_debug_line_ids: list[int] = []
        self._troop_debug_line_ids: list[int] = []
        self._ruin_ids: list[int] = []
        self.drone_positions = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_velocities = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_yaws = np.zeros(config.num_drones, dtype=np.float32)
        self.troop_positions = np.zeros((config.num_troops, 3), dtype=np.float32)
        self.troop_yaws = np.zeros(config.num_troops, dtype=np.float32)
        self._troop_headings = np.zeros(config.num_troops, dtype=np.float32)
        self._troop_offsets = np.zeros((config.num_troops, 2), dtype=np.float32)
        self._troop_anchor_ids = np.zeros(config.num_troops, dtype=np.int32)
        self._troop_anchor_bases = np.zeros((4, 2), dtype=np.float32)
        self._troop_anchor_dirs = np.zeros(4, dtype=np.float32)
        self._troop_personal_phase = np.zeros(config.num_troops, dtype=np.float32)
        self._troop_mesh_mask = np.zeros(config.num_troops, dtype=bool)
        self.sim_time = 0.0
        self.camera_mode: Literal["observer", "chase", "fpv"] = "observer"
        self.selected_drone_id = 0
        self.manual_drone_id: int | None = None
        self._plane_id: int | None = None
        self._fpv_hidden_drone_id: int | None = None
        self._resource_root = config.resources_dir
        self._resource_cache = self._resource_root / ".cache"
        self._asset_messages: set[str] = set()
        self._texture_ids: dict[Path, int] = {}
        self._obj_bounds_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        self._split_obj_cache: dict[Path, list[tuple[str, Path]]] = {}
        self._vignette_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self._frame_history: dict[int, np.ndarray] = {}
        self._keyboard_events: dict[int, int] = {}
        self._drone_rgba = [1.0, 1.0, 1.0, 1.0]
        self._drone_mesh_asset_ref: MeshAsset | None = None
        self._troop_mesh_asset_ref: MeshAsset | None = None
        self._troop_visual_z_offset = 0.0
        self._troop_visual_yaw_offset = 0.0
        self._salt_dome_active = False
        self._ground_offset_z = 0.0

        self.policy = ScriptedSurveillancePolicy(
            num_drones=config.num_drones,
            ring_radius_m=config.drone_ring_radius_m,
            cruise_altitude_m=config.drone_altitude_m,
            max_speed_mps=config.drone_speed_mps,
            separation_gain=config.drone_separation_gain,
        )

    def __enter__(self) -> DroneSurveillanceSimulation:
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> None:
        self.close()
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.config.time_step, physicsClientId=self.client_id)
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        if self.gui:
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI,
                0,
                physicsClientId=self.client_id,
            )

        self._spawn_ground_markers()
        self._spawn_troops()
        self._spawn_drones()
        self.sim_time = 0.0
        if self.gui:
            self._print_controls()

    def close(self) -> None:
        if self.client_id is None:
            return
        try:
            if self.is_connected():
                p.disconnect(physicsClientId=self.client_id)
        except Exception:
            pass
        finally:
            self.client_id = None

    def is_connected(self) -> bool:
        if self.client_id is None:
            return False
        try:
            if hasattr(p, "getConnectionInfo"):
                info = p.getConnectionInfo(physicsClientId=self.client_id)
                if isinstance(info, dict) and "isConnected" in info:
                    return bool(info["isConnected"])
        except Exception:
            return False

        try:
            if hasattr(p, "isConnected"):
                try:
                    return bool(p.isConnected(self.client_id))
                except TypeError:
                    return bool(p.isConnected())
        except Exception:
            return False

        try:
            p.getPhysicsEngineParameters(physicsClientId=self.client_id)
            return True
        except Exception:
            return False

    def _require_connection(self) -> int:
        if not self.is_connected():
            raise SimulationDisconnectedError("PyBullet client disconnected")
        assert self.client_id is not None
        return self.client_id

    def _print_controls(self) -> None:
        print(
            "[sim] controls: C cycle camera | B observer | H chase | F fpv | 1-9 select drone | "
            "M toggle manual for selected drone | R return selected drone to scripted mode | "
            "arrow keys or I/K/J/L move | U/O altitude | Z/X yaw"
        )

    def _log_asset(self, message: str) -> None:
        if message in self._asset_messages:
            return
        self._asset_messages.add(message)
        print(f"[sim] {message}")

    def _event_flags(self, *keys: int) -> int:
        flags = 0
        for key in keys:
            flags |= int(self._keyboard_events.get(key, 0))
        return flags

    def _key_triggered(self, *keys: int) -> bool:
        return bool(self._event_flags(*keys) & p.KEY_WAS_TRIGGERED)

    def _key_down(self, *keys: int) -> bool:
        return bool(self._event_flags(*keys) & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED))

    def _resource_path(self, *names: str) -> Path:
        return self._resource_root.joinpath(*names)

    def _extract_archive_tree(self, archive_path: Path, target_dir_name: str) -> Path | None:
        if not archive_path.exists():
            return None
        target_dir = self._resource_cache / target_dir_name
        stamp_path = target_dir / ".stamp"
        archive_stamp = str(archive_path.stat().st_mtime_ns)
        try:
            if stamp_path.exists() and stamp_path.read_text().strip() == archive_stamp:
                return target_dir
        except OSError:
            pass

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target_dir)
            stamp_path.write_text(archive_stamp)
        except (OSError, zipfile.BadZipFile):
            self._log_asset(f"failed to extract asset archive {archive_path.name}; using fallback visuals")
            return None
        return target_dir

    def _extract_zip_member(self, archive_path: Path, suffixes: Sequence[str]) -> Path | None:
        if not archive_path.exists():
            return None
        self._resource_cache.mkdir(parents=True, exist_ok=True)
        wanted = tuple(suffix.lower() for suffix in suffixes)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    lowered = member.lower()
                    if not any(lowered.endswith(suffix) for suffix in wanted):
                        continue
                    target = self._resource_cache / Path(member).name
                    if not target.exists():
                        with archive.open(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    return target
        except (OSError, zipfile.BadZipFile):
            self._log_asset(f"failed to inspect asset archive {archive_path.name}; using fallback visuals")
        return None

    def _find_file(self, root: Path, name: str | None = None, suffix: str | None = None) -> Path | None:
        if not root.exists():
            return None
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            if name is not None and candidate.name != name:
                continue
            if suffix is not None and candidate.suffix.lower() != suffix.lower():
                continue
            return candidate
        return None

    def _obj_bounds(self, obj_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        cached = self._obj_bounds_cache.get(obj_path)
        if cached is not None:
            return cached
        mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        try:
            with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.startswith("v "):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    vertex = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float64,
                    )
                    mins = np.minimum(mins, vertex)
                    maxs = np.maximum(maxs, vertex)
        except OSError:
            self._log_asset(f"failed to read mesh bounds from {obj_path.name}; using fallback visuals")
            return None

        if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
            self._log_asset(f"mesh {obj_path.name} did not contain vertex data; using fallback visuals")
            return None

        result = (mins, maxs)
        self._obj_bounds_cache[obj_path] = result
        return result

    def _parse_mtl_diffuse_colors(self, mtl_path: Path | None) -> dict[str, tuple[float, float, float, float]]:
        if mtl_path is None or not mtl_path.exists():
            return {}
        colors: dict[str, tuple[float, float, float, float]] = {}
        current_name: str | None = None
        try:
            with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.startswith("newmtl "):
                        current_name = stripped.split(None, 1)[1]
                        continue
                    if stripped.startswith("Kd ") and current_name is not None:
                        parts = stripped.split()
                        if len(parts) >= 4:
                            colors[current_name] = (
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                1.0,
                            )
        except OSError:
            self._log_asset(f"failed to read material data from {mtl_path.name}; using fallback colors")
        return colors

    @staticmethod
    def _strip_face_uvs(face_line: str) -> str:
        parts = face_line.rstrip("\n").split()
        out = ["f"]
        for vert in parts[1:]:
            segs = vert.split("/")
            if len(segs) == 3 and segs[1]:
                out.append(f"{segs[0]}//{segs[2]}")
            elif len(segs) == 2 and segs[1]:
                out.append(segs[0])
            else:
                out.append(vert)
        return " ".join(out) + "\n"

    def _strip_uv_obj(self, obj_path: Path) -> Path:
        """Return a UV-stripped copy of obj_path (f v//n) so PyBullet GUI renders it."""
        stripped_path = obj_path.with_suffix("").with_name(obj_path.stem + "_nouv.obj")
        stamp_path = stripped_path.with_suffix(".stamp")
        obj_stamp = str(obj_path.stat().st_mtime_ns)
        try:
            if stripped_path.exists() and stamp_path.read_text().strip() == obj_stamp:
                return stripped_path
        except OSError:
            pass
        try:
            needs_strip = False
            with obj_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("f "):
                        parts = line.split()
                        if len(parts) > 1 and parts[1].count("/") == 2:
                            idx = parts[1].split("/")
                            if idx[1]:
                                needs_strip = True
                        break
            if not needs_strip:
                return obj_path
            with obj_path.open("r", encoding="utf-8", errors="ignore") as fin, \
                 stripped_path.open("w", encoding="utf-8") as fout:
                for line in fin:
                    if line.startswith("vt "):
                        continue
                    if line.startswith("f "):
                        fout.write(self._strip_face_uvs(line))
                    else:
                        fout.write(line)
            stamp_path.write_text(obj_stamp)
        except OSError:
            self._log_asset(f"failed to strip UV from {obj_path.name}; using original (may be invisible in GUI)")
            return obj_path
        return stripped_path

    def _split_obj_by_material(self, obj_path: Path) -> list[tuple[str, Path]]:
        cached = self._split_obj_cache.get(obj_path)
        if cached is not None:
            return cached

        target_dir = obj_path.parent / f"{obj_path.stem}_parts"
        stamp_path = target_dir / ".stamp"
        obj_stamp = "v2:" + str(obj_path.stat().st_mtime_ns)
        manifest_path = target_dir / "manifest.txt"
        if stamp_path.exists() and manifest_path.exists():
            try:
                if stamp_path.read_text().strip() == obj_stamp:
                    pairs: list[tuple[str, Path]] = []
                    for line in manifest_path.read_text().splitlines():
                        if not line:
                            continue
                        material, rel_path = line.split("|", 1)
                        pairs.append((material, target_dir / rel_path))
                    self._split_obj_cache[obj_path] = pairs
                    return pairs
            except OSError:
                pass

        target_dir.mkdir(parents=True, exist_ok=True)
        vertices: list[str] = []
        texcoords: list[str] = []
        normals: list[str] = []
        faces_by_material: dict[str, list[str]] = {}
        material_order: list[str] = []
        current_material = "default"

        try:
            with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("v "):
                        vertices.append(line)
                    elif line.startswith("vt "):
                        texcoords.append(line)
                    elif line.startswith("vn "):
                        normals.append(line)
                    elif line.startswith("usemtl "):
                        current_material = line.split(None, 1)[1].strip()
                        if current_material not in faces_by_material:
                            faces_by_material[current_material] = []
                            material_order.append(current_material)
                    elif line.startswith("f "):
                        faces_by_material.setdefault(current_material, []).append(line)
                        if current_material not in material_order:
                            material_order.append(current_material)
        except OSError:
            self._log_asset(f"failed to split {obj_path.name} by material; using fallback visuals")
            return []

        pairs: list[tuple[str, Path]] = []
        manifest_lines: list[str] = []
        for index, material in enumerate(material_order):
            faces = faces_by_material.get(material, [])
            if not faces:
                continue
            out_path = target_dir / f"{obj_path.stem}_{index:02d}.obj"
            with out_path.open("w", encoding="utf-8") as handle:
                handle.writelines(vertices)
                handle.writelines(normals)
                handle.write("s off\n")
                for face_line in faces:
                    handle.write(self._strip_face_uvs(face_line))
            pairs.append((material, out_path))
            manifest_lines.append(f"{material}|{out_path.name}")

        try:
            stamp_path.write_text(obj_stamp)
            manifest_path.write_text("\n".join(manifest_lines))
        except OSError:
            pass

        self._split_obj_cache[obj_path] = pairs
        return pairs

    def _mesh_asset_from_archive(
        self,
        archive_name: str,
        obj_name: str,
        target_height_m: float | None = None,
        target_max_dim_m: float | None = None,
        yaw_offset: float = 0.0,
        texture_name: str | None = None,
        rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        up_axis: Literal["x", "y", "z"] = "y",
        roll_offset: float = math.pi / 2.0,
        pitch_offset: float = 0.0,
        scale_xyz_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> MeshAsset | None:
        archive = self._resource_path(archive_name)
        extracted = self._extract_archive_tree(archive, archive.stem)
        if extracted is None:
            return None
        obj_path = self._find_file(extracted, name=obj_name)
        if obj_path is None:
            self._log_asset(f"{archive_name} did not contain {obj_name}; using fallback visuals")
            return None
        obj_path = self._strip_uv_obj(obj_path)

        bounds = self._obj_bounds(obj_path)
        if bounds is None:
            return None
        mins, maxs = bounds
        size = maxs - mins
        up_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
        up_size = float(size[up_idx])
        max_dim = float(np.max(size))
        if target_height_m is not None and up_size > 1e-9:
            scale = target_height_m / up_size
        elif target_max_dim_m is not None and max_dim > 1e-9:
            scale = target_max_dim_m / max_dim
        else:
            scale = 1.0

        texture_path = self._find_file(extracted, name=texture_name) if texture_name else None
        scale_xyz = tuple(float(scale * axis_scale) for axis_scale in scale_xyz_multiplier)
        return MeshAsset(
            path=obj_path,
            scale_xyz=scale_xyz,
            roll_offset=roll_offset,
            pitch_offset=pitch_offset,
            yaw_offset=yaw_offset,
            vertical_offset=float(-mins[up_idx] * scale_xyz[up_idx]),
            texture_path=texture_path,
            rgba=rgba,
        )

    def _load_texture_if_present(self, texture_path: Path | None) -> int | None:
        if texture_path is None or not texture_path.exists():
            return None
        cached = self._texture_ids.get(texture_path)
        if cached is not None:
            return cached
        try:
            texture_id = int(
                p.loadTexture(str(texture_path), physicsClientId=self._require_connection())
            )
        except Exception:
            self._log_asset(f"failed to load texture {texture_path.name}; leaving default material")
            return None
        self._texture_ids[texture_path] = texture_id
        return texture_id

    def _vignette_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        cached = self._vignette_mask_cache.get(key)
        if cached is not None:
            return cached
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        radius = np.sqrt(xx * xx + yy * yy)
        mask = np.clip(1.06 - 0.38 * radius**1.7, 0.62, 1.0).astype(np.float32)
        self._vignette_mask_cache[key] = mask
        return mask

    def _postprocess_frame(self, drone_idx: int, frame: np.ndarray) -> np.ndarray:
        image = frame.astype(np.float32) / 255.0
        mask = self._vignette_mask(frame.shape[1], frame.shape[0])
        image *= mask[:, :, None]

        # Mild surveillance-camera grade: less saturation, slightly harder contrast.
        luma = image @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        image = image * 0.88 + luma[:, :, None] * 0.12
        image = np.clip((image - 0.5) * 1.08 + 0.5, 0.0, 1.0)

        speed = float(np.linalg.norm(self.drone_velocities[drone_idx]))
        noise_sigma = 0.006 + 0.006 * min(speed, 8.0) / 8.0
        if noise_sigma > 0.0:
            image = np.clip(
                image + self.rng.normal(0.0, noise_sigma, size=image.shape).astype(np.float32),
                0.0,
                1.0,
            )

        previous = self._frame_history.get(drone_idx)
        if previous is not None:
            blur_mix = 0.05 + 0.06 * min(speed, 8.0) / 8.0
            image = np.clip(image * (1.0 - blur_mix) + previous * blur_mix, 0.0, 1.0)
        self._frame_history[drone_idx] = image.copy()
        return (image * 255.0).astype(np.uint8)

    def _create_mesh_visual(self, asset: MeshAsset | None) -> int | None:
        if asset is None or not asset.path.exists():
            return None
        try:
            visual = int(
                p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=str(asset.path),
                    meshScale=list(asset.scale_xyz),
                    rgbaColor=list(asset.rgba),
                    physicsClientId=self._require_connection(),
                )
            )
        except Exception:
            self._log_asset(f"failed to create mesh visual from {asset.path.name}; using fallback visuals")
            return None
        if visual < 0:
            self._log_asset(f"PyBullet rejected mesh {asset.path.name} (returned {visual}); using fallback visuals")
            return None
        return visual

    def _create_composite_body(
        self,
        collision_shape_id: int,
        assets: Sequence[MeshAsset],
        visual_ids: Sequence[int | None],
        base_position: Sequence[float],
        base_orientation_euler: Sequence[float],
        base_mass: float = 0.0,
    ) -> int | None:
        usable: list[tuple[MeshAsset, int]] = [
            (asset, visual_id)
            for asset, visual_id in zip(assets, visual_ids, strict=True)
            if visual_id is not None
        ]
        if not usable:
            return None

        base_visual = usable[0][1]
        base_orientation = p.getQuaternionFromEuler(list(base_orientation_euler))
        if len(usable) == 1:
            body = p.createMultiBody(
                baseMass=base_mass,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=base_visual,
                basePosition=list(base_position),
                baseOrientation=base_orientation,
                physicsClientId=self.client_id,
            )
            self._apply_body_texture(body, usable[0][0].texture_path)
            return body

        link_count = len(usable) - 1
        body = p.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=base_visual,
            basePosition=list(base_position),
            baseOrientation=base_orientation,
            linkMasses=[0.0] * link_count,
            linkCollisionShapeIndices=[-1] * link_count,
            linkVisualShapeIndices=[visual for _, visual in usable[1:]],
            linkPositions=[[0.0, 0.0, 0.0]] * link_count,
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * link_count,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * link_count,
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * link_count,
            linkParentIndices=[0] * link_count,
            linkJointTypes=[p.JOINT_FIXED] * link_count,
            linkJointAxis=[[0.0, 0.0, 0.0]] * link_count,
            physicsClientId=self.client_id,
        )
        for part_index, (asset, _) in enumerate(usable):
            self._apply_body_texture(
                body_id=body,
                texture_path=asset.texture_path,
                link_index=-1 if part_index == 0 else part_index - 1,
            )
        return body

    def _apply_body_texture(self, body_id: int, texture_path: Path | None, link_index: int = -1) -> None:
        texture_id = self._load_texture_if_present(texture_path)
        if texture_id is None:
            return
        try:
            p.changeVisualShape(
                body_id,
                link_index,
                textureUniqueId=texture_id,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=self._require_connection(),
            )
        except Exception:
            if texture_path is not None:
                self._log_asset(f"failed to apply texture for {texture_path.name}; using mesh without it")

    def _mesh_assets_with_material_colors(
        self,
        archive_name: str,
        obj_name: str,
        target_height_m: float | None = None,
        target_max_dim_m: float | None = None,
        yaw_offset: float = 0.0,
        texture_name: str | None = None,
        rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        up_axis: Literal["x", "y", "z"] = "y",
        roll_offset: float = math.pi / 2.0,
        pitch_offset: float = 0.0,
        scale_xyz_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_gain: float = 1.0,
    ) -> list[MeshAsset]:
        base_asset = self._mesh_asset_from_archive(
            archive_name=archive_name,
            obj_name=obj_name,
            target_height_m=target_height_m,
            target_max_dim_m=target_max_dim_m,
            yaw_offset=yaw_offset,
            texture_name=texture_name,
            rgba=rgba,
            up_axis=up_axis,
            roll_offset=roll_offset,
            pitch_offset=pitch_offset,
            scale_xyz_multiplier=scale_xyz_multiplier,
        )
        if base_asset is None:
            return []
        extracted_root = base_asset.path.parent
        mtl_path = self._find_file(extracted_root, suffix=".mtl")
        diffuse_colors = self._parse_mtl_diffuse_colors(mtl_path)
        parts = self._split_obj_by_material(base_asset.path)
        if not parts:
            return [base_asset]
        assets: list[MeshAsset] = []
        for material_name, part_path in parts:
            assets.append(
                MeshAsset(
                    path=part_path,
                    scale_xyz=base_asset.scale_xyz,
                    roll_offset=base_asset.roll_offset,
                    pitch_offset=base_asset.pitch_offset,
                    yaw_offset=base_asset.yaw_offset,
                    vertical_offset=base_asset.vertical_offset,
                    texture_path=base_asset.texture_path,
                    rgba=tuple(
                        min(1.0, float(channel) * color_gain)
                        for channel in diffuse_colors.get(material_name, base_asset.rgba)[:3]
                    ) + (diffuse_colors.get(material_name, base_asset.rgba)[3],),
                )
            )
        return assets

    def _spawn_static_mesh(
        self,
        assets: Sequence[MeshAsset],
        x: float,
        y: float,
        yaw: float = 0.0,
        z_offset: float = 0.0,
        fallback_half_extents: Sequence[float] | None = None,
        fallback_rgba: Sequence[float] = (0.55, 0.52, 0.46, 1.0),
    ) -> int | None:
        visual_ids = [self._create_mesh_visual(asset) for asset in assets]
        if any(visual_id is not None for visual_id in visual_ids):
            asset_ref = assets[0]
            return self._create_composite_body(
                collision_shape_id=-1,
                assets=assets,
                visual_ids=visual_ids,
                base_position=[x, y, z_offset + asset_ref.vertical_offset],
                base_orientation_euler=[
                    asset_ref.roll_offset,
                    asset_ref.pitch_offset,
                    yaw + asset_ref.yaw_offset,
                ],
            )
        if fallback_half_extents is None:
            return None
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(fallback_half_extents),
            rgbaColor=list(fallback_rgba),
            physicsClientId=self.client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z_offset + float(fallback_half_extents[2])],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            physicsClientId=self.client_id,
        )

    def _spawn_marker(
        self,
        position: Sequence[float],
        shape: int,
        dimensions: Sequence[float],
        rgba: Sequence[float],
    ) -> int:
        if shape == p.GEOM_CYLINDER:
            collision = p.createCollisionShape(
                shape,
                radius=float(dimensions[0]),
                height=float(dimensions[1]),
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                shape,
                radius=float(dimensions[0]),
                length=float(dimensions[1]),
                rgbaColor=list(rgba),
                physicsClientId=self.client_id,
            )
        elif shape == p.GEOM_SPHERE:
            collision = p.createCollisionShape(
                shape,
                radius=float(dimensions[0]),
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                shape,
                radius=float(dimensions[0]),
                rgbaColor=list(rgba),
                physicsClientId=self.client_id,
            )
        else:
            collision = p.createCollisionShape(
                shape,
                halfExtents=list(dimensions),
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                shape,
                halfExtents=list(dimensions),
                rgbaColor=list(rgba),
                physicsClientId=self.client_id,
            )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=list(position),
            physicsClientId=self.client_id,
        )

    def _sync_debug_lines(self) -> None:
        if not self.gui or self.client_id is None:
            return

        while len(self._troop_debug_line_ids) < len(self.troop_positions):
            self._troop_debug_line_ids.append(-1)
        while len(self._drone_debug_line_ids) < len(self.drone_positions):
            self._drone_debug_line_ids.append(-1)

        for idx, pos in enumerate(self.troop_positions):
            start = [float(pos[0]), float(pos[1]), self._ground_offset_z + 0.15]
            end = [float(pos[0]), float(pos[1]), self._ground_offset_z + 3.8]
            self._troop_debug_line_ids[idx] = p.addUserDebugLine(
                start,
                end,
                lineColorRGB=[0.12, 0.95, 0.22],
                lineWidth=4.0,
                lifeTime=0.0,
                replaceItemUniqueId=self._troop_debug_line_ids[idx],
                physicsClientId=self.client_id,
            )

        for idx, pos in enumerate(self.drone_positions):
            start = [float(pos[0]), float(pos[1]), 0.25]
            end = [float(pos[0]), float(pos[1]), float(pos[2]) + 1.8]
            self._drone_debug_line_ids[idx] = p.addUserDebugLine(
                start,
                end,
                lineColorRGB=[1.0, 0.45, 0.05],
                lineWidth=5.0,
                lifeTime=0.0,
                replaceItemUniqueId=self._drone_debug_line_ids[idx],
                physicsClientId=self.client_id,
            )

    def _create_soldier_surrogate_body(
        self,
        collision_shape_id: int,
        base_position: Sequence[float],
        yaw: float,
    ) -> int:
        torso_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.16, 0.10, 0.42],
            rgbaColor=[0.34, 0.43, 0.20, 1.0],
            physicsClientId=self.client_id,
        )
        head_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.12,
            rgbaColor=[0.78, 0.67, 0.58, 1.0],
            physicsClientId=self.client_id,
        )
        limb_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.04, 0.28],
            rgbaColor=[0.18, 0.18, 0.16, 1.0],
            physicsClientId=self.client_id,
        )
        pack_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.08, 0.05, 0.16],
            rgbaColor=[0.20, 0.27, 0.14, 1.0],
            physicsClientId=self.client_id,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=torso_visual,
            basePosition=list(base_position),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            linkMasses=[0.0] * 6,
            linkCollisionShapeIndices=[-1] * 6,
            linkVisualShapeIndices=[head_visual, limb_visual, limb_visual, limb_visual, limb_visual, pack_visual],
            linkPositions=[
                [0.0, 0.0, 0.58],
                [0.0, -0.08, -0.70],
                [0.0, 0.08, -0.70],
                [0.0, -0.18, 0.02],
                [0.0, 0.18, 0.02],
                [-0.11, 0.0, -0.04],
            ],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * 6,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * 6,
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * 6,
            linkParentIndices=[0] * 6,
            linkJointTypes=[p.JOINT_FIXED] * 6,
            linkJointAxis=[[0.0, 0.0, 0.0]] * 6,
            physicsClientId=self.client_id,
        )

    def _create_drone_surrogate_body(
        self,
        collision_shape_id: int,
        base_position: Sequence[float],
        yaw: float,
    ) -> int:
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.24, 0.18, 0.07],
            rgbaColor=[0.78, 0.80, 0.82, 1.0],
            physicsClientId=self.client_id,
        )
        arm_x_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.46, 0.03, 0.02],
            rgbaColor=[0.16, 0.16, 0.16, 1.0],
            physicsClientId=self.client_id,
        )
        arm_y_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.46, 0.02],
            rgbaColor=[0.16, 0.16, 0.16, 1.0],
            physicsClientId=self.client_id,
        )
        rotor_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.11,
            length=0.025,
            rgbaColor=[0.10, 0.10, 0.10, 1.0],
            physicsClientId=self.client_id,
        )
        sensor_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.08,
            rgbaColor=[0.98, 0.62, 0.18, 1.0],
            physicsClientId=self.client_id,
        )
        quat_rotor = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=body_visual,
            basePosition=list(base_position),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            linkMasses=[0.0] * 7,
            linkCollisionShapeIndices=[-1] * 7,
            linkVisualShapeIndices=[arm_x_visual, arm_y_visual, rotor_visual, rotor_visual, rotor_visual, rotor_visual, sensor_visual],
            linkPositions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.52, 0.52, 0.03],
                [-0.52, 0.52, 0.03],
                [-0.52, -0.52, 0.03],
                [0.52, -0.52, 0.03],
                [0.26, 0.0, -0.10],
            ],
            linkOrientations=[quat_rotor, quat_rotor, quat_rotor, quat_rotor, quat_rotor, quat_rotor, [0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * 7,
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * 7,
            linkParentIndices=[0] * 7,
            linkJointTypes=[p.JOINT_FIXED] * 7,
            linkJointAxis=[[0.0, 0.0, 0.0]] * 7,
            physicsClientId=self.client_id,
        )

    def _ground_texture_path(self) -> Path | None:
        archive = self._resource_path("damaged_concrete_floor_4k.blend.zip")
        if not archive.exists():
            return None
        self._resource_cache.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zip_archive:
                names = zip_archive.namelist()
                preferred = [
                    name
                    for name in names
                    if "diff" in name.lower() and name.lower().endswith((".jpg", ".png"))
                ]
                fallback = [
                    name
                    for name in names
                    if name.lower().endswith((".jpg", ".png"))
                ]
                for member in preferred + fallback:
                    target = self._resource_cache / Path(member).name
                    if not target.exists():
                        with zip_archive.open(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    return target
        except (OSError, zipfile.BadZipFile):
            self._log_asset(f"failed to inspect asset archive {archive.name}; using fallback visuals")
        return None

    def _low_poly_soldier_texture_path(self) -> Path | None:
        archive = self._resource_path("low-poly-soldiers-rigged-free.zip")
        return self._extract_zip_member(archive, (".png",))

    def _soldier_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="free_military_soldier_rigged.zip",
            obj_name="free_military_soldier_rigged.obj",
            target_height_m=1.72,
            yaw_offset=math.pi / 2.0,
            rgba=(0.38, 0.46, 0.28, 1.0),
            color_gain=80.0,
        )

    def _sandbag_mesh_asset(self) -> MeshAsset | None:
        return self._mesh_asset_from_archive(
            archive_name="single_sandbag.zip",
            obj_name="single_sandbag.obj",
            target_height_m=0.30,
            rgba=(0.57, 0.45, 0.28, 1.0),
        )

    def _drone_mesh_asset(self) -> MeshAsset | None:
        return self._mesh_asset_from_archive(
            archive_name="fpv_drone.zip",
            obj_name="fpv_drone.obj",
            target_max_dim_m=1.6,
            rgba=(0.90, 0.90, 0.92, 1.0),
        )

    def _salt_dome_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="salt_dome_11_iran.zip",
            obj_name="salt_dome_11_iran.obj",
            target_max_dim_m=min(self.config.world_half_extent_m * 0.25, 68.0),
            texture_name="TerrainNodeMaterial.png",
            scale_xyz_multiplier=(1.0, 0.12, 1.0),
        )

    def _truck_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="low_poly_military_truck.zip",
            obj_name="low_poly_military_truck.obj",
            target_height_m=0.50,
        )

    def _tanker_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="low_poly_truck_tank.zip",
            obj_name="low_poly_truck_tank.obj",
            target_height_m=0.54,
        )

    def _tent_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="militarytent.zip",
            obj_name="militarytent.obj",
            target_height_m=0.56,
        )

    def _container_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="low_poly_container.zip",
            obj_name="low_poly_container.obj",
            target_height_m=0.48,
        )

    def _tank_mesh_assets(self) -> list[MeshAsset]:
        return self._mesh_assets_with_material_colors(
            archive_name="low_poly_tank.zip",
            obj_name="low_poly_tank.obj",
            target_height_m=0.46,
        )

    def _spawn_salt_dome_scene(self, size: float) -> bool:
        self._ground_offset_z = 0.8
        salt_dome_assets = self._salt_dome_mesh_assets()
        salt_dome_body = self._spawn_static_mesh(
            salt_dome_assets,
            x=0.0,
            y=0.0,
            z_offset=self._ground_offset_z,
            fallback_half_extents=[size * 0.8, size * 0.7, 0.25],
            fallback_rgba=[0.70, 0.66, 0.60, 1.0],
        )
        if salt_dome_body is None:
            return False

        self._ruin_ids = []
        if self._plane_id is not None:
            texture_id = self._load_texture_if_present(self._ground_texture_path())
            if texture_id is not None:
                try:
                    p.changeVisualShape(
                        self._plane_id, -1,
                        textureUniqueId=texture_id,
                        rgbaColor=[1.0, 1.0, 1.0, 1.0],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    p.changeVisualShape(
                        self._plane_id, -1,
                        rgbaColor=[0.58, 0.53, 0.47, 1.0],
                        physicsClientId=self.client_id,
                    )
            else:
                p.changeVisualShape(
                    self._plane_id, -1,
                    rgbaColor=[0.58, 0.53, 0.47, 1.0],
                    physicsClientId=self.client_id,
                )

        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, 0.15, 0.7],
            rgbaColor=[0.24, 0.24, 0.26, 1.0],
            physicsClientId=self.client_id,
        )
        for sign in (-1.0, 1.0):
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[0.0, sign * size, 0.7],
                physicsClientId=self.client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[sign * size, 0.0, 0.7],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, math.pi / 2.0]),
                physicsClientId=self.client_id,
            )

        truck_assets = self._truck_mesh_assets()
        tanker_assets = self._tanker_mesh_assets()
        tent_assets = self._tent_mesh_assets()
        container_assets = self._container_mesh_assets()
        tank_assets = self._tank_mesh_assets()
        prop_specs = [
            (truck_assets, -3.5, -2.6, 0.30, [0.24, 0.12, 0.14], [0.30, 0.34, 0.24, 1.0]),
            (truck_assets, 2.9, -2.2, -0.34, [0.24, 0.12, 0.14], [0.30, 0.34, 0.24, 1.0]),
            (tanker_assets, 4.0, -0.2, -0.20, [0.26, 0.13, 0.15], [0.33, 0.34, 0.24, 1.0]),
            (tent_assets, -2.8, 2.8, 0.42, [0.30, 0.22, 0.15], [0.23, 0.29, 0.18, 1.0]),
            (tent_assets, 2.0, 3.5, -0.28, [0.30, 0.22, 0.15], [0.23, 0.29, 0.18, 1.0]),
            (container_assets, -4.5, 0.7, 0.12, [0.24, 0.12, 0.12], [0.34, 0.39, 0.23, 1.0]),
            (container_assets, -3.9, 1.9, -0.22, [0.24, 0.12, 0.12], [0.34, 0.39, 0.23, 1.0]),
            (container_assets, 4.6, 1.5, -0.36, [0.24, 0.12, 0.12], [0.34, 0.39, 0.23, 1.0]),
            (tank_assets, 3.0, 4.1, 0.58, [0.28, 0.15, 0.12], [0.30, 0.33, 0.21, 1.0]),
        ]
        prop_z = self._ground_offset_z + 0.28
        for assets, x, y, yaw, fallback_half_extents, fallback_rgba in prop_specs:
            self._spawn_static_mesh(
                assets=assets,
                x=x,
                y=y,
                yaw=yaw,
                z_offset=prop_z,
                fallback_half_extents=fallback_half_extents,
                fallback_rgba=fallback_rgba,
            )

        self._spawn_sandbag_emplacements()
        self._salt_dome_active = True
        return True

    def _spawn_ground_markers(self) -> None:
        size = self.config.world_half_extent_m
        self._salt_dome_active = False
        self._ground_offset_z = 0.0
        if self._spawn_salt_dome_scene(size):
            return
        if self._plane_id is not None:
            p.changeVisualShape(
                self._plane_id,
                -1,
                rgbaColor=[0.36, 0.30, 0.21, 1.0],
                physicsClientId=self.client_id,
            )
            texture_id = self._load_texture_if_present(self._ground_texture_path())
            if texture_id is not None:
                try:
                    p.changeVisualShape(
                        self._plane_id,
                        -1,
                        textureUniqueId=texture_id,
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    self._log_asset("failed to apply battlefield ground texture; using flat ground color")
        marker_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size * 3.0, size * 3.0, 0.005],
            rgbaColor=[0.36, 0.30, 0.21, 1.0],
            physicsClientId=self.client_id,
        )
        marker_body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0.0, 0.0, 0.005],
            physicsClientId=self.client_id,
        )
        if self._plane_id is not None:
            texture_id = self._load_texture_if_present(self._ground_texture_path())
            if texture_id is not None:
                try:
                    p.changeVisualShape(
                        marker_body,
                        -1,
                        textureUniqueId=texture_id,
                        rgbaColor=[1.0, 1.0, 1.0, 1.0],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    pass

        wall_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, 0.15, 0.7],
            rgbaColor=[0.17, 0.18, 0.2, 1.0],
            physicsClientId=self.client_id,
        )
        for sign in (-1.0, 1.0):
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[0.0, sign * size, 0.7],
                physicsClientId=self.client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=wall_visual,
                basePosition=[sign * size, 0.0, 0.7],
                baseOrientation=p.getQuaternionFromEuler(
                    [0.0, 0.0, math.pi / 2.0]
                ),
                physicsClientId=self.client_id,
            )

        blast_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=1.5,
            length=0.012,
            rgbaColor=[0.15, 0.12, 0.11, 0.95],
            physicsClientId=self.client_id,
        )
        dust_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=2.8,
            length=0.010,
            rgbaColor=[0.31, 0.26, 0.18, 1.0],
            physicsClientId=self.client_id,
        )
        berm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.2, 0.45, 0.45],
            rgbaColor=[0.42, 0.34, 0.22, 1.0],
            physicsClientId=self.client_id,
        )
        concrete_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.3, 1.3, 1.5],
            rgbaColor=[0.31, 0.31, 0.33, 1.0],
            physicsClientId=self.client_id,
        )
        wreck_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.9, 0.45, 0.35],
            rgbaColor=[0.20, 0.21, 0.22, 1.0],
            physicsClientId=self.client_id,
        )

        blast_positions = [
            (-11.5, -7.5),
            (-4.0, 8.5),
            (6.5, -4.5),
            (12.0, 6.0),
            (16.5, -9.0),
            (-15.5, 3.0),
        ]
        for x, y in blast_positions:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=blast_visual,
                basePosition=[x, y, 0.100],
                physicsClientId=self.client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=dust_visual,
                basePosition=[x + 0.4, y - 0.2, 0.115],
                baseOrientation=p.getQuaternionFromEuler(
                    [0.0, 0.0, self.rng.uniform(0.0, math.pi)]
                ),
                physicsClientId=self.client_id,
            )

        for x, y, yaw in [
            (-8.5, -1.5, 0.35),
            (-2.0, 5.5, -0.2),
            (7.0, 1.2, 0.7),
            (13.5, -5.0, -0.55),
            (4.0, -10.0, 0.15),
        ]:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=berm_visual,
                basePosition=[x, y, 0.42],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

        road_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size * 0.88, 1.8, 0.006],
            rgbaColor=[0.23, 0.22, 0.21, 1.0],
            physicsClientId=self.client_id,
        )
        road_body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=road_visual,
            basePosition=[0.0, -1.2, 0.085],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.08]),
            physicsClientId=self.client_id,
        )
        if self._plane_id is not None:
            texture_id = self._load_texture_if_present(self._ground_texture_path())
            if texture_id is not None:
                try:
                    p.changeVisualShape(
                        road_body,
                        -1,
                        textureUniqueId=texture_id,
                        rgbaColor=[0.42, 0.42, 0.42, 1.0],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    pass

        crate_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.48, 0.32, 0.28],
            rgbaColor=[0.39, 0.32, 0.21, 1.0],
            physicsClientId=self.client_id,
        )
        drum_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.24,
            length=0.62,
            rgbaColor=[0.27, 0.31, 0.33, 1.0],
            physicsClientId=self.client_id,
        )
        tarp_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.95, 0.6, 0.08],
            rgbaColor=[0.22, 0.28, 0.24, 1.0],
            physicsClientId=self.client_id,
        )
        for x, y, yaw in [
            (-10.8, 6.0, 0.22),
            (-6.4, 6.8, -0.18),
            (5.0, -7.2, 0.33),
            (11.5, -2.8, -0.41),
            (15.2, 8.4, 0.12),
        ]:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=crate_visual,
                basePosition=[x, y, 0.28],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )
        for x, y in [(-9.2, 4.7), (6.4, -8.9), (12.7, 3.8)]:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=drum_visual,
                basePosition=[x, y, 0.32],
                baseOrientation=p.getQuaternionFromEuler([math.pi / 2.0, 0.0, self.rng.uniform(-0.6, 0.6)]),
                physicsClientId=self.client_id,
            )
        for x, y, yaw in [(-14.2, 7.5, 0.1), (2.2, 11.0, -0.32), (9.8, -10.6, 0.26)]:
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=tarp_visual,
                basePosition=[x, y, 0.08],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )

        self._spawn_terrain_patches()
        self._spawn_rock_clusters()
        self._spawn_sandbag_emplacements()

        ruin_specs = [
            (-13.0, 10.0, concrete_visual, 0.15),
            (10.5, 10.5, concrete_visual, -0.32),
            (2.5, -13.0, concrete_visual, 0.4),
            (-16.0, -11.0, wreck_visual, -0.6),
            (15.0, 1.0, wreck_visual, 0.22),
        ]
        self._ruin_ids = []
        for x, y, visual, yaw in ruin_specs:
            body = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, 0.8 if visual == concrete_visual else 0.38],
                baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                physicsClientId=self.client_id,
            )
            self._ruin_ids.append(body)

    def _spawn_terrain_patches(self) -> None:
        # Flat colored zones laid just above the ground plane to break up the uniform surface.
        # Groups: scorched earth (near blast craters), warm dirt, rock/gravel, dry scrub.
        patch_defs: list[tuple[list[float], list[tuple[float, float, float, float, float]]]] = [
            # scorched earth -- co-located with blast craters so the camera sees burn marks
            ([0.16, 0.12, 0.08, 1.0], [
                (-11.5, -7.5, 4.2, 4.0, 0.18),
                (-4.0,   8.5, 3.8, 3.6, -0.08),
                ( 6.5,  -4.5, 4.0, 3.8, 0.05),
                (12.0,   6.0, 3.4, 3.4, 0.22),
                (16.5,  -9.0, 3.8, 3.6, -0.12),
                (-15.5,  3.0, 3.4, 3.2, 0.08),
            ]),
            # warm dirt patches
            ([0.50, 0.36, 0.20, 1.0], [
                (-15.5,  -8.5, 5.8, 3.6,  0.28),
                (  3.2,  12.5, 4.5, 6.2, -0.18),
                ( -7.5,   1.8, 3.2, 2.8,  0.72),
                ( 18.5,  -4.5, 3.8, 2.6,  0.12),
                (-18.5,  14.5, 2.8, 4.2, -0.55),
                (  8.5, -17.0, 4.0, 2.5,  0.38),
            ]),
            # rock / gravel
            ([0.40, 0.37, 0.34, 1.0], [
                ( -9.5,  -5.5, 2.8, 2.2,  0.42),
                (  9.5,   8.5, 3.2, 2.0, -0.28),
                ( 14.5,  -8.5, 2.2, 3.4,  0.08),
                (-17.5,   2.5, 3.6, 2.4,  0.52),
                ( 20.0,  12.0, 2.5, 3.0, -0.30),
            ]),
            # dry scrub / sparse vegetation
            ([0.29, 0.27, 0.13, 1.0], [
                (  6.2, -14.5, 3.2, 4.8,  0.22),
                ( -3.2, -10.5, 2.8, 2.2, -0.38),
                ( 16.5,   4.5, 4.2, 2.8,  0.58),
                (-12.5,  15.5, 3.8, 3.2, -0.12),
                (-21.0,  -7.0, 3.0, 5.0,  0.45),
            ]),
        ]
        patch_z = 0.015
        for rgba, positions in patch_defs:
            for x, y, hw, hd, yaw in positions:
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[hw, hd, 0.006],
                    rgbaColor=rgba,
                    physicsClientId=self.client_id,
                )
                p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=visual,
                    basePosition=[x, y, patch_z],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )
                patch_z += 0.002  # unique z per patch -- no two patches share a depth plane

    def _spawn_rock_clusters(self) -> None:
        # Clusters of rocks and rubble scattered across the map.
        # Mix of flat slabs, medium upright rocks, and small rubble -- some are
        # ambiguous human height from above and act as false-positive candidates
        # for the perception module.
        cluster_centers = [
            (-17.0, -12.0), (-14.0,  7.0), ( -8.0, -14.0), ( -1.5,  17.0),
            (  5.5,  10.5), (  8.0, -12.0), ( 11.0,  15.0), ( 17.0, -13.0),
            ( 22.0,   5.0), (-22.0,   5.0), (  0.0, -20.0), ( -5.5, -17.0),
            ( 13.5, -17.0), (-20.0, -15.0), ( 19.0,  18.0),
        ]
        for cx, cy in cluster_centers:
            n_rocks = int(self.rng.integers(3, 7))
            for _ in range(n_rocks):
                ox = float(self.rng.uniform(-2.2, 2.2))
                oy = float(self.rng.uniform(-2.2, 2.2))
                rock_type = int(self.rng.integers(0, 3))
                if rock_type == 0:  # flat slab
                    hw = float(self.rng.uniform(0.25, 0.65))
                    hd = float(self.rng.uniform(0.18, 0.50))
                    hz = float(self.rng.uniform(0.06, 0.14))
                elif rock_type == 1:  # medium upright -- ambiguous silhouette from above
                    hw = float(self.rng.uniform(0.15, 0.30))
                    hd = float(self.rng.uniform(0.15, 0.28))
                    hz = float(self.rng.uniform(0.20, 0.45))
                else:  # small rubble
                    hw = float(self.rng.uniform(0.10, 0.22))
                    hd = float(self.rng.uniform(0.08, 0.18))
                    hz = float(self.rng.uniform(0.06, 0.18))
                grey = float(self.rng.uniform(0.28, 0.46))
                tint = float(self.rng.uniform(-0.04, 0.04))
                rgba = [
                    min(1.0, grey + tint),
                    min(1.0, grey),
                    min(1.0, grey - tint * 0.5),
                    1.0,
                ]
                yaw = float(self.rng.uniform(0.0, math.pi))
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[hw, hd, hz],
                    rgbaColor=rgba,
                    physicsClientId=self.client_id,
                )
                p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=visual,
                    basePosition=[cx + ox, cy + oy, hz],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )

    def _spawn_sandbag_emplacements(self) -> None:
        sandbag_asset = self._sandbag_mesh_asset()
        sandbag_visual = self._create_mesh_visual(sandbag_asset)
        fallback_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.38, 0.18, 0.16],
            rgbaColor=[0.55, 0.49, 0.36, 1.0],
            specularColor=[0.06, 0.06, 0.06],
            physicsClientId=self.client_id,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.38, 0.18, 0.16],
            physicsClientId=self.client_id,
        )

        for center_x, center_y, yaw in [
            (-11.0, -3.8, 0.28),
            (-4.6, 9.8, -0.42),
            (8.2, -0.8, 0.18),
            (13.0, 5.8, -0.22),
        ]:
            for offset in (-1.1, -0.35, 0.4, 1.15):
                local = np.array([offset, 0.0], dtype=np.float32)
                rot = np.array(
                    [
                        [math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)],
                    ],
                    dtype=np.float32,
                )
                world = rot @ local
                visual = sandbag_visual if sandbag_visual is not None else fallback_visual
                z = (
                    self._ground_offset_z
                    + (0.18 if sandbag_visual is None or sandbag_asset is None else sandbag_asset.vertical_offset)
                )
                orientation = p.getQuaternionFromEuler(
                    [
                        0.0 if sandbag_asset is None else sandbag_asset.roll_offset,
                        0.0 if sandbag_asset is None else sandbag_asset.pitch_offset,
                        yaw if sandbag_asset is None else yaw + sandbag_asset.yaw_offset,
                    ]
                )
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=[center_x + float(world[0]), center_y + float(world[1]), z],
                    baseOrientation=orientation,
                    physicsClientId=self.client_id,
                )

    def _spawn_troops(self) -> None:
        fallback_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.34, 0.05, 0.88],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            specularColor=[0.04, 0.04, 0.04],
            physicsClientId=self.client_id,
        )
        soldier_assets = self._soldier_mesh_assets()
        soldier_visuals = [self._create_mesh_visual(asset) for asset in soldier_assets]
        mesh_assets_usable = any(visual_id is not None for visual_id in soldier_visuals)
        self._troop_mesh_asset_ref = soldier_assets[0] if mesh_assets_usable and soldier_assets else None
        troop_texture_id = self._load_texture_if_present(self._low_poly_soldier_texture_path())
        self._troop_visual_z_offset = (
            0.0
            if self._troop_mesh_asset_ref is None
            else self._troop_mesh_asset_ref.vertical_offset
        )
        self._troop_visual_yaw_offset = (
            math.pi / 2.0
            if self._troop_mesh_asset_ref is None
            else self._troop_mesh_asset_ref.yaw_offset
        )
        fallback_collision = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=TROOP_COLLIDER_RADIUS_M,
            height=TROOP_COLLIDER_HEIGHT_M,
            physicsClientId=self.client_id,
        )
        mesh_collision = fallback_collision
        if self._troop_mesh_asset_ref is not None:
            _, mesh_collision_orientation = p.invertTransform(
                [0.0, 0.0, 0.0],
                p.getQuaternionFromEuler(
                    [
                        self._troop_mesh_asset_ref.roll_offset,
                        self._troop_mesh_asset_ref.pitch_offset,
                        0.0,
                    ]
                ),
            )
            mesh_collision = p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=TROOP_COLLIDER_RADIUS_M,
                height=TROOP_COLLIDER_HEIGHT_M,
                collisionFrameOrientation=mesh_collision_orientation,
                physicsClientId=self.client_id,
            )
        if self._salt_dome_active:
            self._troop_anchor_bases = np.array(
                [
                    [-3.8, -3.0],
                    [-2.4, 3.3],
                    [3.4, -2.7],
                    [4.5, 2.8],
                ],
                dtype=np.float32,
            )
            self._troop_anchor_dirs = np.array([0.18, -0.28, 0.22, -0.14], dtype=np.float32)
        else:
            self._troop_anchor_bases = np.array(
                [
                    [-12.0, -4.5],
                    [-3.4, 10.0],
                    [9.2, -0.8],
                    [13.2, 6.2],
                ],
                dtype=np.float32,
            )
            self._troop_anchor_dirs = np.array([0.12, -0.34, 0.26, -0.18], dtype=np.float32)

        self.troop_ids = []
        self._troop_marker_ids = []
        for idx in range(self.config.num_troops):
            anchor = idx % len(self._troop_anchor_bases)
            radial = self.rng.uniform(0.8, 3.6)
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            offset = np.array(
                [math.cos(theta) * radial, math.sin(theta) * radial],
                dtype=np.float32,
            )
            self._troop_offsets[idx] = offset
            self._troop_anchor_ids[idx] = anchor
            self._troop_personal_phase[idx] = self.rng.uniform(0.0, 2.0 * math.pi)
            x = float(self._troop_anchor_bases[anchor, 0] + offset[0])
            y = float(self._troop_anchor_bases[anchor, 1] + offset[1])
            z = 1.0 + self._ground_offset_z
            use_mesh = mesh_assets_usable and idx % 2 == 0
            self._troop_mesh_mask[idx] = use_mesh
            collision = mesh_collision if use_mesh else fallback_collision
            spawn_z = z + (self._troop_visual_z_offset if use_mesh else 0.0)
            yaw = self._troop_visual_yaw_offset
            if use_mesh:
                body = self._create_composite_body(
                    collision_shape_id=collision,
                    assets=soldier_assets,
                    visual_ids=soldier_visuals,
                    base_position=[x, y, spawn_z],
                    base_orientation_euler=[
                        0.0 if self._troop_mesh_asset_ref is None else self._troop_mesh_asset_ref.roll_offset,
                        0.0 if self._troop_mesh_asset_ref is None else self._troop_mesh_asset_ref.pitch_offset,
                        yaw,
                    ],
                    base_mass=TROOP_MASS_KG,
                )
            else:
                body = p.createMultiBody(
                    baseMass=TROOP_MASS_KG,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=fallback_visual,
                    basePosition=[x, y, spawn_z],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )
            if body is None:
                body = p.createMultiBody(
                    baseMass=TROOP_MASS_KG,
                    baseCollisionShapeIndex=fallback_collision,
                    baseVisualShapeIndex=fallback_visual,
                    basePosition=[x, y, z],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )
                self._troop_mesh_mask[idx] = False
            p.changeDynamics(
                body,
                -1,
                lateralFriction=0.8,
                rollingFriction=0.05,
                spinningFriction=0.05,
                linearDamping=0.04,
                angularDamping=0.9,
                physicsClientId=self.client_id,
            )
            if troop_texture_id is not None and not self._troop_mesh_mask[idx]:
                try:
                    p.changeVisualShape(
                        body,
                        -1,
                        textureUniqueId=troop_texture_id,
                        rgbaColor=[1.0, 1.0, 1.0, 1.0],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    self._log_asset("failed to apply low-poly soldier texture; using untextured troop visuals")
            self.troop_ids.append(body)
            self._troop_marker_ids.append(
                self._spawn_marker(
                    position=[x, y, z + 1.4],
                    shape=p.GEOM_BOX,
                    dimensions=[0.14, 0.14, 1.2],
                    rgba=[0.18, 0.82, 0.24, 0.92],
                )
            )
            self.troop_positions[idx] = (x, y, z)
            self.troop_yaws[idx] = yaw
        self._sync_debug_lines()

    def _spawn_drones(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.08],
            physicsClientId=self.client_id,
        )
        fallback_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.25, 0.08],
            rgbaColor=self._drone_rgba,
            specularColor=[0.5, 0.5, 0.5],
            physicsClientId=self.client_id,
        )
        if self._resource_path("Drone qui plane 1.glb").exists():
            self._log_asset("Drone qui plane 1.glb is present but PyBullet does not support GLB directly; convert it to OBJ/MTL to use it")
        drone_asset = self._drone_mesh_asset()
        drone_visual = self._create_mesh_visual(drone_asset)
        self._drone_mesh_asset_ref = drone_asset if drone_visual is not None else None
        center = self._troop_centroid()

        self.drone_ids = []
        self._drone_marker_ids = []
        for idx in range(self.config.num_drones):
            theta = 2.0 * math.pi * idx / self.config.num_drones
            x = center[0] + self.config.drone_ring_radius_m * math.cos(theta)
            y = center[1] + self.config.drone_ring_radius_m * math.sin(theta)
            z = self.config.drone_altitude_m
            yaw = math.atan2(center[1] - y, center[0] - x)
            visual = drone_visual if drone_visual is not None else fallback_visual
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[
                    x,
                    y,
                    z if drone_asset is None or drone_visual is None else z + drone_asset.vertical_offset,
                ],
                baseOrientation=p.getQuaternionFromEuler(
                    [
                        0.0 if drone_asset is None or drone_visual is None else drone_asset.roll_offset,
                        0.0 if drone_asset is None or drone_visual is None else drone_asset.pitch_offset,
                        yaw if drone_asset is None or drone_visual is None else yaw + drone_asset.yaw_offset,
                    ]
                ),
                physicsClientId=self.client_id,
            )
            if drone_visual is not None and drone_asset is not None:
                self._apply_body_texture(body, drone_asset.texture_path)
            self.drone_ids.append(body)
            self._drone_marker_ids.append(
                self._spawn_marker(
                    position=[x, y, z + 0.9],
                    shape=p.GEOM_SPHERE,
                    dimensions=[0.85],
                    rgba=[0.98, 0.42, 0.08, 0.98],
                )
            )
            self.drone_positions[idx] = (x, y, z)
            self.drone_yaws[idx] = yaw
        self._sync_debug_lines()

    def _troop_centroid(self) -> np.ndarray:
        return self.troop_positions.mean(axis=0)

    def step(self) -> SimulationSnapshot:
        try:
            client_id = self._require_connection()
            if self.gui:
                self._handle_keyboard()
            self._advance_troops()
            self._advance_drones()
            if self.gui:
                self._update_observer_camera()
            p.stepSimulation(physicsClientId=client_id)
            self._sync_troop_physics_positions()
        except SimulationDisconnectedError:
            raise
        except Exception as exc:
            if (not self.is_connected()) or ("Not connected" in str(exc)):
                raise SimulationDisconnectedError(
                    "PyBullet client disconnected during step"
                ) from exc
            raise
        self.sim_time += self.config.time_step
        return SimulationSnapshot(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions.copy(),
            troop_positions=self.troop_positions.copy(),
        )

    def _update_observer_camera(self) -> None:
        if self.camera_mode == "fpv":
            self._update_fpv_camera()
            return
        if self.camera_mode == "chase":
            self._update_chase_camera()
            return
        self._restore_fpv_drone_visibility()

        focus = self._troop_centroid()
        camera_pos = focus + np.array([0.0, -22.0, 18.0], dtype=np.float32)
        target = focus + np.array([0.0, 0.0, 1.5], dtype=np.float32)
        self._set_debug_camera(camera_pos, target)

    def _update_chase_camera(self) -> None:
        self._restore_fpv_drone_visibility()
        pos = self.drone_positions[self.selected_drone_id]
        yaw = float(self.drone_yaws[self.selected_drone_id])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        camera_pos = pos - forward * 8.0 + np.array([0.0, 0.0, 3.4], dtype=np.float32)
        target = pos + forward * 5.5 + np.array([0.0, 0.0, -1.2], dtype=np.float32)
        self._set_debug_camera(camera_pos, target)

    def _update_fpv_camera(self) -> None:
        self._hide_selected_drone_for_fpv()
        camera = self.camera_pose(self.selected_drone_id)
        camera_pos = camera.eye - camera.forward * 0.15 + camera.up * 0.02
        target = camera.eye + camera.forward * 8.0
        self._set_debug_camera(camera_pos, target)

    def _set_debug_camera(self, camera_pos: np.ndarray, target: np.ndarray) -> None:
        delta = np.asarray(camera_pos, dtype=np.float32) - np.asarray(target, dtype=np.float32)
        distance = max(0.2, float(np.linalg.norm(delta)))
        horiz = max(1e-6, math.hypot(float(delta[0]), float(delta[1])))
        yaw = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        pitch = -math.degrees(math.atan2(float(delta[2]), horiz))
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
            physicsClientId=self.client_id,
        )

    def _hide_selected_drone_for_fpv(self) -> None:
        if self._fpv_hidden_drone_id == self.selected_drone_id:
            return
        self._restore_fpv_drone_visibility()
        body = self.drone_ids[self.selected_drone_id]
        p.changeVisualShape(
            body,
            -1,
            rgbaColor=[self._drone_rgba[0], self._drone_rgba[1], self._drone_rgba[2], 0.0],
            physicsClientId=self.client_id,
        )
        self._fpv_hidden_drone_id = self.selected_drone_id

    def _restore_fpv_drone_visibility(self) -> None:
        if self._fpv_hidden_drone_id is None:
            return
        body = self.drone_ids[self._fpv_hidden_drone_id]
        p.changeVisualShape(
            body,
            -1,
            rgbaColor=self._drone_rgba,
            physicsClientId=self.client_id,
        )
        self._fpv_hidden_drone_id = None

    def _handle_keyboard(self) -> None:
        client_id = self._require_connection()
        self._keyboard_events = p.getKeyboardEvents(physicsClientId=client_id)
        for digit in range(1, min(self.config.num_drones, 9) + 1):
            if self._key_triggered(ord(str(digit))):
                self.selected_drone_id = digit - 1
        if self._key_triggered(ord("c"), ord("C")):
            modes = ("observer", "chase", "fpv")
            idx = (modes.index(self.camera_mode) + 1) % len(modes)
            self.camera_mode = modes[idx]
        if self._key_triggered(ord("b"), ord("B")):
            self.camera_mode = "observer"
        if self._key_triggered(ord("h"), ord("H")):
            self.camera_mode = "chase"
        if self._key_triggered(ord("f"), ord("F")):
            self.camera_mode = "fpv"
        if self._key_triggered(ord("m"), ord("M")):
            self.manual_drone_id = (
                None if self.manual_drone_id == self.selected_drone_id else self.selected_drone_id
            )
        if self._key_triggered(ord("r"), ord("R")):
            if self.manual_drone_id == self.selected_drone_id:
                self.manual_drone_id = None

    def _advance_troops(self) -> None:
        anchor_positions = self._troop_anchor_bases.copy()
        anchor_positions[0] += np.array(
            [self.sim_time * 0.45, 1.2 * math.sin(self.sim_time * 0.22)],
            dtype=np.float32,
        )
        anchor_positions[1] += np.array(
            [0.8 * math.sin(self.sim_time * 0.18), -0.7 * self.sim_time * 0.18],
            dtype=np.float32,
        )
        anchor_positions[2] += np.array(
            [0.95 * self.sim_time * 0.2, 1.0 * math.sin(self.sim_time * 0.31)],
            dtype=np.float32,
        )
        anchor_positions[3] += np.array(
            [-0.65 * math.sin(self.sim_time * 0.21), -0.55 * self.sim_time * 0.16],
            dtype=np.float32,
        )

        for idx, body in enumerate(self.troop_ids):
            anchor = self._troop_anchor_ids[idx]
            phase = float(self._troop_personal_phase[idx])
            spread = self._troop_offsets[idx]
            drift = np.array(
                [
                    0.55 * math.sin(self.sim_time * 0.75 + phase),
                    0.55 * math.cos(self.sim_time * 0.63 + phase * 0.7),
                ],
                dtype=np.float32,
            )
            x = float(anchor_positions[anchor, 0] + spread[0] + drift[0])
            y = float(anchor_positions[anchor, 1] + spread[1] + drift[1])
            z = 1.0 + self._ground_offset_z
            dx = x - float(self.troop_positions[idx, 0])
            dy = y - float(self.troop_positions[idx, 1])
            if math.hypot(dx, dy) > 0.002:
                self._troop_headings[idx] = math.atan2(dy, dx)
            yaw = self._troop_headings[idx]
            use_mesh = bool(self._troop_mesh_mask[idx])
            body_position, _ = p.getBasePositionAndOrientation(
                body,
                physicsClientId=self.client_id,
            )
            body_velocity, _ = p.getBaseVelocity(body, physicsClientId=self.client_id)
            body_z = float(body_position[2])
            self.troop_positions[idx] = (x, y, body_z - (self._troop_visual_z_offset if use_mesh else 0.0))
            self.troop_yaws[idx] = yaw + self._troop_visual_yaw_offset
            p.resetBasePositionAndOrientation(
                body,
                [x, y, body_z],
                p.getQuaternionFromEuler(
                    [
                        0.0 if not use_mesh or self._troop_mesh_asset_ref is None else self._troop_mesh_asset_ref.roll_offset,
                        0.0 if not use_mesh or self._troop_mesh_asset_ref is None else self._troop_mesh_asset_ref.pitch_offset,
                        self.troop_yaws[idx],
                    ]
                ),
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                body,
                linearVelocity=[0.0, 0.0, float(body_velocity[2])],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.client_id,
            )
            if idx < len(self._troop_marker_ids):
                p.resetBasePositionAndOrientation(
                    self._troop_marker_ids[idx],
                    [x, y, float(self.troop_positions[idx, 2]) + 1.4],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
        self._sync_debug_lines()

    def _sync_troop_physics_positions(self) -> None:
        for idx, body in enumerate(self.troop_ids):
            use_mesh = bool(self._troop_mesh_mask[idx])
            body_position, _ = p.getBasePositionAndOrientation(
                body,
                physicsClientId=self.client_id,
            )
            visual_offset = self._troop_visual_z_offset if use_mesh else 0.0
            self.troop_positions[idx, 2] = float(body_position[2]) - visual_offset
            if idx < len(self._troop_marker_ids):
                p.resetBasePositionAndOrientation(
                    self._troop_marker_ids[idx],
                    [
                        float(self.troop_positions[idx, 0]),
                        float(self.troop_positions[idx, 1]),
                        float(self.troop_positions[idx, 2]) + 1.4,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
        self._sync_debug_lines()

    def _advance_drones(self) -> None:
        velocities, desired_yaws = self.policy.commands(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions,
            troop_positions=self.troop_positions,
        )
        if self.manual_drone_id is not None and self.gui:
            velocity, yaw = self._manual_command(self.manual_drone_id)
            velocities[self.manual_drone_id] = velocity
            desired_yaws[self.manual_drone_id] = yaw
        next_positions = self.drone_positions + velocities * self.config.time_step
        limit = self.config.world_half_extent_m - 1.0
        next_positions[:, 0:2] = np.clip(next_positions[:, 0:2], -limit, limit)
        next_positions[:, 2] = np.clip(next_positions[:, 2], 6.0, 20.0)

        self.drone_velocities = velocities
        self.drone_positions = next_positions.astype(np.float32)
        self.drone_yaws = desired_yaws.astype(np.float32)

        for idx, body in enumerate(self.drone_ids):
            p.resetBasePositionAndOrientation(
                body,
                [
                    float(self.drone_positions[idx][0]),
                    float(self.drone_positions[idx][1]),
                    float(self.drone_positions[idx][2])
                    + (0.0 if self._drone_mesh_asset_ref is None else self._drone_mesh_asset_ref.vertical_offset),
                ],
                p.getQuaternionFromEuler(
                    [
                        0.0 if self._drone_mesh_asset_ref is None else self._drone_mesh_asset_ref.roll_offset,
                        0.0 if self._drone_mesh_asset_ref is None else self._drone_mesh_asset_ref.pitch_offset,
                        float(self.drone_yaws[idx])
                        + (0.0 if self._drone_mesh_asset_ref is None else self._drone_mesh_asset_ref.yaw_offset),
                    ]
                ),
                physicsClientId=self.client_id,
            )
            if idx < len(self._drone_marker_ids):
                p.resetBasePositionAndOrientation(
                    self._drone_marker_ids[idx],
                    [
                        float(self.drone_positions[idx][0]),
                        float(self.drone_positions[idx][1]),
                        float(self.drone_positions[idx][2]) + 0.9,
                    ],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
        self._sync_debug_lines()

    def _manual_command(self, drone_idx: int) -> tuple[np.ndarray, float]:
        yaw = float(self.drone_yaws[drone_idx])
        speed = self.config.drone_speed_mps
        vel = np.zeros(3, dtype=np.float32)
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

        up_arrow = getattr(p, "B3G_UP_ARROW", None)
        down_arrow = getattr(p, "B3G_DOWN_ARROW", None)
        left_arrow = getattr(p, "B3G_LEFT_ARROW", None)
        right_arrow = getattr(p, "B3G_RIGHT_ARROW", None)

        if self._key_down(*(key for key in (ord("i"), ord("I"), up_arrow) if key is not None)):
            vel[0:2] += forward * speed
        if self._key_down(*(key for key in (ord("k"), ord("K"), down_arrow) if key is not None)):
            vel[0:2] -= forward * speed
        if self._key_down(*(key for key in (ord("j"), ord("J"), left_arrow) if key is not None)):
            vel[0:2] -= right * speed
        if self._key_down(*(key for key in (ord("l"), ord("L"), right_arrow) if key is not None)):
            vel[0:2] += right * speed
        if self._key_down(ord("u"), ord("U")):
            vel[2] += speed * 0.55
        if self._key_down(ord("o"), ord("O")):
            vel[2] -= speed * 0.55
        if self._key_down(ord("z"), ord("Z")):
            yaw += 1.9 * self.config.time_step * math.pi
        if self._key_down(ord("x"), ord("X")):
            yaw -= 1.9 * self.config.time_step * math.pi

        norm = float(np.linalg.norm(vel))
        if norm > speed:
            vel *= speed / norm
        return vel, yaw

    def camera_pose(self, drone_idx: int) -> CameraPose:
        cam_cfg = self.config.camera
        pos = self.drone_positions[drone_idx]
        yaw = float(self.drone_yaws[drone_idx])
        velocity = self.drone_velocities[drone_idx]
        eye = pos + np.array(
            [
                cam_cfg.forward_offset_m * math.cos(yaw),
                cam_cfg.forward_offset_m * math.sin(yaw),
                -0.08,
            ],
            dtype=np.float32,
        )
        forward_heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right_heading = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        forward_speed = float(np.dot(velocity[0:2], forward_heading))
        lateral_speed = float(np.dot(velocity[0:2], right_heading))
        tilt_rad = math.radians(
            cam_cfg.tilt_deg
            + np.clip(0.9 * forward_speed, -3.0, 5.0)
            + 0.8 * math.sin(self.sim_time * 2.7 + drone_idx * 0.7)
        )
        forward = np.array(
            [
                math.cos(yaw) * math.cos(tilt_rad),
                math.sin(yaw) * math.cos(tilt_rad),
                -math.sin(tilt_rad),
            ],
            dtype=np.float32,
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm
        up = np.cross(right, forward)
        up /= max(float(np.linalg.norm(up)), 1e-6)
        bank_rad = np.clip(-0.045 * lateral_speed, -0.16, 0.16)
        up = up * math.cos(bank_rad) + right * math.sin(bank_rad)
        up /= max(float(np.linalg.norm(up)), 1e-6)
        eye += (
            up * (0.018 * math.sin(self.sim_time * 13.0 + drone_idx))
            + right * (0.012 * math.sin(self.sim_time * 9.0 + drone_idx * 1.9))
        )
        return CameraPose(
            eye=eye,
            forward=forward,
            up=up.astype(np.float32),
            width=cam_cfg.width,
            height=cam_cfg.height,
            fov_deg=cam_cfg.fov_deg,
        )

    def troop_targets(self) -> list[dict]:
        return [
            {
                "id": idx,
                "cls": "troop",
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "z": round(float(pos[2]), 4),
                "width_m": TROOP_DETECTION_WIDTH_M,
                "height_m": TROOP_DETECTION_HEIGHT_M,
            }
            for idx, pos in enumerate(self.troop_positions)
        ]

    def render_drone_camera(self, drone_idx: int) -> np.ndarray:
        client_id = self._require_connection()
        camera = self.camera_pose(drone_idx)
        target = camera.eye + camera.forward * 30.0
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera.eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=camera.up.tolist(),
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera.fov_deg,
            aspect=camera.width / camera.height,
            nearVal=self.config.camera.near,
            farVal=self.config.camera.far,
        )
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if self.gui and hasattr(p, "ER_BULLET_HARDWARE_OPENGL")
            else p.ER_TINY_RENDERER
        )
        try:
            _, _, rgba, _, _ = p.getCameraImage(
                width=camera.width,
                height=camera.height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                lightDirection=[0.42, 0.18, 1.0],
                lightColor=[1.0, 0.96, 0.9],
                lightAmbientCoeff=0.55,
                lightDiffuseCoeff=0.68,
                lightSpecularCoeff=0.08,
                shadow=1,
                renderer=renderer,
                physicsClientId=client_id,
            )
        except Exception as exc:
            raise SimulationDisconnectedError(
                "PyBullet client disconnected during camera render"
            ) from exc
        frame = np.asarray(rgba, dtype=np.uint8).reshape(camera.height, camera.width, 4)[:, :, :3]
        return self._postprocess_frame(drone_idx, frame)

    def render_all_drone_cameras(self) -> list[np.ndarray]:
        return [self.render_drone_camera(idx) for idx in range(self.config.num_drones)]

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            sim_time=self.sim_time,
            drone_positions=self.drone_positions.copy(),
            troop_positions=self.troop_positions.copy(),
        )

    def drone_states(self) -> Sequence[tuple[np.ndarray, float]]:
        return [
            (self.drone_positions[idx].copy(), float(self.drone_yaws[idx]))
            for idx in range(self.config.num_drones)
        ]
