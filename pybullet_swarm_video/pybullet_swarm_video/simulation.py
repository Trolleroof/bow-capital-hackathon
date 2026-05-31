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


class DroneSurveillanceSimulation:
    def __init__(self, config: SimulationConfig, gui: bool = False) -> None:
        self.config = config
        self.gui = gui
        self.client_id: int | None = None
        self.rng = np.random.default_rng(7)

        self.drone_ids: list[int] = []
        self.troop_ids: list[int] = []
        self._ruin_ids: list[int] = []
        self.drone_positions = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_velocities = np.zeros((config.num_drones, 3), dtype=np.float32)
        self.drone_yaws = np.zeros(config.num_drones, dtype=np.float32)
        self.troop_positions = np.zeros((config.num_troops, 3), dtype=np.float32)
        self.troop_yaws = np.zeros(config.num_troops, dtype=np.float32)
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

    def _split_obj_by_material(self, obj_path: Path) -> list[tuple[str, Path]]:
        cached = self._split_obj_cache.get(obj_path)
        if cached is not None:
            return cached

        target_dir = obj_path.parent / f"{obj_path.stem}_parts"
        stamp_path = target_dir / ".stamp"
        obj_stamp = str(obj_path.stat().st_mtime_ns)
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
                handle.writelines(texcoords)
                handle.writelines(normals)
                handle.write("s off\n")
                handle.writelines(faces)
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
    ) -> MeshAsset | None:
        archive = self._resource_path(archive_name)
        extracted = self._extract_archive_tree(archive, archive.stem)
        if extracted is None:
            return None
        obj_path = self._find_file(extracted, name=obj_name)
        if obj_path is None:
            self._log_asset(f"{archive_name} did not contain {obj_name}; using fallback visuals")
            return None

        bounds = self._obj_bounds(obj_path)
        if bounds is None:
            return None
        mins, maxs = bounds
        size = maxs - mins
        up_idx = 1  # exported assets in this directory are Y-up
        up_size = float(size[up_idx])
        max_dim = float(np.max(size))
        if target_height_m is not None and up_size > 1e-9:
            scale = target_height_m / up_size
        elif target_max_dim_m is not None and max_dim > 1e-9:
            scale = target_max_dim_m / max_dim
        else:
            scale = 1.0

        texture_path = self._find_file(extracted, name=texture_name) if texture_name else None
        return MeshAsset(
            path=obj_path,
            scale_xyz=(scale, scale, scale),
            roll_offset=math.pi / 2.0,
            yaw_offset=yaw_offset,
            vertical_offset=float(-mins[up_idx] * scale),
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
        return visual

    def _create_composite_body(
        self,
        collision_shape_id: int,
        assets: Sequence[MeshAsset],
        visual_ids: Sequence[int | None],
        base_position: Sequence[float],
        base_orientation_euler: Sequence[float],
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
                baseMass=0.0,
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
            baseMass=0.0,
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
        base_asset = self._mesh_asset_from_archive(
            archive_name="free_military_soldier_rigged.zip",
            obj_name="free_military_soldier_rigged.obj",
            target_height_m=1.72,
            yaw_offset=math.pi / 2.0,
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
                    rgba=diffuse_colors.get(material_name, (0.75, 0.75, 0.75, 1.0)),
                )
            )
        return assets

    def _sandbag_mesh_asset(self) -> MeshAsset | None:
        return self._mesh_asset_from_archive(
            archive_name="single_sandbag.zip",
            obj_name="single_sandbag.obj",
            target_height_m=0.30,
            rgba=(0.57, 0.45, 0.28, 1.0),
        )

    def _drone_mesh_asset(self) -> MeshAsset | None:
        return self._mesh_asset_from_archive(
            archive_name="drone_design.zip",
            obj_name="drone_design.obj",
            target_max_dim_m=0.78,
            texture_name="Hely.png",
        )

    def _spawn_ground_markers(self) -> None:
        size = self.config.world_half_extent_m
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
            halfExtents=[size * 0.68, size * 0.42, 0.05],
            rgbaColor=[0.33, 0.24, 0.17, 0.75],
            physicsClientId=self.client_id,
        )
        marker_body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0.0, 0.0, 0.05],
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
            length=0.04,
            rgbaColor=[0.15, 0.12, 0.11, 0.95],
            physicsClientId=self.client_id,
        )
        dust_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=2.8,
            length=0.03,
            rgbaColor=[0.31, 0.26, 0.18, 0.32],
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
                basePosition=[x, y, 0.03],
                physicsClientId=self.client_id,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=dust_visual,
                basePosition=[x + 0.4, y - 0.2, 0.02],
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
            halfExtents=[size * 0.88, 1.8, 0.02],
            rgbaColor=[0.23, 0.22, 0.21, 1.0],
            physicsClientId=self.client_id,
        )
        road_body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=road_visual,
            basePosition=[0.0, -1.2, 0.02],
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
                z = 0.18 if sandbag_visual is None or sandbag_asset is None else sandbag_asset.vertical_offset
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
        collision = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=0.18,
            height=1.0,
            physicsClientId=self.client_id,
        )
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
            z = 1.0
            use_mesh = mesh_assets_usable and idx % 2 == 0
            self._troop_mesh_mask[idx] = use_mesh
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
                )
            else:
                body = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=fallback_visual,
                    basePosition=[x, y, spawn_z],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )
            if body is None:
                body = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=fallback_visual,
                    basePosition=[x, y, z],
                    baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
                    physicsClientId=self.client_id,
                )
                self._troop_mesh_mask[idx] = False
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
            self.troop_positions[idx] = (x, y, z)
            self.troop_yaws[idx] = yaw

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
        drone_asset = self._drone_mesh_asset()
        drone_visual = self._create_mesh_visual(drone_asset)
        self._drone_mesh_asset_ref = drone_asset if drone_visual is not None else None
        center = self._troop_centroid()

        self.drone_ids = []
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
            self.drone_positions[idx] = (x, y, z)
            self.drone_yaws[idx] = yaw

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
        mean_drone = self.drone_positions.mean(axis=0)
        look_target = [
            round(float((focus[0] + mean_drone[0]) * 0.5), 3),
            round(float((focus[1] + mean_drone[1]) * 0.5), 3),
            round(float(focus[2] + 2.0), 3),
        ]
        yaw = 38.0 + 18.0 * math.sin(self.sim_time * 0.08)
        pitch = -36.0
        distance = max(18.0, self.config.world_half_extent_m * 0.95)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=look_target,
            physicsClientId=self.client_id,
        )

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
            z = 1.0
            heading_vec = anchor_positions[anchor] - np.array([x, y], dtype=np.float32)
            yaw = math.atan2(float(heading_vec[1]), float(heading_vec[0]))
            self.troop_positions[idx] = (x, y, z)
            use_mesh = bool(self._troop_mesh_mask[idx])
            self.troop_yaws[idx] = yaw + self._troop_visual_yaw_offset
            p.resetBasePositionAndOrientation(
                body,
                [x, y, z + (self._troop_visual_z_offset if use_mesh else 0.0)],
                p.getQuaternionFromEuler(
                    [
                        0.0 if not use_mesh or self._troop_mesh_asset_ref is None else self._troop_mesh_asset_ref.roll_offset,
                        0.0,
                        self.troop_yaws[idx],
                    ]
                ),
                physicsClientId=self.client_id,
            )

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
                "width_m": 0.55,
                "height_m": 1.75,
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
                shadow=1 if renderer != p.ER_TINY_RENDERER else 0,
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
