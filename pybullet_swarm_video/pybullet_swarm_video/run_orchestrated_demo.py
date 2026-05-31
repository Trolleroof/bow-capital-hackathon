from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .bus_client import OrchestratorBusClient
from .config import DroneCameraConfig, OrchestratorConfig, RecordingConfig, SimulationConfig
from .messages import bgr_to_rgb, decode_jpeg_to_bgr, make_frame_id
from .perception_node import GroundTruthPerceptionNode
from .record import MultiVideoRecorder, TiledVideoRecorder, tile_frames
from .simulation import DroneSurveillanceSimulation, SimulationDisconnectedError

GUI_MIN_SECONDS = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyBullet FPV -> orchestrator -> perception -> drone round-trip demo."
    )
    parser.add_argument("--output", type=Path, default=Path("output/drone_spy_roundtrip.mp4"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--time-step", type=float, default=1.0 / 30.0)
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--troops", type=int, default=6)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--control-ws", default="ws://localhost:8000")
    parser.add_argument("--image-ws", default="ws://localhost:8001")
    parser.add_argument(
        "--per-drone-dir",
        type=Path,
        default=None,
        help="directory for one annotated MP4 per drone; default is derived from --output",
    )
    parser.add_argument(
        "--spawn-ground-truth-perception",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="launch the simulated perception worker that subscribes through the orchestrator",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    sim_cfg = SimulationConfig(
        num_drones=args.drones,
        num_troops=args.troops,
        duration_sec=max(args.seconds, GUI_MIN_SECONDS) if args.gui else args.seconds,
        time_step=args.time_step,
        camera=DroneCameraConfig(width=args.width, height=args.height),
    )
    per_drone_dir = args.per_drone_dir or args.output.with_name(f"{args.output.stem}_feeds")
    rec_cfg = RecordingConfig(
        output_path=args.output,
        per_drone_dir=per_drone_dir,
        fps=args.fps,
    )
    orch_cfg = OrchestratorConfig(
        control_ws_url=args.control_ws,
        image_ws_url=args.image_ws,
    )

    total_frames = max(1, int(sim_cfg.duration_sec * rec_cfg.fps))
    frame_dt = 1.0 / rec_cfg.fps
    next_capture_time = 0.0
    captured_frames = 0

    perception_node = GroundTruthPerceptionNode(orch_cfg)
    if args.spawn_ground_truth_perception:
        await perception_node.start()

    try:
        bus = OrchestratorBusClient(orch_cfg)
        try:
            await bus.connect(image_topics=[orch_cfg.hud_topic])
            with DroneSurveillanceSimulation(sim_cfg, gui=args.gui) as sim:
                with TiledVideoRecorder(rec_cfg.output_path, rec_cfg.fps) as recorder:
                    with MultiVideoRecorder(rec_cfg.per_drone_dir, rec_cfg.fps) as per_drone:
                        try:
                            while captured_frames < total_frames:
                                sim.step()
                                if sim.sim_time + sim_cfg.time_step * 0.5 < next_capture_time:
                                    continue

                                raw_frames = sim.render_all_drone_cameras()
                                for drone_id, raw_frame in enumerate(raw_frames):
                                    frame_id = make_frame_id(drone_id, captured_frames)
                                    camera = sim.camera_pose(drone_id)
                                    source = f"drone:{drone_id}"
                                    frame_payload = {
                                        "t": round(sim.sim_time, 3),
                                        "seq": captured_frames,
                                        "frame_id": frame_id,
                                        "drone_id": drone_id,
                                        "source": source,
                                        "width": camera.width,
                                        "height": camera.height,
                                        "encoding": "jpeg",
                                    }
                                    await bus.publish_rgb_frame(orch_cfg.raw_topic, frame_payload, raw_frame)
                                    if drone_id == orch_cfg.dashboard_drone_id:
                                        await bus.publish_rgb_frame(
                                            orch_cfg.dashboard_raw_topic,
                                            frame_payload,
                                            raw_frame,
                                        )
                                    await bus.publish_control(
                                        orch_cfg.state_topic,
                                        {
                                            "t": round(sim.sim_time, 3),
                                            "seq": captured_frames,
                                            "frame_id": frame_id,
                                            "drone_id": drone_id,
                                            "source": source,
                                            "width": camera.width,
                                            "height": camera.height,
                                            "fov_deg": camera.fov_deg,
                                            "eye": [round(float(v), 4) for v in camera.eye],
                                            "forward": [round(float(v), 5) for v in camera.forward],
                                            "up": [round(float(v), 5) for v in camera.up],
                                            "targets": sim.troop_targets(),
                                        },
                                    )

                                hud_frames = await _collect_hud_frames(
                                    bus,
                                    seq=captured_frames,
                                    num_drones=sim_cfg.num_drones,
                                    timeout=max(0.35, frame_dt * 2.5),
                                )
                                ordered = []
                                for drone_id, raw_frame in enumerate(raw_frames):
                                    hud = hud_frames.get(drone_id)
                                    ordered.append(hud if hud is not None else raw_frame)
                                per_drone.append(ordered)
                                recorder.append(
                                    tile_frames(
                                        ordered,
                                        tile_columns=rec_cfg.tile_columns,
                                        gap_px=rec_cfg.tile_gap_px,
                                        background_rgb=rec_cfg.background_rgb,
                                    )
                                )
                                captured_frames += 1
                                next_capture_time += frame_dt
                                if captured_frames % max(1, rec_cfg.fps) == 0:
                                    print(f"frame {captured_frames}/{total_frames}")
                        except SimulationDisconnectedError:
                            print("[sim] PyBullet connection closed; ending run cleanly")
        finally:
            await bus.close()
    finally:
        if args.spawn_ground_truth_perception:
            await perception_node.stop()

    print(f"wrote {rec_cfg.output_path}")


async def _collect_hud_frames(
    bus: OrchestratorBusClient,
    seq: int,
    num_drones: int,
    timeout: float,
) -> dict[int, object]:
    deadline = asyncio.get_running_loop().time() + timeout
    frames: dict[int, object] = {}
    while len(frames) < num_drones:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        try:
            msg = await bus.next_image(timeout=remaining)
        except asyncio.TimeoutError:
            break
        if msg.get("topic") != bus.config.hud_topic:
            continue
        if msg.get("seq") != seq:
            continue
        drone_id = msg.get("drone_id")
        if not isinstance(drone_id, int):
            continue
        frames[drone_id] = bgr_to_rgb(decode_jpeg_to_bgr(msg["data"]))
    return frames


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
