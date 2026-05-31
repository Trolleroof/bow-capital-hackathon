from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from .bus_client import OrchestratorBusClient
from .config import DroneCameraConfig, OrchestratorConfig, RecordingConfig, SimulationConfig
from .messages import bgr_to_rgb, decode_jpeg_to_bgr, make_frame_id
from .perception_node import GroundTruthPerceptionNode
from .record import TiledVideoRecorder
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
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="drone index whose camera to record (0-based); in --gui mode, 1-9 keys change this live",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--control-ws", default="ws://localhost:8000")
    parser.add_argument("--image-ws", default="ws://localhost:8001")
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
    rec_cfg = RecordingConfig(
        output_path=args.output,
        fps=args.fps,
    )
    orch_cfg = OrchestratorConfig(
        control_ws_url=args.control_ws,
        image_ws_url=args.image_ws,
    )

    camera_idx = max(0, min(args.camera, sim_cfg.num_drones - 1))
    print(
        f"recording drone {camera_idx} camera (orchestrated) over {sim_cfg.num_troops} troops "
        f"for {sim_cfg.duration_sec:.1f}s -> {rec_cfg.output_path}"
    )
    if args.gui:
        print("[sim] press 1-9 to switch which drone camera is recorded live")

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
                sim.selected_drone_id = camera_idx
                with TiledVideoRecorder(rec_cfg.output_path, rec_cfg.fps) as recorder:
                    step_deadline = time.monotonic()
                    try:
                        while captured_frames < total_frames:
                            step_deadline += sim_cfg.time_step
                            sim.step()

                            if sim.sim_time + sim_cfg.time_step * 0.5 < next_capture_time:
                                if args.gui:
                                    remaining = step_deadline - time.monotonic()
                                    if remaining > 0.0:
                                        await asyncio.sleep(remaining)
                                continue

                            active_camera = sim.selected_drone_id if args.gui else camera_idx
                            raw_frame = sim.render_drone_camera(active_camera)

                            frame_id = make_frame_id(active_camera, captured_frames)
                            camera = sim.camera_pose(active_camera)
                            source = f"drone:{active_camera}"
                            frame_payload = {
                                "t": round(sim.sim_time, 3),
                                "seq": captured_frames,
                                "frame_id": frame_id,
                                "drone_id": active_camera,
                                "source": source,
                                "width": camera.width,
                                "height": camera.height,
                                "encoding": "jpeg",
                            }
                            await bus.publish_rgb_frame(orch_cfg.raw_topic, frame_payload, raw_frame)
                            if active_camera == orch_cfg.dashboard_drone_id:
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
                                    "drone_id": active_camera,
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

                            hud_frame = await _collect_hud_frame(
                                bus,
                                seq=captured_frames,
                                drone_id=active_camera,
                                timeout=max(0.35, frame_dt * 2.5),
                            )
                            recorder.append(hud_frame if hud_frame is not None else raw_frame)
                            captured_frames += 1
                            next_capture_time += frame_dt

                            if args.gui:
                                remaining = step_deadline - time.monotonic()
                                if remaining > 0.0:
                                    await asyncio.sleep(remaining)

                            if captured_frames % max(1, rec_cfg.fps) == 0:
                                print(f"frame {captured_frames}/{total_frames} (drone {active_camera})")
                    except SimulationDisconnectedError:
                        print("[sim] PyBullet connection closed; ending run cleanly")
        finally:
            await bus.close()
    finally:
        if args.spawn_ground_truth_perception:
            await perception_node.stop()

    print(f"wrote {rec_cfg.output_path}")


async def _collect_hud_frame(
    bus: OrchestratorBusClient,
    seq: int,
    drone_id: int,
    timeout: float,
) -> object | None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        try:
            msg = await bus.next_image(timeout=remaining)
        except asyncio.TimeoutError:
            return None
        if msg.get("topic") != bus.config.hud_topic:
            continue
        if msg.get("seq") != seq:
            continue
        if msg.get("drone_id") != drone_id:
            continue
        return bgr_to_rgb(decode_jpeg_to_bgr(msg["data"]))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
