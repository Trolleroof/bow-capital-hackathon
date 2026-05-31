from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import DroneCameraConfig, RecordingConfig, SimulationConfig
from .policies import policy_note
from .record import TiledVideoRecorder
from .simulation import DroneSurveillanceSimulation, SimulationDisconnectedError

GUI_MIN_SECONDS = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a first-person PyBullet surveillance video for one drone camera."
    )
    parser.add_argument("--output", type=Path, default=Path("output/drone_spy_demo.mp4"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--time-step", type=float, default=1.0 / 30.0)
    parser.add_argument("--drones", type=int, default=5)
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
    return parser.parse_args()


def main() -> None:
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

    print(policy_note())

    camera_idx = max(0, min(args.camera, sim_cfg.num_drones - 1))
    print(
        f"recording drone {camera_idx} camera over {sim_cfg.num_troops} troops "
        f"for {sim_cfg.duration_sec:.1f}s -> {rec_cfg.output_path}"
    )
    if args.gui:
        print("[sim] press 1-9 to switch which drone camera is recorded live")

    total_frames = max(1, int(sim_cfg.duration_sec * rec_cfg.fps))
    frame_dt = 1.0 / rec_cfg.fps
    next_capture_time = 0.0
    captured_frames = 0

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
                                time.sleep(remaining)
                        continue

                    active_camera = sim.selected_drone_id if args.gui else camera_idx
                    frame = sim.render_drone_camera(active_camera)
                    recorder.append(frame)
                    captured_frames += 1
                    next_capture_time += frame_dt

                    if args.gui:
                        remaining = step_deadline - time.monotonic()
                        if remaining > 0.0:
                            time.sleep(remaining)

                    if captured_frames % max(1, rec_cfg.fps) == 0:
                        print(f"frame {captured_frames}/{total_frames} (drone {active_camera})")
            except SimulationDisconnectedError:
                print("[sim] PyBullet connection closed; ending run cleanly")

    print(f"wrote {rec_cfg.output_path}")


if __name__ == "__main__":
    main()
