from __future__ import annotations

import argparse
from pathlib import Path

from .config import DroneCameraConfig, RecordingConfig, SimulationConfig
from .policies import policy_note
from .record import MultiVideoRecorder, TiledVideoRecorder, tile_frames
from .simulation import DroneSurveillanceSimulation, SimulationDisconnectedError

GUI_MIN_SECONDS = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a tiled first-person PyBullet surveillance video."
    )
    parser.add_argument("--output", type=Path, default=Path("output/drone_spy_demo.mp4"))
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--time-step", type=float, default=1.0 / 30.0)
    parser.add_argument("--drones", type=int, default=5)
    parser.add_argument("--troops", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--per-drone-dir",
        type=Path,
        default=None,
        help="directory for individual drone MP4 outputs; default is derived from --output",
    )
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
    per_drone_dir = args.per_drone_dir or args.output.with_name(f"{args.output.stem}_feeds")
    rec_cfg = RecordingConfig(
        output_path=args.output,
        per_drone_dir=per_drone_dir,
        fps=args.fps,
    )

    print(policy_note())
    print(
        f"recording {sim_cfg.num_drones} drone feeds over {sim_cfg.num_troops} troops "
        f"for {sim_cfg.duration_sec:.1f}s -> {rec_cfg.output_path}"
    )
    print(f"individual drone feeds -> {rec_cfg.per_drone_dir}")

    total_frames = max(1, int(sim_cfg.duration_sec * rec_cfg.fps))
    frame_dt = 1.0 / rec_cfg.fps
    steps_per_frame = max(1, round(frame_dt / sim_cfg.time_step))

    with DroneSurveillanceSimulation(sim_cfg, gui=args.gui) as sim:
        with TiledVideoRecorder(rec_cfg.output_path, rec_cfg.fps) as recorder:
            with MultiVideoRecorder(rec_cfg.per_drone_dir, rec_cfg.fps) as per_drone:
                try:
                    for frame_idx in range(total_frames):
                        for _ in range(steps_per_frame):
                            sim.step()
                        frames = sim.render_all_drone_cameras()
                        per_drone.append(frames)
                        tiled = tile_frames(
                            frames,
                            tile_columns=rec_cfg.tile_columns,
                            gap_px=rec_cfg.tile_gap_px,
                            background_rgb=rec_cfg.background_rgb,
                        )
                        recorder.append(tiled)
                        if frame_idx % max(1, rec_cfg.fps) == 0:
                            print(f"frame {frame_idx + 1}/{total_frames}")
                except SimulationDisconnectedError:
                    print("[sim] PyBullet connection closed; ending run cleanly")

    print(f"wrote {rec_cfg.output_path}")


if __name__ == "__main__":
    main()
