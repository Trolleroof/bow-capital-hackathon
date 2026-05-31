"""Easy-to-hard hunt-and-seek curriculum launcher.

This follows the disaster-rescue training pattern: train an easier stage,
warm-start the next stage from the just-saved checkpoint, then finish on the
deployment distribution. Each stage also gets behavior-cloning steps from the
hunt expert before PPO updates begin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from swarm.hunt_env import HUNT_CURRICULUM_STAGES

HERE = os.path.dirname(__file__)
CKPT_ROOT = os.path.join(HERE, "checkpoints")


def checkpoint_paths(env_id: str) -> dict[str, str]:
    base = os.path.join(CKPT_ROOT, env_id)
    return {"policy": os.path.join(base, "policy.pt")}


DEFAULT_STAGE_WEIGHTS = (0.22, 0.30, 0.48)
DEFAULT_ASSISTS = (0.70, 0.35, 0.0)
DEFAULT_BC_STEPS = (650, 350, 150)


def _parse_csv_floats(raw: str, *, name: str, expected: int) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"--{name} needs {expected} comma-separated values")
    return values


def _parse_csv_ints(raw: str, *, name: str, expected: int) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"--{name} needs {expected} comma-separated values")
    return values


def _allocate_steps(total: int, weights: list[float], floor: int) -> list[int]:
    if total <= 0:
        raise ValueError("--timesteps must be positive")
    raw = [max(floor, int(total * w)) for w in weights]
    delta = total - sum(raw)
    raw[-1] = max(floor, raw[-1] + delta)
    if sum(raw) != total:
        # Very small smoke runs can be below the per-stage floor; split evenly.
        base = max(1, total // len(raw))
        raw = [base for _ in raw]
        raw[-1] += total - sum(raw)
    return raw


def build_stage_commands(args: argparse.Namespace) -> list[list[str]]:
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in HUNT_CURRICULUM_STAGES]
    if unknown:
        raise ValueError(f"unknown hunt stage(s): {', '.join(unknown)}")

    weights = _parse_csv_floats(args.stage_weights, name="stage-weights", expected=len(stages))
    assists = _parse_csv_floats(args.assists, name="assists", expected=len(stages))
    bc_steps = _parse_csv_ints(args.bc_steps, name="bc-steps", expected=len(stages))
    stage_steps = _allocate_steps(args.timesteps, weights, args.stage_floor)
    policy_path = checkpoint_paths("hunt-and-seek")["policy"]

    commands: list[list[str]] = []
    for idx, stage in enumerate(stages):
        cmd = [
            sys.executable,
            "-m",
            "swarm.train",
            "--env-id",
            "hunt-and-seek",
            "--profile",
            args.profile,
            "--timesteps",
            str(stage_steps[idx]),
            "--hunt-stage",
            stage,
            "--pursuit-assist",
            str(assists[idx]),
            "--bc_steps",
            str(bc_steps[idx]),
            "--rollout_steps",
            str(args.rollout_steps),
            "--num_envs",
            str(args.num_envs),
            "--eval-every",
            str(args.eval_every),
            "--seed",
            str(args.seed + idx * 100),
            "--run-name",
            f"{args.run_name}_{stage}_{int(time.time())}",
        ]
        if idx > 0:
            cmd.extend(["--init-from", policy_path])
        commands.append(cmd)
    return commands


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["garrison", "combat"], default="combat")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--stages", default="close,slow,standard")
    p.add_argument("--stage-weights", default=",".join(str(v) for v in DEFAULT_STAGE_WEIGHTS))
    p.add_argument("--stage-floor", type=int, default=1)
    p.add_argument("--assists", default=",".join(str(v) for v in DEFAULT_ASSISTS))
    p.add_argument("--bc-steps", default=",".join(str(v) for v in DEFAULT_BC_STEPS))
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="hunt_curriculum")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    commands = build_stage_commands(args)
    if args.dry_run:
        print(json.dumps({"commands": commands}, indent=2))
        return 0

    env = os.environ.copy()
    for idx, cmd in enumerate(commands, start=1):
        payload = {
            "topic": "train",
            "env_id": "hunt-and-seek",
            "profile": args.profile,
            "phase": "curriculum_stage",
            "stage_index": idx,
            "stage_count": len(commands),
            "command": cmd,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)
        subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
