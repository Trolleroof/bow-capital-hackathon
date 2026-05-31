"""MAPPO/CTDE training entrypoint for the CombatOS swarm.

Run:
    uv run --project swarm python -m swarm.train --env-id search-and-interdict
    uv run --project swarm python -m swarm.train --env-id drone-vs-drone --profile garrison

Logs episode reward + coverage to TensorBoard (swarm/runs/) and checkpoints the
best policy (by coverage) under swarm/checkpoints/<env_id>/ so scenario training
runs do not overwrite each other.

The checkpoint stores everything needed to rebuild and ONNX-export the actor
without importing the training code:
    actor_state_dict, obs_dim, act_dim, actor_hidden, log_std_init.
Rebuild with: Actor(obs_dim, act_dim, actor_hidden, log_std_init);
              actor.load_state_dict(ckpt["actor_state_dict"]).
The exportable inference graph is Actor.forward(obs[B,obs_dim]) -> action[B,act_dim].
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from swarm.env_config import config_to_json_dict, make_profile_config
from swarm.eval import EvalResult, eval_policy, is_better
from swarm.mappo import MAPPO, MAPPOConfig
from swarm.scenarios import make_scenario_env

# ----------------------------------------------------------- HYPERPARAMETERS ---
DEFAULTS = dict(
    timesteps=300_000,
    rollout_steps=400,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    update_epochs=8,
    num_minibatches=4,
    actor_hidden=64,
    critic_hidden=128,
    log_std_init=-0.5,
    seed=0,
)

HERE = os.path.dirname(__file__)
CKPT_ROOT = os.path.join(HERE, "checkpoints")
RUNS_DIR = os.path.join(HERE, "runs")


def checkpoint_dir(env_id: str) -> str:
    return os.path.join(CKPT_ROOT, env_id)


def checkpoint_paths(env_id: str) -> dict[str, str]:
    base = checkpoint_dir(env_id)
    return {
        "dir": base,
        "policy": os.path.join(base, "policy.pt"),
        "params": os.path.join(base, "params.json"),
        "meta": os.path.join(base, "meta.json"),
        "events": os.path.join(base, "train-events.ndjson"),
    }


def emit_train_event(path: str, payload: dict) -> None:
    event = {"topic": "train", **payload}
    encoded = json.dumps(event, sort_keys=True)
    print(encoded, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(encoded + "\n")


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--env-id", type=str, default="search-and-interdict")
    p.add_argument("--profile", choices=["garrison", "combat"], default="combat")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    paths = checkpoint_paths(args.env_id)
    os.makedirs(paths["dir"], exist_ok=True)

    battlefield = make_profile_config(args.env_id, args.profile)
    env = make_scenario_env(args.env_id, battlefield=battlefield, seed=args.seed)
    params_hash = env.battlefield_hash()
    save_json(paths["params"], config_to_json_dict(battlefield))
    open(paths["events"], "w", encoding="utf-8").close()

    run_name = args.run_name or f"mappo_{args.env_id}_{args.profile}_{int(time.time())}"
    writer = SummaryWriter(os.path.join(RUNS_DIR, run_name))

    cfg = MAPPOConfig(
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        log_std_init=args.log_std_init,
        seed=args.seed,
    )

    np.random.seed(args.seed)
    algo = MAPPO(env, cfg)

    emit_train_event(
        paths["events"],
        {
            "env_id": args.env_id,
            "profile": args.profile,
            "phase": "init",
            "step": 0,
            "reward_mean": 0.0,
            "coverage": 0.0,
            "task_score": 0.0,
            "primary_metric": env.task_profile.primary_metric,
            "primary_value": 0.0,
            "task_metrics": {},
            "losses": {},
            "params_hash": params_hash,
        },
    )

    random_eval = eval_policy(
        None,
        env_id=args.env_id,
        battlefield=battlefield,
        n_episodes=10,
    )
    print(
        f"[baseline] random policy {random_eval.primary_metric} = "
        f"{random_eval.primary_value:.3f} | coverage = {random_eval.coverage:.3f}"
    )
    writer.add_scalar("eval/random_coverage", random_eval.coverage, 0)
    writer.add_scalar(f"eval/random_{random_eval.primary_metric}", random_eval.primary_value, 0)
    emit_train_event(
        paths["events"],
        {
            "env_id": args.env_id,
            "profile": args.profile,
            "phase": "baseline",
            "step": 0,
            "reward_mean": 0.0,
            "coverage": round(random_eval.coverage, 6),
            "task_score": round(random_eval.task_score, 6),
            "primary_metric": random_eval.primary_metric,
            "primary_value": round(random_eval.primary_value, 6),
            "task_metrics": random_eval.metrics,
            "losses": {},
            "params_hash": params_hash,
        },
    )

    obs = env.reset(seed=args.seed)
    n_updates = max(1, args.timesteps // cfg.rollout_steps)
    global_step = 0
    best_eval: EvalResult | None = None
    best_global_step = 0
    best_train_stats: dict[str, float] = {}
    t0 = time.time()

    for update in range(1, n_updates + 1):
        batch, obs, stats = algo.collect_rollout(obs)
        global_step += cfg.rollout_steps
        train = algo.update(batch)

        ep_rew = float(np.mean(stats["ep_rewards"])) if stats["ep_rewards"] else 0.0
        ep_cov = float(np.mean(stats["ep_coverage"])) if stats["ep_coverage"] else 0.0
        ep_task = float(np.mean(stats["ep_task_score"])) if stats["ep_task_score"] else 0.0
        ep_primary = float(np.mean(stats["ep_primary_value"])) if stats["ep_primary_value"] else ep_task

        writer.add_scalar("charts/episode_reward", ep_rew, global_step)
        writer.add_scalar("charts/episode_coverage", ep_cov, global_step)
        writer.add_scalar("charts/task_score", ep_task, global_step)
        writer.add_scalar(f"charts/{env.task_profile.primary_metric}", ep_primary, global_step)
        writer.add_scalar("losses/pg_loss", train["pg_loss"], global_step)
        writer.add_scalar("losses/v_loss", train["v_loss"], global_step)
        writer.add_scalar("losses/entropy", train["entropy"], global_step)
        writer.add_scalar("losses/approx_kl", train["approx_kl"], global_step)

        losses = {
            "pg_loss": round(float(train["pg_loss"]), 6),
            "v_loss": round(float(train["v_loss"]), 6),
            "entropy": round(float(train["entropy"]), 6),
            "approx_kl": round(float(train["approx_kl"]), 6),
        }
        emit_train_event(
            paths["events"],
            {
                "env_id": args.env_id,
                "profile": args.profile,
                "phase": "update",
                "step": global_step,
                "reward_mean": round(ep_rew, 6),
                "coverage": round(ep_cov, 6),
                "task_score": round(ep_task, 6),
                "primary_metric": env.task_profile.primary_metric,
                "primary_value": round(ep_primary, 6),
                "task_metrics": {
                    "task_score": round(ep_task, 6),
                    env.task_profile.primary_metric: round(ep_primary, 6),
                },
                "losses": losses,
                "params_hash": params_hash,
            },
        )

        if update % 5 == 0 or update == 1:
            sps = int(global_step / max(time.time() - t0, 1e-6))
            print(
                f"upd {update:3d}/{n_updates} step {global_step:7d} "
                f"rew {ep_rew:7.2f} cov {ep_cov:.3f} "
                f"pg {train['pg_loss']:+.3f} v {train['v_loss']:.3f} "
                f"ent {train['entropy']:.2f} kl {train['approx_kl']:.4f} "
                f"[{sps} sps]"
            )

        if update % 5 == 0 or update == n_updates:
            det_eval = eval_policy(
                algo.actor,
                env_id=args.env_id,
                battlefield=battlefield,
                n_episodes=5,
            )
            writer.add_scalar("eval/coverage", det_eval.coverage, global_step)
            writer.add_scalar("eval/task_score", det_eval.task_score, global_step)
            writer.add_scalar(f"eval/{det_eval.primary_metric}", det_eval.primary_value, global_step)
            emit_train_event(
                paths["events"],
                {
                    "env_id": args.env_id,
                    "profile": args.profile,
                    "phase": "eval",
                    "step": global_step,
                    "reward_mean": round(ep_rew, 6),
                    "coverage": round(det_eval.coverage, 6),
                    "task_score": round(det_eval.task_score, 6),
                    "primary_metric": det_eval.primary_metric,
                    "primary_value": round(det_eval.primary_value, 6),
                    "task_metrics": det_eval.metrics,
                    "losses": losses,
                    "params_hash": params_hash,
                },
            )
            if is_better(det_eval, best_eval, args.env_id):
                best_eval = det_eval
                best_global_step = global_step
                best_train_stats = losses
                torch.save(
                    {
                        "env_id": args.env_id,
                        "profile": args.profile,
                        "params_hash": params_hash,
                        "battlefield": config_to_json_dict(battlefield),
                        "actor_state_dict": algo.actor.state_dict(),
                        "obs_dim": env.obs_dim,
                        "act_dim": env.act_dim,
                        "actor_hidden": cfg.actor_hidden,
                        "log_std_init": cfg.log_std_init,
                        "coverage": det_eval.coverage,
                        "task_score": det_eval.task_score,
                        "primary_metric": det_eval.primary_metric,
                        "primary_value": det_eval.primary_value,
                        "task_metrics": det_eval.metrics,
                        "global_step": global_step,
                    },
                    paths["policy"],
                )
                emit_train_event(
                    paths["events"],
                    {
                        "env_id": args.env_id,
                        "profile": args.profile,
                        "phase": "checkpoint",
                        "step": global_step,
                        "reward_mean": round(ep_rew, 6),
                        "coverage": round(det_eval.coverage, 6),
                        "task_score": round(det_eval.task_score, 6),
                        "primary_metric": det_eval.primary_metric,
                        "primary_value": round(det_eval.primary_value, 6),
                        "task_metrics": det_eval.metrics,
                        "losses": losses,
                        "params_hash": params_hash,
                    },
                )

    final_eval = eval_policy(
        algo.actor,
        env_id=args.env_id,
        battlefield=battlefield,
        n_episodes=10,
    )
    if best_eval is None:
        best_eval = final_eval
    print(
        f"\n[result] random {random_eval.primary_metric} = {random_eval.primary_value:.3f} | "
        f"trained final {final_eval.primary_metric} = {final_eval.primary_value:.3f} | "
        f"best checkpoint {best_eval.primary_metric} = {best_eval.primary_value:.3f}"
    )
    print(f"[checkpoint] saved best policy -> {paths['policy']}")
    writer.add_scalar("eval/final_coverage", final_eval.coverage, global_step)
    writer.add_scalar("eval/final_task_score", final_eval.task_score, global_step)
    writer.add_scalar(f"eval/final_{final_eval.primary_metric}", final_eval.primary_value, global_step)
    writer.close()

    meta = {
        "env_id": args.env_id,
        "profile": args.profile,
        "params_hash": params_hash,
        "run_name": run_name,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "rollout_steps": cfg.rollout_steps,
        "global_step": global_step,
        "best_global_step": best_global_step,
        "primary_metric": best_eval.primary_metric,
        "random_primary_value": round(random_eval.primary_value, 6),
        "final_primary_value": round(final_eval.primary_value, 6),
        "best_checkpoint_primary_value": round(best_eval.primary_value, 6),
        "random_coverage": round(random_eval.coverage, 6),
        "final_coverage": round(final_eval.coverage, 6),
        "best_checkpoint_coverage": round(best_eval.coverage, 6),
        "final_task_score": round(final_eval.task_score, 6),
        "best_checkpoint_task_score": round(best_eval.task_score, 6),
        "best_checkpoint_task_metrics": best_eval.metrics,
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "checkpoint": os.path.relpath(paths["policy"], HERE),
        "params": os.path.relpath(paths["params"], HERE),
        "events": os.path.relpath(paths["events"], HERE),
        "losses_at_best": best_train_stats,
        "config": asdict(cfg),
    }
    save_json(paths["meta"], meta)
    emit_train_event(
        paths["events"],
        {
            "env_id": args.env_id,
            "profile": args.profile,
            "phase": "final",
            "step": global_step,
            "reward_mean": round(final_eval.task_score, 6),
            "coverage": round(best_eval.coverage, 6),
            "task_score": round(best_eval.task_score, 6),
            "primary_metric": best_eval.primary_metric,
            "primary_value": round(best_eval.primary_value, 6),
            "task_metrics": best_eval.metrics,
            "losses": best_train_stats,
            "params_hash": params_hash,
        },
    )


if __name__ == "__main__":
    main()
