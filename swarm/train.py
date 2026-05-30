"""MAPPO/CTDE training entrypoint for the CombatOS swarm.

Run:
    cd swarm && uv run python -m swarm.train                 # short default budget
    cd swarm && uv run python -m swarm.train --timesteps 400000

Logs episode reward + coverage to TensorBoard (swarm/runs/) and checkpoints the
best policy (by coverage) to swarm/checkpoints/policy.pt.

The checkpoint stores everything Phase 2 needs to rebuild and ONNX-export the
actor WITHOUT importing this training code:
    actor_state_dict, obs_dim, act_dim, actor_hidden, log_std_init.
Rebuild with: Actor(obs_dim, act_dim, actor_hidden, log_std_init);
              actor.load_state_dict(ckpt["actor_state_dict"]).
The exportable inference graph is Actor.forward(obs[B,obs_dim]) -> action[B,act_dim].
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from swarm.env import SwarmEnv
from swarm.mappo import MAPPO, MAPPOConfig
from swarm.models import Actor

# ----------------------------------------------------------- HYPERPARAMETERS ---
# (Easy to find. Short default budget converges in a few minutes on CPU.)
DEFAULTS = dict(
    timesteps=300_000,      # total env steps (CLI: --timesteps)
    rollout_steps=400,      # = one full episode per update
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

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "policy.pt")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# --------------------------------------------------------------- baselines ---
def eval_policy(actor: Actor | None, n_episodes: int = 10, seed: int = 1234,
                deterministic: bool = True) -> float:
    """Mean final coverage over n_episodes. actor=None => random policy."""
    env = SwarmEnv(seed=seed)
    covs = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        info = {"coverage": 0.0}
        while not done:
            if actor is None:
                a = env.rng.uniform(-1, 1, size=(env.n, env.act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs)
                    if deterministic:
                        a = actor(obs_t).numpy().astype(np.float32)
                    else:
                        a, _ = actor.sample(obs_t)
                        a = a.numpy().astype(np.float32)
            obs, _, dones, info = env.step(a)
            done = bool(dones.any())
        covs.append(info["coverage"])
    return float(np.mean(covs))


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_name = args.run_name or f"mappo_{int(time.time())}"
    writer = SummaryWriter(os.path.join(RUNS_DIR, run_name))

    cfg = MAPPOConfig(
        rollout_steps=args.rollout_steps, lr=args.lr, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_coef=args.clip_coef, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches, actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden, log_std_init=args.log_std_init, seed=args.seed,
    )

    np.random.seed(args.seed)
    env = SwarmEnv(seed=args.seed)
    algo = MAPPO(env, cfg)

    # random baseline (for comparison + the slide)
    random_cov = eval_policy(None, n_episodes=10)
    print(f"[baseline] random policy mean coverage = {random_cov:.3f}")
    writer.add_scalar("eval/random_coverage", random_cov, 0)

    obs = env.reset(seed=args.seed)
    n_updates = args.timesteps // (cfg.rollout_steps)
    global_step = 0
    best_cov = -1.0
    t0 = time.time()

    for update in range(1, n_updates + 1):
        batch, obs, stats = algo.collect_rollout(obs)
        global_step += cfg.rollout_steps
        train = algo.update(batch)

        ep_rew = float(np.mean(stats["ep_rewards"])) if stats["ep_rewards"] else 0.0
        ep_cov = float(np.mean(stats["ep_coverage"])) if stats["ep_coverage"] else 0.0

        writer.add_scalar("charts/episode_reward", ep_rew, global_step)
        writer.add_scalar("charts/episode_coverage", ep_cov, global_step)
        writer.add_scalar("losses/pg_loss", train["pg_loss"], global_step)
        writer.add_scalar("losses/v_loss", train["v_loss"], global_step)
        writer.add_scalar("losses/entropy", train["entropy"], global_step)
        writer.add_scalar("losses/approx_kl", train["approx_kl"], global_step)

        if update % 5 == 0 or update == 1:
            sps = int(global_step / (time.time() - t0))
            print(f"upd {update:3d}/{n_updates} step {global_step:7d} "
                  f"rew {ep_rew:7.2f} cov {ep_cov:.3f} "
                  f"pg {train['pg_loss']:+.3f} v {train['v_loss']:.3f} "
                  f"ent {train['entropy']:.2f} kl {train['approx_kl']:.4f} "
                  f"[{sps} sps]")

        # checkpoint best by a quick deterministic eval every few updates
        if update % 5 == 0 or update == n_updates:
            det_cov = eval_policy(algo.actor, n_episodes=5)
            writer.add_scalar("eval/coverage", det_cov, global_step)
            if det_cov > best_cov:
                best_cov = det_cov
                torch.save({
                    "actor_state_dict": algo.actor.state_dict(),
                    "obs_dim": env.obs_dim,
                    "act_dim": env.act_dim,
                    "actor_hidden": cfg.actor_hidden,
                    "log_std_init": cfg.log_std_init,
                    "coverage": det_cov,
                    "global_step": global_step,
                }, CKPT_PATH)

    # final report
    final_cov = eval_policy(algo.actor, n_episodes=10)
    print(f"\n[result] random coverage = {random_cov:.3f} | "
          f"trained (final) coverage = {final_cov:.3f} | "
          f"best checkpoint coverage = {best_cov:.3f}")
    print(f"[checkpoint] saved best policy -> {CKPT_PATH}")
    writer.add_scalar("eval/final_coverage", final_cov, global_step)
    writer.close()


if __name__ == "__main__":
    main()
