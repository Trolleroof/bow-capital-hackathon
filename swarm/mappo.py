"""MAPPO + CTDE, CleanRL-style, in one file.

Centralized Training, Decentralized Execution:
  * ONE shared-parameter actor consumes each agent's LOCAL obs (36-dim).
  * ONE centralized critic consumes the GLOBAL state (env.global_state()),
    used only to compute advantages at train time. Never deployed.

The N agents share a team reward, so we have a single value per timestep
(critic on global state) and broadcast the resulting advantage/return to every
live agent's transitions. This is the standard "shared-reward MAPPO" setup.

Standard tricks: GAE, advantage normalization, PPO clip, value-loss clip,
entropy bonus, gradient clipping, several update epochs over minibatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from swarm.env import SwarmEnv
from swarm.models import Actor, Critic


@dataclass
class MAPPOConfig:
    # rollout / optimization
    rollout_steps: int = 400          # env steps per update (one full episode)
    update_epochs: int = 8
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    vf_clip: float = 0.2
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    # net sizes
    actor_hidden: int = 64
    critic_hidden: int = 128
    log_std_init: float = -0.5
    # misc
    device: str = "cpu"
    seed: int = 0


class MAPPO:
    """Holds the actor, critic, optimizer, and the rollout/update logic."""

    def __init__(self, env: SwarmEnv, cfg: MAPPOConfig):
        self.env = env
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        self.actor = Actor(env.obs_dim, env.act_dim, cfg.actor_hidden,
                           cfg.log_std_init).to(self.device)
        self.critic = Critic(env.state_dim, cfg.critic_hidden).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr, eps=1e-5,
        )

    # ------------------------------------------------------------ rollout ---
    @torch.no_grad()
    def collect_rollout(self, obs: np.ndarray):
        """Run one rollout of cfg.rollout_steps. Returns a batch dict + the
        next obs (for the next rollout) and episode stats.

        Buffers are stored per (step, agent) for actor data and per step for
        the shared critic. Dead agents are masked out of the policy loss.
        """
        cfg, env, n = self.cfg, self.env, self.env.n
        T = cfg.rollout_steps

        obs_buf = np.zeros((T, n, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, n, env.act_dim), dtype=np.float32)
        logp_buf = np.zeros((T, n), dtype=np.float32)
        alive_buf = np.zeros((T, n), dtype=np.float32)
        state_buf = np.zeros((T, env.state_dim), dtype=np.float32)
        rew_buf = np.zeros(T, dtype=np.float32)     # shared team reward
        val_buf = np.zeros(T, dtype=np.float32)
        done_buf = np.zeros(T, dtype=np.float32)

        ep_rewards, ep_coverage = [], []
        ep_ret = 0.0

        for t in range(T):
            state = env.global_state()
            obs_t = torch.as_tensor(obs, device=self.device)
            action, logp = self.actor.sample(obs_t)
            value = self.critic(torch.as_tensor(state, device=self.device).unsqueeze(0))

            a_np = action.cpu().numpy().astype(np.float32)
            next_obs, rewards, dones, info = env.step(a_np)

            obs_buf[t] = obs
            act_buf[t] = a_np
            logp_buf[t] = logp.cpu().numpy()
            alive_buf[t] = env.alive.astype(np.float32)
            state_buf[t] = state
            # team reward: take the shared scalar (max over agents; dead get 0)
            rew_buf[t] = float(rewards.max()) if rewards.size else 0.0
            val_buf[t] = float(value.item())
            done_buf[t] = float(dones.any())

            ep_ret += rew_buf[t]
            obs = next_obs

            if dones.any():
                ep_rewards.append(ep_ret)
                ep_coverage.append(info["coverage"])
                ep_ret = 0.0
                obs = env.reset(seed=int(env.rng.integers(1 << 30)))

        # bootstrap value for the final state
        with torch.no_grad():
            last_state = torch.as_tensor(env.global_state(), device=self.device).unsqueeze(0)
            last_val = float(self.critic(last_state).item())

        # ---- GAE on the shared (per-step) reward/value track ----
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterm = 1.0 - done_buf[t]
            next_val = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_val * next_nonterm - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * last_gae
            adv[t] = last_gae
        ret = adv + val_buf

        batch = {
            "obs": torch.as_tensor(obs_buf.reshape(T * n, env.obs_dim)),
            "act": torch.as_tensor(act_buf.reshape(T * n, env.act_dim)),
            "logp": torch.as_tensor(logp_buf.reshape(T * n)),
            "alive": torch.as_tensor(alive_buf.reshape(T * n)),
            # broadcast per-step advantage/return/value to each agent
            "adv": torch.as_tensor(np.repeat(adv, n)),
            "ret": torch.as_tensor(np.repeat(ret, n)),
            "state": torch.as_tensor(state_buf),       # (T, state_dim) for critic
            "val": torch.as_tensor(val_buf),           # (T,)
            "ret_step": torch.as_tensor(ret),          # (T,)
        }
        stats = {
            "ep_rewards": ep_rewards,
            "ep_coverage": ep_coverage,
        }
        return batch, obs, stats

    # ------------------------------------------------------------ update ----
    def update(self, batch: dict):
        cfg = self.cfg
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        old_logp = batch["logp"].to(self.device)
        alive = batch["alive"].to(self.device)
        adv = batch["adv"].to(self.device)
        ret_agent = batch["ret"].to(self.device)
        state = batch["state"].to(self.device)
        old_val = batch["val"].to(self.device)
        ret_step = batch["ret_step"].to(self.device)

        # advantage normalization (over live transitions)
        live_mask = alive > 0.5
        if live_mask.sum() > 1:
            a_live = adv[live_mask]
            adv = (adv - a_live.mean()) / (a_live.std() + 1e-8)

        n_agent = obs.shape[0]
        n_step = state.shape[0]
        agent_idx = np.arange(n_agent)
        step_idx = np.arange(n_step)
        mb_agent = max(1, n_agent // cfg.num_minibatches)
        mb_step = max(1, n_step // cfg.num_minibatches)

        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(agent_idx)
            np.random.shuffle(step_idx)
            for start in range(0, n_agent, mb_agent):
                ai = agent_idx[start:start + mb_agent]
                # --- actor (policy) loss over a minibatch of agent transitions ---
                mb_obs = obs[ai]
                mb_act = act[ai]
                mb_alive = alive[ai]
                logp, entropy = self.actor.evaluate(mb_obs, mb_act)
                ratio = torch.exp(logp - old_logp[ai])
                mb_adv = adv[ai]

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_per = torch.max(pg1, pg2)
                # mask dead agents out of the policy loss
                denom = mb_alive.sum().clamp(min=1.0)
                pg_loss = (pg_per * mb_alive).sum() / denom
                ent = (entropy * mb_alive).sum() / denom

                # --- critic (value) loss over a minibatch of timesteps ---
                si = step_idx[start % n_step: start % n_step + mb_step]
                if len(si) == 0:
                    si = step_idx[:mb_step]
                new_val = self.critic(state[si])
                v_unclipped = (new_val - ret_step[si]).pow(2)
                v_clipped = old_val[si] + torch.clamp(
                    new_val - old_val[si], -cfg.vf_clip, cfg.vf_clip
                )
                v_clipped = (v_clipped - ret_step[si]).pow(2)
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean()
                stats["pg_loss"] += float(pg_loss.item())
                stats["v_loss"] += float(v_loss.item())
                stats["entropy"] += float(ent.item())
                stats["approx_kl"] += float(approx_kl.item())
                n_updates += 1

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats
