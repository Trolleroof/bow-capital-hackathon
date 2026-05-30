"""Neural networks for MAPPO / CTDE.

Two nets:

  * ``Actor`` — shared-parameter decentralized policy. Consumes ONLY the 36-dim
    local observation. Plain MLP (Linear + Tanh), Gaussian policy with a
    state-independent learnable log-std. Designed to be trivially exportable to
    ONNX in Phase 2: ``Actor.forward(obs)`` maps obs -> deterministic mean
    action (tanh-squashed to [-1, 1]). No sampling / no exotic ops in forward.

  * ``Critic`` — centralized value function. Consumes ONLY ``env.global_state()``
    (train-time only, never deployed). Plain MLP -> scalar value.

--------------------------------------------------------------------------------
PHASE 2 / ONNX CONTRACT (the actor's inference signature)
--------------------------------------------------------------------------------
    input :  obs   float32  shape (B, OBS_DIM)   (OBS_DIM = 36 for defaults)
    output:  mean_action float32 shape (B, ACT_DIM)  (ACT_DIM = 2), in [-1, 1]

``Actor.forward`` IS the inference graph to export (deterministic mean action).
Sampling / log-prob / entropy live in separate methods used at train time only,
so they never enter the exported graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """Plain Linear + Tanh MLP (two hidden layers). ONNX-friendly ops only."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Shared-parameter Gaussian policy over local obs. Squashed to [-1, 1].

    forward(obs) -> deterministic mean action (tanh of the Gaussian mean). This
    is exactly the graph Phase 2 exports to ONNX.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.net = _mlp(obs_dim, hidden, act_dim)
        # state-independent learnable log-std (one per action dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action in [-1, 1]. (ONNX inference graph.)"""
        return torch.tanh(self.net(obs))

    # ---- training-only helpers (never exported) ----
    def _dist(self, obs: torch.Tensor):
        """Pre-tanh Gaussian distribution over raw actions."""
        mean = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs: torch.Tensor):
        """Sample a squashed action; return (action, log_prob).

        Uses the tanh-squashed-Gaussian change-of-variables correction so
        log-probs are consistent with actions in [-1, 1].
        """
        dist = self._dist(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(-1)
        # tanh correction
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Given stored squashed actions, return (log_prob, entropy)."""
        dist = self._dist(obs)
        # invert tanh to recover raw action (clamp for numerical safety)
        a = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(a)
        log_prob = dist.log_prob(raw).sum(-1)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
        # entropy of the underlying Gaussian (a fine, cheap proxy for the bonus)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized value function over the global state. Train-time only."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = _mlp(state_dim, hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
