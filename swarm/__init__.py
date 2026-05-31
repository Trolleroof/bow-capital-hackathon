"""Outcast Virus swarm vertical — decentralized multi-agent RL (MAPPO/CTDE).

Phase 0 exposes the point-mass coverage env (`env`) and the WebSocket bus
(`bus`). Phases 1+ add MAPPO training, ONNX export, and edge inference.
"""

from .env import SwarmEnv
from .scenarios import SCENARIOS, get_scenario, list_scenarios, make_scenario_env

__all__ = [
    "SCENARIOS",
    "SwarmEnv",
    "get_scenario",
    "list_scenarios",
    "make_scenario_env",
]
