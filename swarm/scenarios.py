"""Scenario registry for hard-coded CombatOS gym environments.

The current RL core is still the point-mass `SwarmEnv`, but issue #7 needs
operator-facing scenarios that can be selected consistently across the frontend,
docs, and training entry points. This module provides that mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .env import SwarmEnv
from .env_config import BattlefieldConfig, get_scenario_defaults


@dataclass(frozen=True)
class ScenarioDefinition:
    id: str
    name: str
    summary: str
    env_kwargs: dict[str, int | float] = field(default_factory=dict)
    observation: str = ""
    action: str = ""
    reward: str = ""


SCENARIOS: dict[str, ScenarioDefinition] = {
    "drone-vs-drone": ScenarioDefinition(
        id="drone-vs-drone",
        name="Drone vs Drone",
        summary="Two swarms contest the same airspace with elimination and area-denial pressure.",
        env_kwargs={"n_agents": 6, "grid": 24, "max_steps": 320},
        observation="Local neighbors plus contested-lane occupancy and alive counts.",
        action="Continuous 2D velocity command per drone.",
        reward="Favor lane control, survival, and separation from friendlies.",
    ),
    "moving-target-track": ScenarioDefinition(
        id="moving-target-track",
        name="Moving Target Track",
        summary="Shadow one or more ground movers without dropping visual custody.",
        env_kwargs={"n_agents": 4, "grid": 22, "max_steps": 300},
        observation="Target-relative bearings, occlusion bins, and wingman offsets.",
        action="Continuous 2D velocity command around moving target tracks.",
        reward="Reward continuous custody and multi-angle coverage; penalize lost track.",
    ),
    "search-and-interdict": ScenarioDefinition(
        id="search-and-interdict",
        name="Search & Interdict",
        summary="Search cluttered space under GPS denial, then converge on contact.",
        env_kwargs={"n_agents": 5, "grid": 24, "max_steps": 360},
        observation="Coverage patch, jammer pockets, obstacle slices, and last-seen target cue.",
        action="Continuous 2D velocity command with decentralized local observations only.",
        reward="New search coverage before contact, then rapid intercept once found.",
    ),
    "defend-asset": ScenarioDefinition(
        id="defend-asset",
        name="Defend Asset",
        summary="Hold a fixed perimeter around a protected point against inbound agents.",
        env_kwargs={"n_agents": 5, "grid": 20, "max_steps": 280},
        observation="Asset-relative bearings, defended sectors, and inbound velocity cues.",
        action="Continuous 2D velocity command around a fixed defended asset.",
        reward="Reward keeping hostiles outside the ring and intercepting early.",
    ),
    "swarm-vs-swarm-race": ScenarioDefinition(
        id="swarm-vs-swarm-race",
        name="Swarm vs Swarm Coverage Race",
        summary="Competitive coverage under jamming where first-touch scoring matters.",
        env_kwargs={"n_agents": 6, "grid": 26, "max_steps": 320},
        observation="Coverage patch, contested cells, rival offsets, and jammer corridors.",
        action="Continuous 2D velocity command using the same point-mass dynamics.",
        reward="Reward first-touch coverage and zone control; penalize collisions.",
    ),
}


def list_scenarios() -> list[ScenarioDefinition]:
    """Return the hard-coded scenario registry in insertion order."""
    return list(SCENARIOS.values())


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    """Look up a scenario by id with a useful error message."""
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"unknown scenario '{scenario_id}'. known scenarios: {known}") from exc


def make_scenario_env(
    scenario_id: str,
    battlefield: BattlefieldConfig | None = None,
    **overrides: int | float,
) -> SwarmEnv:
    """Instantiate the shared point-mass env using the scenario's preset knobs.

    Args:
        scenario_id:  Key into the SCENARIOS registry.
        battlefield:  Optional BattlefieldConfig.  If None, the scenario's
                      combat-stress defaults from env_config.py are used.
                      Pass `BattlefieldConfig()` (garrison defaults) to run
                      without any P0 parameter effects.
        **overrides:  Any SwarmEnv keyword args that further override the
                      scenario's env_kwargs (n_agents, grid, max_steps …).
                      Note: when a BattlefieldConfig is passed, n_agents and
                      max_steps are derived from the config; **overrides still
                      take precedence via the SwarmEnv constructor.
    """
    scenario = get_scenario(scenario_id)
    if battlefield is None:
        battlefield = get_scenario_defaults(scenario_id)
    env_kwargs = {**scenario.env_kwargs, **overrides}
    return SwarmEnv(**env_kwargs, battlefield=battlefield, scenario_id=scenario_id)
