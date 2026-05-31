"""env_config.py — Battlefield parameter schema for the CombatOS swarm sim.

This is the Python mirror of frontend/src/gym/battlefieldParams.ts.
Field names, ranges, and defaults are kept in sync across both files.

Priority tiers (see docs/battlefield-parameters.md §2):
  P0  — wired into SwarmEnv dynamics / obs / rewards
  P1  — scenario narrative; no sim change at hackathon scope
  P2  — deferred post-demo
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


# ─────────────────────────────────────────────────── sub-config dataclasses ──

@dataclass
class WeatherConfig:
    """Weather knobs.  P0: wind_speed, wind_dir_rad.  P1: visibility, temperature_c."""
    wind_speed: float = 0.0       # [P0] 0–15 world-units/s; added as drift each step
    wind_dir_rad: float = 0.0     # [P0] 0–2π; 0 = +x axis
    visibility: float = 1.0       # [P1] 0–1 fraction — display only
    temperature_c: float = 20.0   # [P1] −20–50 °C — display only


@dataclass
class EWConfig:
    """Electronic warfare knobs.  P0: gps_denial_level, jam_duty_cycle.  P2: spoofing."""
    gps_denial_level: float = 0.0  # [P0] 0–1; adds Gaussian noise σ = level×0.2 to obs[0:2]
    jam_duty_cycle: float = 0.0    # [P0] 0–1; each neighbor slot zeroed with this probability
    spoofing_enabled: bool = False  # [P2] deferred


@dataclass
class TerrainConfig:
    """Terrain knobs — P1 display only at hackathon scope."""
    elev_roughness: float = 0.0   # [P1] 0–1
    urban_density: float = 0.0    # [P1] 0–1


@dataclass
class ThreatConfig:
    """Threat knobs — P1 narrative only."""
    hostile_uas_count: int = 0          # [P1] 0–10; attrition_inject_rate drives actual kills
    moving_target_speed: float = 0.3   # [P1] 0–1 normalized


@dataclass
class ROEConfig:
    """Rules of engagement knobs.  time_limit_sec maps to max_steps (P0 effect)."""
    engagement_authority: str = "hold-fire"  # [P1] display only: hold-fire / weapons-tight / weapons-free
    min_standoff_m: float = 0.0              # [P1] 0–20 m
    civilian_density: float = 0.0            # [P2] deferred
    time_limit_sec: float = 400.0            # [P0→max_steps] 30–600 s; caps episode length


@dataclass
class LogisticsConfig:
    """Logistics knobs.  All three are P0."""
    swarm_size: int = 5                   # [P0] 2–12; must match trained checkpoint
    battery_envelope_sec: float = 400.0   # [P0] 30–600 s; caps max_steps
    attrition_inject_rate: float = 0.0    # [P0] 0–0.5 per-step kill probability


@dataclass
class BattlefieldConfig:
    """Top-level battlefield configuration.  Consumed by SwarmEnv."""
    env_id: str = ""
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    ew: EWConfig = field(default_factory=EWConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    roe: ROEConfig = field(default_factory=ROEConfig)
    logistics: LogisticsConfig = field(default_factory=LogisticsConfig)

    @property
    def max_steps(self) -> int:
        """Derive max_steps from battery_envelope_sec (floored to int)."""
        return max(30, int(min(self.logistics.battery_envelope_sec, self.roe.time_limit_sec)))

    @property
    def n_agents(self) -> int:
        return self.logistics.swarm_size


# ───────────────────────────────────────────────────────────────── validation ──

class ConfigValidationError(ValueError):
    """Raised when a BattlefieldConfig field is out of its allowed range."""


def validate_config(cfg: BattlefieldConfig) -> None:
    """Validate all P0/P1 numeric fields.  Raises ConfigValidationError on first violation."""
    def check(domain: str, field_name: str, value: float, lo: float, hi: float) -> None:
        if not (lo <= value <= hi):
            raise ConfigValidationError(
                f"{domain}.{field_name} = {value!r} is out of range [{lo}, {hi}]"
            )

    # weather
    check("weather", "wind_speed",    cfg.weather.wind_speed,    0,   15)
    check("weather", "wind_dir_rad",  cfg.weather.wind_dir_rad,  0,   2 * math.pi)
    check("weather", "visibility",    cfg.weather.visibility,    0,   1)
    check("weather", "temperature_c", cfg.weather.temperature_c, -20, 50)

    # ew
    check("ew", "gps_denial_level", cfg.ew.gps_denial_level, 0, 1)
    check("ew", "jam_duty_cycle",   cfg.ew.jam_duty_cycle,   0, 1)

    # terrain
    check("terrain", "elev_roughness", cfg.terrain.elev_roughness, 0, 1)
    check("terrain", "urban_density",  cfg.terrain.urban_density,  0, 1)

    # threat
    check("threat", "hostile_uas_count",   cfg.threat.hostile_uas_count,   0,  10)
    check("threat", "moving_target_speed", cfg.threat.moving_target_speed, 0,  1)

    # roe
    check("roe", "min_standoff_m",    cfg.roe.min_standoff_m,    0,  20)
    check("roe", "civilian_density",  cfg.roe.civilian_density,  0,  1)
    check("roe", "time_limit_sec",    cfg.roe.time_limit_sec,    30, 600)

    # logistics
    if not (2 <= cfg.logistics.swarm_size <= 12):
        raise ConfigValidationError(
            f"logistics.swarm_size = {cfg.logistics.swarm_size} is out of range [2, 12]"
        )
    check("logistics", "battery_envelope_sec",  cfg.logistics.battery_envelope_sec,  30,  600)
    check("logistics", "attrition_inject_rate", cfg.logistics.attrition_inject_rate, 0,   0.5)


# ────────────────────────────────────────────────────────── scenario defaults ──

def _pi(x: float) -> float:
    return x * math.pi


SCENARIO_DEFAULTS: dict[str, BattlefieldConfig] = {
    "drone-vs-drone": BattlefieldConfig(
        env_id="drone-vs-drone",
        weather=WeatherConfig(wind_speed=3.0, wind_dir_rad=_pi(1/6)),
        ew=EWConfig(gps_denial_level=0.0, jam_duty_cycle=0.2),
        threat=ThreatConfig(hostile_uas_count=3),
        roe=ROEConfig(engagement_authority="weapons-tight", time_limit_sec=320),
        logistics=LogisticsConfig(swarm_size=6, battery_envelope_sec=320, attrition_inject_rate=0.0),
    ),
    "moving-target-track": BattlefieldConfig(
        env_id="moving-target-track",
        weather=WeatherConfig(wind_speed=2.0, wind_dir_rad=_pi(1/3)),
        ew=EWConfig(gps_denial_level=0.0, jam_duty_cycle=0.0),
        threat=ThreatConfig(moving_target_speed=0.8),
        roe=ROEConfig(engagement_authority="weapons-tight", time_limit_sec=300),
        logistics=LogisticsConfig(swarm_size=4, battery_envelope_sec=300, attrition_inject_rate=0.02),
    ),
    "search-and-interdict": BattlefieldConfig(
        env_id="search-and-interdict",
        weather=WeatherConfig(wind_speed=4.0, wind_dir_rad=_pi(1/4), visibility=0.6, temperature_c=15),
        ew=EWConfig(gps_denial_level=0.7, jam_duty_cycle=0.4),
        threat=ThreatConfig(hostile_uas_count=1, moving_target_speed=0.7),
        roe=ROEConfig(engagement_authority="weapons-tight", time_limit_sec=360),
        logistics=LogisticsConfig(swarm_size=5, battery_envelope_sec=360, attrition_inject_rate=0.02),
    ),
    "defend-asset": BattlefieldConfig(
        env_id="defend-asset",
        weather=WeatherConfig(wind_speed=2.0, wind_dir_rad=_pi(1/2)),
        ew=EWConfig(gps_denial_level=0.0, jam_duty_cycle=0.1),
        threat=ThreatConfig(hostile_uas_count=4, moving_target_speed=0.5),
        roe=ROEConfig(engagement_authority="weapons-tight", min_standoff_m=5, time_limit_sec=280),
        logistics=LogisticsConfig(swarm_size=5, battery_envelope_sec=280, attrition_inject_rate=0.05),
    ),
    "swarm-vs-swarm-race": BattlefieldConfig(
        env_id="swarm-vs-swarm-race",
        weather=WeatherConfig(wind_speed=5.0, wind_dir_rad=_pi(1/4), visibility=0.8, temperature_c=10),
        ew=EWConfig(gps_denial_level=0.5, jam_duty_cycle=0.4),
        threat=ThreatConfig(hostile_uas_count=6, moving_target_speed=0.6),
        roe=ROEConfig(engagement_authority="weapons-free", time_limit_sec=320),
        logistics=LogisticsConfig(swarm_size=6, battery_envelope_sec=320, attrition_inject_rate=0.04),
    ),
    "navigate-to-target": BattlefieldConfig(
        env_id="navigate-to-target",
        weather=WeatherConfig(wind_speed=1.0, wind_dir_rad=_pi(0)),
        ew=EWConfig(gps_denial_level=0.0, jam_duty_cycle=0.0),
        threat=ThreatConfig(hostile_uas_count=0),
        roe=ROEConfig(engagement_authority="weapons-tight", time_limit_sec=300),
        logistics=LogisticsConfig(swarm_size=1, battery_envelope_sec=300, attrition_inject_rate=0.0),
    ),
}


def get_scenario_defaults(scenario_id: str) -> BattlefieldConfig:
    """Return combat-stress defaults for a scenario, falling back to garrison defaults."""
    return SCENARIO_DEFAULTS.get(scenario_id, BattlefieldConfig(env_id=scenario_id))


def make_profile_config(scenario_id: str, profile: str) -> BattlefieldConfig:
    """Return a BattlefieldConfig for the requested training/inference profile.

    `combat` uses the scenario's stressed defaults.
    `garrison` preserves scenario sizing/time limits but zeros the P0 stressors so
    the same scenario can run in an uncontested baseline configuration.
    """
    if profile not in {"combat", "garrison"}:
        raise ValueError(f"unknown profile '{profile}'. expected combat or garrison")

    combat = get_scenario_defaults(scenario_id)
    if profile == "combat":
        return combat

    cfg = BattlefieldConfig(
        env_id=scenario_id,
        weather=WeatherConfig(
            wind_speed=0.0,
            wind_dir_rad=combat.weather.wind_dir_rad,
            visibility=combat.weather.visibility,
            temperature_c=combat.weather.temperature_c,
        ),
        ew=EWConfig(
            gps_denial_level=0.0,
            jam_duty_cycle=0.0,
            spoofing_enabled=combat.ew.spoofing_enabled,
        ),
        terrain=combat.terrain,
        threat=combat.threat,
        roe=ROEConfig(
            engagement_authority=combat.roe.engagement_authority,
            min_standoff_m=combat.roe.min_standoff_m,
            civilian_density=combat.roe.civilian_density,
            time_limit_sec=combat.roe.time_limit_sec,
        ),
        logistics=LogisticsConfig(
            swarm_size=combat.logistics.swarm_size,
            battery_envelope_sec=combat.logistics.battery_envelope_sec,
            attrition_inject_rate=0.0,
        ),
    )
    validate_config(cfg)
    return cfg


def config_to_json_dict(cfg: BattlefieldConfig) -> dict:
    """Stable JSON-serializable dict for checkpoint metadata and docs."""
    return asdict(cfg)


if __name__ == "__main__":
    for sid, cfg in SCENARIO_DEFAULTS.items():
        validate_config(cfg)
        print(f"✓ {sid}: wind={cfg.weather.wind_speed} jam={cfg.ew.jam_duty_cycle}"
              f" gps={cfg.ew.gps_denial_level} attrition={cfg.logistics.attrition_inject_rate}"
              f" max_steps={cfg.max_steps}")
