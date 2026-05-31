# Battlefield Parameters — General's Decision Log

**Classification:** HACKATHON WORKING DOCUMENT  
**Author:** Five-Star General Planning Cell × CombatOS LM Chief of Staff  
**Date:** 2026-05-30  
**Status:** P0 set LOCKED for demo build  

> **FRAGO Annex C — Swarm Sim Parameter Governance**  
> This document is the authoritative record of which environmental knobs are wired into
> the CombatOS point-mass swarm sim, which are operator-facing UI-only, and which are
> deferred post-demo. It binds `swarm/env_config.py`, `frontend/src/gym/battlefieldParams.ts`,
> and `swarm/env.py` to a single truth.

---

## 1. Parameter Catalog

Parameters are grouped by domain. Each row records:
- **Range** — valid input interval
- **Garrison default** — peacetime / uncontested starting value
- **Combat default** — stressed value used in training / demo
- **Policy behavior change** — what the MAPPO actor does differently when this knob moves

### 1.1 Weather

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `wind_speed` | 0–15 m/s | 0.0 | 6.0 | Agents must compensate with up-wind bias; clumping increases as downwind agents drift together |
| `wind_dir_rad` | 0–2π | 0.0 | π/4 | Spreads the coverage gap asymmetrically; policy learns to anchor upwind scouts |
| `visibility` | 0–1 (fraction) | 1.0 | 0.4 | Reduces effective coverage-patch radius; agents must fly tighter search lines (**P1 — UI feedback only, no obs change at P0**) |
| `temperature_c` | −20–50 °C | 20.0 | −5.0 | Low temperature deretes battery envelope (reduces `max_steps`); policy learns to close coverage faster (**P1**) |

### 1.2 EM / Electronic Warfare (EW)

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `gps_denial_level` | 0–1 | 0.0 | 0.7 | Gaussian position noise injected into own-position obs; policy spreads wider to hedge uncertain self-location |
| `jam_duty_cycle` | 0–1 | 0.0 | 0.4 | Each step, each neighbor slot dropped with probability `jam_duty_cycle`; policy learns to maintain coverage without neighbor info |
| `spoofing_enabled` | bool | false | false | Injects false neighbor positions (**P2 — post-demo**) |

### 1.3 Terrain / AO

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `elev_roughness` | 0–1 | 0.0 | 0.0 | Increases `BOUNDS_PENALTY` near obstacle cells; agents hug open lanes (**P1**) |
| `urban_density` | 0–1 | 0.0 | 0.0 | Adds no-fly cells to the coverage grid; reduces effective search area (**P1**) |

### 1.4 Threat

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `hostile_uas_count` | 0–10 | 0 | 3 | Inbound agents that trigger attrition when within kill radius; policy spreads to reduce kill-zone exposure (**P1 — count drives scenario narrative; actual kill mechanics via attrition_rate at P0**) |
| `moving_target_speed` | 0–1 normalized | 0.3 | 0.8 | Affects target escape rate in `moving-target-track`; policy must tighten pursuit geometry (**P1**) |

### 1.5 ROE / Task

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `engagement_authority` | `hold-fire` / `weapons-tight` / `weapons-free` | `hold-fire` | `weapons-tight` | UI label only at P0; future: gates the "fire" action expansion (**P1**) |
| `min_standoff_m` | 0–20 m | 0.0 | 5.0 | Adds penalty when agents enter a protected civilian / ROE ring (**P1**) |
| `civilian_density` | 0–1 | 0.0 | 0.0 | Scales `min_standoff_m` effect (**P2**) |
| `time_limit_sec` | 30–600 s | 400 s | 200 s | Maps to `max_steps`; shorter horizon forces faster coverage strategies |

### 1.6 Logistics

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `swarm_size` | 2–12 | 5 | 5 | `n_agents`; directly changes obs/action tensor shape — must match trained checkpoint |
| `battery_envelope_sec` | 60–600 s | 400 s | 180 s | Caps `max_steps`; policy learns urgency when budget shrinks |
| `attrition_inject_rate` | 0–0.5 per-step probability | 0.0 | 0.05 | Random kills during rollout; policy learns gap-filling from sparse attrition signal |

### 1.7 Blue Force

| Parameter | Range | Garrison | Combat | Policy behavior change |
|---|---|---|---|---|
| `relay_drone_available` | bool | false | false | Comms relay extends effective neighbor range (**P2**) |

---

## 2. Priority Tiers

### P0 — Must wire into sim (changes dynamics and observations)

These parameters are wired into `SwarmEnv` physics, reward, and/or observation vector.
Changing them in the UI triggers a real behavioral change in rollouts.

| Parameter | Sim impact | Obs impact | Reward impact |
|---|---|---|---|
| `wind_speed` + `wind_dir_rad` | Adds wind drift vector to position integration every step | None (position obs already encodes the effect) | None directly; coverage geometry changes |
| `gps_denial_level` | None on physics | Own-position obs `[0:2]` receives Gaussian noise `σ = gps_denial_level * 0.2` | None |
| `jam_duty_cycle` | None on physics | Each of the K neighbor slots zeroed-out independently with probability `jam_duty_cycle` | None |
| `attrition_inject_rate` | Calls `kill(random_agent)` probabilistically each step | Kills remove agent from neighbor sets immediately | Dead agents score 0; team loses coverage contributors |
| `time_limit_sec` → `max_steps` | Episode length | None | Shorter horizon increases urgency |

### P1 — UI display only (no Python or TypeScript sim change at hackathon scope)

`visibility`, `temperature_c`, `elev_roughness`, `urban_density`, `hostile_uas_count`,
`moving_target_speed`, `engagement_authority`, `min_standoff_m`, `time_limit_sec` (as label)

These appear as operator-facing sliders/toggles in the frontend but do **not** alter the
point-mass dynamics or the ONNX policy input vector during the hackathon build.
They are annotated `// P1: display only` in `battlefieldParams.ts`.

### P2 — Post-demo

`spoofing_enabled`, `civilian_density`, `relay_drone_available`.  
Require non-trivial obs/reward redesign. Deferred.

---

## 3. Scenario Matrix — P0 Parameter Subsets per Gym Environment

Which P0 parameters are **active** (non-default) in each scenario's preset. Non-listed
parameters use garrison defaults.

| Parameter | `drone-vs-drone` | `moving-target-track` | `search-and-interdict` | `defend-asset` | `navigate-to-target` |
|---|:---:|:---:|:---:|:---:|:---:|
| `wind_speed` | 3.0 | 2.0 | 4.0 | 2.0 | 1.0 |
| `wind_dir_rad` | π/6 | π/3 | π/4 | π/2 | 0 |
| `gps_denial_level` | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 |
| `jam_duty_cycle` | 0.2 | 0.0 | 0.4 | 0.1 | 0.0 |
| `attrition_inject_rate` | 0.0 | 0.02 | 0.02 | 0.05 | 0.0 |
| `max_steps` | 320 | 300 | 360 | 280 | 300 |

---

## 4. CTDE Constraint Check (vs `swarm/SWARM.md`)

The SWARM.md architecture mandates **local-only observations at deploy time**.  
All P0 parameter effects are verified against this constraint:

| Parameter | CTDE safe? | Rationale |
|---|---|---|
| Wind drift | ✅ | Applied to physics; actor only sees own perturbed position — no global state leaks |
| GPS denial noise | ✅ | Only adds noise to obs `[0:2]` (own position) — pure local |
| Jamming (neighbor dropout) | ✅ | Zeros slots in per-agent obs; the policy already handles zero-filled slots (fewer than K live neighbors) |
| Attrition | ✅ | Modifies `alive[]` — already handled by existing kill/neighbor-zero logic |

**No global state is added to the actor observation. Centralized critic may optionally see raw
parameter values at train time (appended to `global_state()`) but that is optional and noted
in the implementation.**

---

## 5. Parameter → Obs / Reward Mapping

| Obs index range | Meaning | Affected by |
|---|---|---|
| `[0:2]` | own position (normalized) | `gps_denial_level` adds Gaussian noise |
| `[2:4]` | own velocity | unaffected by battlefield params |
| `[4:10]` | K neighbor relative positions | `jam_duty_cycle` randomly zeros these slots |
| `[10:35]` | local coverage patch | `wind_speed/dir` alters which cells agents reach; coverage map reflects real positions |
| `[35]` | role flag | unaffected |

Reward terms affected by params:

| Reward term | Parameter | Effect |
|---|---|---|
| Coverage reward | Wind pushes agents off target cells | Net coverage rate decreases at high wind |
| Crowd penalty | Jamming → agents fly blind → may clump | Penalty fires more often under jamming |
| Bounds penalty | Wind can push agents to edges | Penalty fires more often at high wind speed |
| Team score for alive agents | `attrition_inject_rate` | Higher attrition → fewer live agents contributing |

---

## 6. General's Sign-Off

**P0 set is LOCKED.** The four knobs — wind, GPS denial, jamming, and attrition — are the
minimum set that:

1. **Demonstrate comms-denied operation** (jamming drops neighbor info → swarm coordinates on
   local obs alone — this *is* the CTDE thesis made visible).
2. **Demonstrate resilience** (attrition kills agents mid-rollout — swarm re-covers without
   comms).
3. **Change trajectories measurably** (wind pushes positions; a human watching the Three.js
   panel can see the drift at wind_speed = 8).
4. **Preserve the existing obs vector shape** (OBS_DIM stays 36; no checkpoint invalidation
   from adding a new dim — params affect inputs, not the vector layout).

P1 parameters are wired as UI labels only. They provide operator narrative without risking
checkpoint/parity breaks during the 48-hour build window.

*Signed: General's Planning Cell, 2026-05-30*
