# PyBullet Swarm Video Prototype

This directory is a standalone PyBullet prototype for drone-swarm surveillance and FPV routing through CombatOS.

## Rules

These are the working rules for this prototype:

1. Raw drone FPV may be sent to the CombatOS orchestrator.
2. Perception may return annotated video for operator display, but autonomy should consume structured detections or tracks, not decoded video frames.
3. The correct round-trip is:
   raw FPV up, detections and optional annotated FPV down.
4. The returned annotated video is a HUD and operator-awareness product, not the primary flight-control signal.
5. Every drone should have its own first-person feed.
6. The system should produce both:
   one tiled composite video for quick review, and one MP4 per drone for detailed inspection.
7. The simulation must support direct observation of swarm behavior in PyBullet GUI mode.
8. The current perception round-trip in this directory is a prototype:
   it uses simulation ground truth to project troop locations into image space.
9. The intended replacement is a real detector/tracker path that infers from pixels and returns metadata plus optional annotated video.
10. The simulation/controller boundary should stay clean so a future policy adapter can replace the scripted controller without rewriting rendering, transport, or recording.

## What This Prototype Does

- a swarm of drones flies over a patrol area
- a small group of ground troops moves in scattered pockets below
- the ground scene is dressed with berms, blast marks, concrete ruins, wrecks, perimeter walls, and sandbag emplacements so it reads more like a battlefield than a test pad
- the sim will automatically use staged assets from `pybullet_swarm_video/resources/` when it can
- the default FPV capture resolution is now `640x360` per drone
- each drone renders its own first-person camera feed
- the feeds are tiled into one output video
- each drone also gets its own MP4 output by default
- a second demo mode sends each FPV frame through the CombatOS orchestrator, lets
  a perception worker annotate targets, and records the returned HUD frames

The current implementation is intentionally simple:

- drone motion is scripted, not learned
- troop count is intentionally lower by default so the scene focuses on drone behavior and target visibility instead of crowd density
- the simulation is headless by default (`DIRECT`), but `--gui` enables direct observation
- GUI mode actively tracks the swarm/troop scene so you can watch behavior live
- in GUI mode you can switch from the observer camera into a selected drone's chase or FPV view
- in GUI mode you can also take manual control of one drone while the rest continue flying policy-driven behavior
- GUI runs are automatically stretched to a much longer duration so the window does not close almost immediately while you are flying or inspecting
- the code is structured so a future policy adapter can replace the scripted controller
- the current perception round-trip uses simulation ground truth for target projection
- FPV rendering now uses mild camera shake, bank, vignette, grain, and motion-blur style post-processing to read more like surveillance footage than a raw test render

## Resource Loading

The prototype now looks for these files in `pybullet_swarm_video/resources/`:

- `Best_Soldier.zip`
- `damaged_concrete_floor_4k.blend.zip`
- `free_military_soldier_rigged.zip`
- `low-poly-soldiers-rigged-free.zip`
- `low_poly_container.zip`
- `low_poly_military_truck.zip`
- `low_poly_tank.zip`
- `low_poly_truck_tank.zip`
- `militarytent.zip`
- `salt_dome_11_iran.zip`
- `single_sandbag.zip`
- `drone_design.zip`

Current behavior:

- the best-soldier archive is preferred as the primary troop mesh;
  the older military-soldier archive is only used as a fallback if the new one is absent or fails to load
- the concrete-floor archive is used immediately:
  the diffuse texture is extracted into `pybullet_swarm_video/resources/.cache/` and applied to the battlefield ground when the salt-dome terrain is not available
- the military-soldier archive is extracted into `pybullet_swarm_video/resources/.cache/free_military_soldier_rigged/`
  and used only as the fallback 3D troop mesh
- the low-poly soldier archive is used immediately:
  the packaged `texture.png` is extracted into `pybullet_swarm_video/resources/.cache/` and applied only when both higher-fidelity soldier meshes fail
- the sandbag archive is extracted into `pybullet_swarm_video/resources/.cache/single_sandbag/`
  and used for the emplacements
- the drone archive is extracted into `pybullet_swarm_video/resources/.cache/drone_design/`
  and used as the drone mesh, with the drone texture applied from the extracted files
- when the salt-dome archive is present, the sim swaps in the textured salt-dome terrain and stages the new trucks, tents, containers, and armor props around it at scene scale
- if one of the extracted meshes fails to load, the sim falls back to the primitive shape for that actor type instead of failing

This keeps the sim runnable while using the new OBJ/MTL asset bundles directly.

## First Run On Another Machine

No, a new user does not need to manually unzip the asset files.

The simulation reads the zip files directly from `pybullet_swarm_video/resources/` and automatically extracts what it needs into `pybullet_swarm_video/resources/.cache/` on first run.

What another machine needs:

- Python `3.12`
- `uv`
- the repo checked out with the zip files still present in `pybullet_swarm_video/resources/`

Recommended first-run sequence from the repo root:

```bash
cd pybullet_swarm_video
uv sync
uv run python -m pybullet_swarm_video.run_demo --gui
```

Equivalent command from the repo root without changing directories:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo --gui
```

What happens on that first run:

1. `uv sync` creates the environment and installs the project dependencies declared in `pyproject.toml`.
2. the sim opens the zip files in `pybullet_swarm_video/resources/`
3. extracted asset files are written into `pybullet_swarm_video/resources/.cache/`
4. the sim loads the ground texture, soldier mesh parts, low-poly soldier fallback texture, sandbag mesh, and drone mesh from that cache
5. output videos are written under `output/`

If someone updates or replaces one of the zip files later:

- the sim should re-extract automatically because it stamps extracted folders against the archive modification time
- if the cache ever looks stale or corrupted, delete `pybullet_swarm_video/resources/.cache/` and run again

Minimum quick-check command:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo \
  --seconds 5 \
  --fps 8 \
  --output output/smoke_test.mp4
```

If another machine is slower and the new default render resolution is too heavy in GUI mode, lower it explicitly:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo \
  --gui \
  --width 320 \
  --height 180
```

If they also want the orchestrated round-trip on another machine:

```bash
uv run --project combatos python -m combatos
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_orchestrated_demo --gui
```

## Observation Modes

- Headless recording mode:
  generates videos without opening the PyBullet viewer.
- Direct observation mode:
  pass `--gui` to open the PyBullet spectator view and watch the swarm behavior while recording.
- FPV review mode:
  inspect the tiled composite MP4 or the individual per-drone MP4s after the run.

## GUI Controls

When the simulation is launched with `--gui`:

- `C`: cycle camera mode between observer, chase, and FPV
- `B`: jump to observer view
- `H`: jump to chase view
- `F`: jump to FPV view
- `1`..`9`: select the active drone
- `M`: toggle manual control for the selected drone
- `R`: return the selected drone to scripted mode
- `Up` / `Down` or `I` / `K`: move forward / backward
- `Left` / `Right` or `J` / `L`: strafe left / right
- `U` / `O`: move up / down
- `Z` / `X`: yaw left / right

Only one drone is manually controlled at a time. The others continue following the scripted surveillance behavior.
The manual path now reads one keyboard snapshot per simulation step, which makes multi-key combinations more stable than the previous per-function polling approach.

## Outputs

For both demo entrypoints, the default outputs are:

- one tiled composite video at the path given by `--output`
- one per-drone directory derived from the output name, for example:
  `output/drone_spy_demo_feeds/drone_00.mp4`
  `output/drone_spy_demo_feeds/drone_01.mp4`
  `output/drone_spy_demo_feeds/drone_02.mp4`

You can override the individual-feed directory with `--per-drone-dir`.

## Run: Local Sim

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo \
  --seconds 12 \
  --fps 12 \
  --output output/drone_spy_demo.mp4
```

Watch the simulation live while recording:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo \
  --gui \
  --seconds 12 \
  --fps 12 \
  --output output/drone_spy_demo.mp4
```

In GUI mode the run duration is automatically extended to at least 600 seconds unless you already requested a longer run.

Write the individual drone feeds into an explicit directory:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_demo \
  --gui \
  --seconds 12 \
  --fps 12 \
  --output output/drone_spy_demo.mp4 \
  --per-drone-dir output/drone_spy_demo_feeds
```

## Orchestrated FPV round-trip

Start the CombatOS orchestrator first:

```bash
uv run --project combatos python -m combatos
```

Then run the PyBullet round-trip demo:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_orchestrated_demo \
  --seconds 10 \
  --fps 8 \
  --output output/drone_spy_roundtrip.mp4
```

Run the orchestrated mode with direct observation enabled:

```bash
uv run --project pybullet_swarm_video python -m pybullet_swarm_video.run_orchestrated_demo \
  --gui \
  --seconds 10 \
  --fps 8 \
  --output output/drone_spy_roundtrip.mp4
```

In GUI mode the orchestrated run is also automatically extended to at least 600 seconds.

What happens in that mode:

- each drone publishes a raw FPV frame on the orchestrator image bus
- the sim publishes camera pose + troop world state on the orchestrator control bus
- a perception worker subscribes through the orchestrator, projects troop locations into image space,
  draws the existing perception HUD style, and publishes the annotated frame back
- the drone demo records the returned HUD frames, not the local raw render
- one annotated MP4 is written per drone in addition to the tiled composite

## Current structure

- `pybullet_swarm_video/config.py`: dataclasses for sim, camera, and recording config
- `pybullet_swarm_video/policies.py`: scripted drone surveillance controller
- `pybullet_swarm_video/simulation.py`: PyBullet world, actors, stepping, and camera capture
- `pybullet_swarm_video/bus_client.py`: orchestrator control/image bus client
- `pybullet_swarm_video/perception_node.py`: prototype perception worker for the FPV round-trip
- `pybullet_swarm_video/record.py`: tiling and video writing
- `pybullet_swarm_video/run_demo.py`: CLI entrypoint
- `pybullet_swarm_video/run_orchestrated_demo.py`: end-to-end FPV -> orchestrator -> perception -> drone demo

## Design Notes

- The local drone controller is still scripted.
- The FPV transport and recording paths are now separated from the controller, which is the right setup for replacing the controller later.
- The orchestrated perception node in this directory is a prototype bridge, not a real detector.
- The frontend currently mirrors the dashboard FPV topics for one drone; per-drone topic selection is still a later step.
- Manual control in GUI mode does not break the bus contract. The drone still publishes FPV and state through the same topics.
- More complex drone behavior is compatible with the protocol as long as the controller consumes local observations or structured detections/tracks, not returned video frames as its primary control input.

## Next likely steps

1. Replace the scripted controller with a policy adapter that consumes a local observation vector shaped like the existing `swarm/` environment.
2. Replace the ground-truth perception bridge with the real detector path so the projection metadata is no longer needed.
3. Add per-drone topic selection in the frontend if you want the dashboard to inspect individual drones instead of mirroring drone `0`.
