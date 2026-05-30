# HOOK_SWARM.md — How swarm (⓸) connects to the CombatOS orchestrator

**Owner:** Nikhil · **Module:** `swarm/` · **Runs on:** Mac CPU (training) + browser (inference)

Good news: **the swarm is already wired up** in Phase 0–3 with zero changes to
`swarm/bus.py`.  The orchestrator connects TO the swarm's existing server as a
relay client.  This doc explains what's happening and what to change for the
Phase 3 upgrade.

---

## 1. Current state (Phase 0–2) — how it works today

```
swarm/bus.py  →  ws://localhost:8765  ←── SwarmModule (relay client, inside orchestrator)
                                               │  relays every frame
                                               ▼
                                    orchestrator bus :8000
                                               │
                                               ▼
                                      React dashboard
```

The orchestrator's `swarm_module.py` connects to `ws://localhost:8765`, reads
every frame your bus publishes, and re-publishes it on the main bus at port 8000.

**You don't need to change anything in `swarm/bus.py` for the demo to work.**

To start your bus so the orchestrator can find it:
```bash
cd swarm
uv run python -m swarm.bus --policy trained   # or --policy random for Phase 0
```

Then in a second terminal:
```bash
cd <repo-root>
uv run --project combatos python -m combatos
```

The dashboard at port 8000 will show the swarm panel.

---

## 2. What the orchestrator expects from your bus

Your `bus.py` already emits the correct schema (confirmed in `swarm/bus.py:swarm_message`):

```json
{
  "topic": "swarm",
  "t": 1234.567,
  "comms": "denied",
  "agents": [
    { "id": 0, "x": 1.2, "y": -0.4, "z": 2.1, "yaw": 0.3, "role": "scout", "alive": true }
  ]
}
```

The orchestrator relay strips the `topic` field, passes the rest to `router.publish("swarm", ...)`,
and it lands on the dashboard exactly as-is.

---

## 3. Phase 3 upgrade — publish directly to the orchestrator bus (optional)

In Phase 3 you can simplify by removing the two-server setup: instead of running
`swarm/bus.py` as a server, publish directly to the orchestrator.

**Change `swarm/bus.py` `serve()` as follows:**

```python
# Replace the websockets.serve(...) block with a client connection:
import websockets as ws_client

async def serve_via_orchestrator(policy_fn_factory, label, orch_url="ws://localhost:8000", hz=10.0):
    env = SwarmEnv(seed=0)
    obs = env.reset()
    policy = policy_fn_factory(env)
    dt = 1.0 / hz

    while True:
        try:
            async with ws_client.connect(orch_url) as ws:
                print(f"[bus] connected to orchestrator at {orch_url}")
                killed_demo = False
                while True:
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)

                    if not killed_demo and env.steps == 200:
                        env.kill(env.n - 1)
                        killed_demo = True

                    msg = json.dumps({"topic": "swarm", **swarm_message(env)})
                    await ws.send(msg)

                    if dones.all():
                        obs = env.reset()
                        killed_demo = False
                    await asyncio.sleep(dt)
        except Exception as e:
            print(f"[bus] reconnecting: {e}")
            await asyncio.sleep(2.0)
```

Add `--mode orchestrator` flag to `main()` to select between old server mode
and new client mode without breaking Phase 0 dev.

When using this mode, update `SWARM_BUS_URL` in the orchestrator config so it
knows the swarm is not running its own server:
```bash
SWARM_BUS_URL="" python -m combatos   # disables relay; swarm connects directly
```

And set `SWARM_BUS_URL` to empty string in `config.py` or as an env var.
The `SwarmModule` gracefully retries so an empty URL just means it never connects
(no crash), and the direct-publishing swarm reaches the bus via the main server.

---

## 4. Health tracking

The orchestrator marks swarm **"up"** when `SwarmModule` is connected and receiving
frames, **"degraded"** after 6 s of silence, **"down"** when `swarm/bus.py` is not
running.

The hero banner shows `swarm: up | degraded | down` in the modules map.

---

## 5. Browser inference (Phase 3 / SWARM.md §Phase 3)

When the policy runs in the browser (`frontend/src/swarm/sim.ts` + `policy.ts`),
it publishes the `swarm` message via the same dashboard WebSocket connection.
No changes to the orchestrator needed — the browser publishes `{"topic": "swarm", ...}`
directly to `ws://localhost:8000` and the orchestrator relays it to any other
subscribers (logs, secondary displays, etc.).

---

## 6. Controlling the swarm from the dashboard

To add a "kill agent" button in the dashboard (Phase 4 money demo):

The dashboard sends:
```json
{ "topic": "swarm_cmd", "cmd": "kill", "id": 3 }
```

In `swarm/bus.py` (or `sim.ts` for browser mode), subscribe and handle:
```python
# After connecting to orchestrator as client:
await ws.send(json.dumps({"type": "subscribe", "topics": ["swarm_cmd"]}))

async for raw in ws:
    msg = json.loads(raw)
    if msg.get("topic") == "swarm_cmd" and msg.get("cmd") == "kill":
        env.kill(msg["id"])
```

For browser mode, the TypeScript sim loop already handles this natively since
it's in the same process as the dashboard.

---

## 7. Quick smoke test

```bash
# Verify relay is working: start both and watch the orchestrator logs
# Terminal 1:
cd swarm && uv run python -m swarm.bus --policy random

# Terminal 2:
cd <repo-root> && uv run --project combatos python -m combatos

# You should see:
#   [swarm] connecting to ws://localhost:8765
#   [swarm] relay active
#   module [swarm] → up
# And the dashboard swarm panel should animate.
```
