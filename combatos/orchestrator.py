"""CombatOS Orchestrator — single entry point.

Starts the WebSocket bus and all module tasks, then runs until SIGINT/SIGTERM.

Usage:
    cd <repo-root>
    uv run --project combatos python -m combatos
      or
    python -m combatos.orchestrator

Environment overrides (see config.py):
    COMBATOS_HOST        bind address          (default 0.0.0.0)
    COMBATOS_PORT        bus port              (default 8000)
    SWARM_BUS_URL        swarm sub-bus         (default ws://localhost:8765)
    RECON_ASSET_PATH     splat file path       (default recon/assets/field.splat)
    COMBATOS_MOCK_POSE   emit mock pose        (default 1)
"""
from __future__ import annotations
import asyncio
import logging
import signal

from .bus import ws_server
from .modules.nav_module import NavModule
from .modules.perception_module import PerceptionModule
from .modules.recon_module import ReconModule
from .modules.swarm_module import SwarmModule
from .state.system_state import run_status_loop
from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║           CombatOS Orchestrator                          ║
║   GPS: DENIED  ·  LINK: NONE  ·  LOCALIZING...          ║
║                                                          ║
║   Bus    →  ws://0.0.0.0:{port:<5}                         ║
║   Swarm  →  {swarm_url:<44}  ║
╚══════════════════════════════════════════════════════════╝
"""


async def _run_module_forever(module) -> None:
    """Run module.run() in a restart loop — crashes are logged, not fatal."""
    while True:
        try:
            await module.run()
        except asyncio.CancelledError:
            log.info("[%s] stopped", module.name)
            break
        except Exception as exc:
            log.error("[%s] crashed: %s — restarting in 3 s", module.name, exc)
            await asyncio.sleep(3.0)


async def main() -> None:
    print(
        _BANNER.format(
            port=config.BUS_PORT,
            swarm_url=config.SWARM_BUS_URL,
        )
    )

    modules = [NavModule(), PerceptionModule(), ReconModule(), SwarmModule()]

    tasks = [
        asyncio.create_task(ws_server.serve(),        name="bus"),
        asyncio.create_task(run_status_loop(),         name="status"),
        *[
            asyncio.create_task(_run_module_forever(m), name=m.name)
            for m in modules
        ],
    ]

    def _shutdown() -> None:
        log.info("shutdown signal received — stopping all tasks")
        for t in tasks:
            t.cancel()

    # add_signal_handler is POSIX-only; use a KeyboardInterrupt wrapper on Windows
    import sys
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

    log.info("all systems go — connect dashboard to ws://localhost:%d", config.BUS_PORT)

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _shutdown()
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for name, result in zip([t.get_name() for t in tasks], results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            log.error("task [%s] exited with error: %s", name, result)

    log.info("=== CombatOS Orchestrator stopped ===")


if __name__ == "__main__":
    asyncio.run(main())
