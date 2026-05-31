"""CombatOS orchestrator entrypoint.

Starts the control WebSocket bus, the image WebSocket bus, and all module
tasks, then runs until SIGINT/SIGTERM.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from . import config
from .bus import image_ws_server, ws_server
from .modules.nav_module import NavModule
from .modules.perception_module import PerceptionModule
from .modules.recon_module import ReconModule
from .modules.ros_slam_module import RosSlamModule
from .modules.swarm_module import SwarmModule
from .state.system_state import run_status_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_BANNER = """
+--------------------------------------------------------------+
|                    CombatOS Orchestrator                     |
|  GPS: DENIED  |  LINK: NONE  |  LOCALIZING...                |
|                                                              |
|  Control -> ws://0.0.0.0:{port:<5}                               |
|  Images  -> ws://0.0.0.0:{image_port:<5}                               |
|  Swarm   -> {swarm_url:<44} |
+--------------------------------------------------------------+
"""


async def _run_module_forever(module) -> None:
    """Run module.run() in a restart loop; crashes are logged, not fatal."""
    while True:
        try:
            await module.run()
        except asyncio.CancelledError:
            log.info("[%s] stopped", module.name)
            break
        except Exception as exc:
            log.error("[%s] crashed: %s; restarting in 3 s", module.name, exc)
            await asyncio.sleep(3.0)


async def main() -> None:
    print(
        _BANNER.format(
            port=config.BUS_PORT,
            image_port=config.IMAGE_BUS_PORT,
            swarm_url=config.SWARM_BUS_URL,
        )
    )

    modules = [NavModule(), PerceptionModule(), ReconModule(), RosSlamModule()]
    if config.ENABLE_SWARM:
        modules.append(SwarmModule())
    else:
        log.info("[swarm] disabled; set COMBATOS_SWARM=1 to enable swarm relay")

    tasks = [
        asyncio.create_task(ws_server.serve(), name="bus"),
        asyncio.create_task(image_ws_server.serve(), name="image_bus"),
        asyncio.create_task(run_status_loop(), name="status"),
        *[
            asyncio.create_task(_run_module_forever(m), name=m.name)
            for m in modules
        ],
    ]

    def _shutdown() -> None:
        log.info("shutdown signal received; stopping all tasks")
        for task in tasks:
            task.cancel()

    import sys

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

    log.info(
        "all systems go; control ws://localhost:%d image ws://localhost:%d",
        config.BUS_PORT,
        config.IMAGE_BUS_PORT,
    )

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except (KeyboardInterrupt, asyncio.CancelledError):
        _shutdown()
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for name, result in zip([task.get_name() for task in tasks], results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            log.error("task [%s] exited with error: %s", name, result)

    log.info("=== CombatOS Orchestrator stopped ===")


if __name__ == "__main__":
    asyncio.run(main())
