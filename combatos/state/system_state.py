"""Module health tracking and status topic aggregation.

Every time a module publishes a message, the WS server calls beat(module_name).
This file tracks the last-seen timestamp per module and derives the hero-banner
status that the dashboard renders.

The status topic is broadcast at STATUS_HZ (default 1 Hz) regardless of whether
any module state has changed — the dashboard always gets a fresh heartbeat.
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field

from .. import config
from ..bus.schema import ModuleHealth, StatusMessage

log = logging.getLogger(__name__)


@dataclass
class _ModuleRecord:
    name: str
    last_seen: float = field(default_factory=lambda: 0.0)
    status: str = "down"   # "up" | "degraded" | "down"

    def beat(self) -> None:
        self.last_seen = time.monotonic()
        if self.status != "up":
            log.info("module [%s] → up", self.name)
        self.status = "up"

    def check(self, timeout: float) -> None:
        if self.status == "up" and (time.monotonic() - self.last_seen) > timeout:
            self.status = "degraded"
            log.warning(
                "module [%s] → degraded (no heartbeat for %.1f s)", self.name, timeout
            )


_records: dict[str, _ModuleRecord] = {
    n: _ModuleRecord(n) for n in ("nav", "perception", "recon", "swarm")
}


def beat(module_name: str) -> None:
    """Reset the heartbeat for a module — call on every message received."""
    if r := _records.get(module_name):
        r.beat()


def set_status(module_name: str, status: str) -> None:
    """Explicitly set a module's status (used by modules that manage their own health)."""
    if r := _records.get(module_name):
        if r.status != status:
            log.info("module [%s] → %s", module_name, status)
        r.status = status


def get_status(module_name: str) -> str:
    return _records[module_name].status if module_name in _records else "down"


def _build_status_message() -> StatusMessage:
    nav_up = get_status("nav") == "up"
    return StatusMessage(
        gps="DENIED",
        link="NONE",
        localized=nav_up,
        modules=ModuleHealth(
            nav=get_status("nav"),
            perception=get_status("perception"),
            recon=get_status("recon"),
            swarm=get_status("swarm"),
        ),
    )


async def run_status_loop() -> None:
    """Check heartbeats and broadcast the status topic at STATUS_HZ forever."""
    from ..bus import router   # local import avoids circular at module load time

    dt = 1.0 / config.STATUS_HZ
    while True:
        await asyncio.sleep(dt)

        for r in _records.values():
            r.check(config.HEARTBEAT_TIMEOUT)

        msg = _build_status_message()
        await router.publish("status", msg.model_dump())
