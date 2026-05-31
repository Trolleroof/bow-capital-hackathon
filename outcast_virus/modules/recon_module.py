"""Recon module — watches for the 3DGS splat asset and emits recon status.

Training happens offline on Colab (⓷).  When the finished splat file lands at
RECON_ASSET_PATH (relative to repo root), this module detects it, emits a
"recon" → ready message, and marks the module healthy.

An optional JSON sidecar at RECON_FRAMES_SIDECAR can carry metadata:
    { "frames_used": 220, "splat_url": "/assets/field.splat" }

See outcast_virus/HOOK_RECON.md for where to put the files and how to signal progress.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os

from .base import AbstractModule
from ..bus import router
from .. import config
from ..state import system_state

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _asset_path() -> str:
    return os.path.join(_REPO_ROOT, config.RECON_ASSET_PATH)


def _sidecar_path() -> str:
    return os.path.join(_REPO_ROOT, config.RECON_FRAMES_SIDECAR)


def _read_sidecar() -> dict:
    try:
        with open(_sidecar_path()) as f:
            return json.load(f)
    except Exception:
        return {}


class ReconModule(AbstractModule):
    name = "recon"

    async def run(self) -> None:
        last_status: str | None = None
        log.info("[recon] polling for splat asset at %s", config.RECON_ASSET_PATH)

        while True:
            asset = _asset_path()
            if os.path.exists(asset):
                meta = _read_sidecar()
                frames = meta.get("frames_used", 0)
                splat_url = meta.get("splat_url", config.RECON_ASSET_URL)

                system_state.beat("recon")

                if last_status != "ready":
                    size_kb = os.path.getsize(asset) // 1024
                    log.info("[recon] splat asset ready (%d KB, %d frames)", size_kb, frames)
                    last_status = "ready"
                    await router.publish("recon", {
                        "status": "ready",
                        "splat_url": splat_url,
                        "frames_used": frames,
                    })
            else:
                system_state.set_status("recon", "down")

                if last_status != "training":
                    log.info("[recon] asset not found — broadcasting status: training")
                    last_status = "training"
                    await router.publish("recon", {
                        "status": "training",
                        "splat_url": "",
                        "frames_used": 0,
                    })

            await asyncio.sleep(config.RECON_POLL_INTERVAL)
