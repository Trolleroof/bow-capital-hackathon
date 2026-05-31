"""Fallback decision helpers — see §7 of TEAM_PLAN.

Each function answers a yes/no question: should this vertical fall back to its
stub/mock right now?  Centralising the decision here means modules don't need
to import each other.
"""
from __future__ import annotations
from . import system_state


def nav_is_live() -> bool:
    return system_state.get_status("nav") == "up"


def perception_is_live() -> bool:
    return system_state.get_status("perception") == "up"


def recon_is_ready() -> bool:
    return system_state.get_status("recon") == "up"


def swarm_is_live() -> bool:
    return system_state.get_status("swarm") == "up"
