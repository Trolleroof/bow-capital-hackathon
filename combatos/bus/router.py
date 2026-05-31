"""Topic-based pub/sub router for the CombatOS orchestrator bus.

All state is module-level so every coroutine in the process shares one router.
Thread-safety is not needed — everything runs in a single asyncio event loop.

Usage:
    q = asyncio.Queue(maxsize=256)
    router.subscribe(q, ["pose", "status"])   # None = all topics
    router.unsubscribe(q)                     # on disconnect

    await router.publish("pose", {"t": 1.0, "x": 0, ...})
"""
from __future__ import annotations
import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

_drop_counts: dict[str, int] = defaultdict(int)

# Last serialized frame for each topic. New dashboard subscribers receive this
# immediately so reloads do not wait for the next ROS/perception publish tick.
_latest: dict[str, str] = {}

# topic → set of subscriber queues
_topic_subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

# Queues subscribed to ALL topics (no filter)
_wildcard_subs: set[asyncio.Queue] = set()


def _enqueue_latest(q: asyncio.Queue, frame: str, topic: str) -> None:
    try:
        q.put_nowait(frame)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.put_nowait(frame)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            _drop_counts[topic] += 1


def subscribe(q: asyncio.Queue, topics: list[str] | None = None) -> None:
    """Register queue `q` for the given topics.  topics=None means all topics."""
    if topics is None:
        _wildcard_subs.add(q)
    else:
        for t in topics:
            _topic_subs[t].add(q)


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove queue `q` from every subscription set."""
    _wildcard_subs.discard(q)
    for s in _topic_subs.values():
        s.discard(q)


def replay_latest(q: asyncio.Queue, topics: list[str] | None = None) -> None:
    """Queue the latest known message for each subscribed topic."""
    if topics is None:
        items = list(_latest.items())
    else:
        items = [(topic, _latest[topic]) for topic in topics if topic in _latest]
    for topic, frame in items:
        _enqueue_latest(q, frame, topic)


async def publish(
    topic: str,
    payload: dict[str, Any],
    exclude: asyncio.Queue | None = None,
) -> None:
    """Fan `payload` out to every subscriber of `topic`.

    `exclude` is the sender's own queue — used by the WS server to prevent
    a publishing client from receiving its own message back.
    """
    frame = json.dumps({"topic": topic, **payload})
    _latest[topic] = frame
    targets = _wildcard_subs | _topic_subs.get(topic, set())
    for q in list(targets):
        if q is exclude:
            continue
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(frame)
                _drop_counts[topic] += 1
                if _drop_counts[topic] in (1, 10, 100) or _drop_counts[topic] % 1000 == 0:
                    log.warning(
                        "queue full on topic=%s; dropped oldest message and kept latest (count=%d)",
                        topic,
                        _drop_counts[topic],
                    )
            except asyncio.QueueEmpty:
                pass
            except asyncio.QueueFull:
                _drop_counts[topic] += 1
