"""Topic-based pub/sub router for the CombatOS image bus."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

_drop_counts: dict[str, int] = defaultdict(int)

_topic_subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
_wildcard_subs: set[asyncio.Queue] = set()


def subscribe(q: asyncio.Queue, topics: list[str] | None = None) -> None:
    if topics is None:
        _wildcard_subs.add(q)
    else:
        for topic in topics:
            _topic_subs[topic].add(q)


def unsubscribe(q: asyncio.Queue) -> None:
    _wildcard_subs.discard(q)
    for subs in _topic_subs.values():
        subs.discard(q)


async def publish(
    topic: str,
    payload: dict[str, Any],
    exclude: asyncio.Queue | None = None,
) -> None:
    frame = json.dumps({"topic": topic, **payload})
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
                        "image queue full on topic=%s; dropped oldest frame and kept latest (count=%d)",
                        topic,
                        _drop_counts[topic],
                    )
            except asyncio.QueueEmpty:
                pass
            except asyncio.QueueFull:
                _drop_counts[topic] += 1
