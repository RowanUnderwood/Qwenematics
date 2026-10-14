"""In-process pub/sub: jobs, the agent and services publish; websocket clients subscribe."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subs.discard(q)

    def publish(self, event: str, /, **data: Any):
        if "kind" in data:
            raise ValueError("'kind' is reserved for the event name; use another payload key")
        msg = {"kind": event, **data}
        for q in list(self._subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:          # a stuck browser tab must not block the harness
                log.warning("dropping slow event subscriber")
                self._subs.discard(q)

    @staticmethod
    def encode(msg: dict) -> str:
        return json.dumps(msg, default=str)


bus = EventBus()
