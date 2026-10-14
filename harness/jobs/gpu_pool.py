"""Which GPU renders each image: the 3090 (image ComfyUI) or a borrowed, idle 5090 (video ComfyUI).

Measured (tools/bench_gpu_split.py, 2026-10-14): Krea takes ~33 s/image on the 3090 and
~11 s on the 5090 (+~3 s for the first after MiniMax was there). Switching the 5090 back to
MiniMax costs ~26 s, and the two models can't share its 32 GB, so it's emptied each way.

Rules
- Video first: while a shot job holds or waits for the video slot, no image starts on the
  5090; the shot waits only for the image already rendering there.
- Borrow the 5090 when it already has Krea loaded, or when at least `min_backlog` image
  jobs are in flight (one image alone isn't worth the switch back).
- Among free GPUs, a waiting image goes where it is expected to finish first, from running
  averages of each GPU's render time (so a short batch queues for the fast 5090 rather than
  taking the slow 3090).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

KREA, MINIMAX = "krea", "minimax"
log = logging.getLogger(__name__)


class GpuPool:
    def __init__(self, *, image: str = "image", video: str = "video",
                 free: Callable[[str], Awaitable[None]],
                 can_borrow: Callable[[], bool],
                 backlog: Callable[[], int],
                 min_backlog: Callable[[], int] = lambda: 2,
                 est: dict[str, float] | None = None,
                 on_change: Callable[[], None] = lambda: None):
        self.I, self.V = image, video
        self._free, self._can_borrow, self._backlog, self._min_backlog = free, can_borrow, backlog, min_backlog
        self.est = {image: 33.0, video: 11.0, **(est or {})}       # running average seconds per image
        self.busy = {image: False, video: False}
        self.started = {image: 0.0, video: 0.0}
        self.loaded: dict[str, str | None] = {image: KREA, video: None}   # None = unknown / not Krea
        self.video_demand = 0
        self._waiters: list[asyncio.Future] = []
        self._video_wait: asyncio.Event = asyncio.Event()
        self._on_change = on_change

    def _set_loaded(self, server: str, family: str | None):
        if self.loaded[server] != family:
            self.loaded[server] = family
            self._on_change()

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        return {"busy": dict(self.busy), "loaded": dict(self.loaded), "waiting_images": len(self._waiters),
                "video_demand": self.video_demand, "est_s": {k: round(v, 1) for k, v in self.est.items()}}

    def borrowable(self) -> bool:
        if self.video_demand or not self._can_borrow():
            return False
        return self.loaded[self.V] == KREA or self._backlog() >= self._min_backlog()

    def _remaining(self, server: str) -> float:
        if not self.busy[server]:
            return 0.0
        return max(1.0, self.est[server] - (time.monotonic() - self.started[server]))

    # ------------------------------------------------------------ dispatch
    def _dispatch(self):
        self._waiters = [w for w in self._waiters if not w.done()]
        while self._waiters:
            borrow = self.borrowable()
            if borrow and not self.busy[self.V]:
                self._take(self._waiters.pop(0), self.V)
                continue
            if self.busy[self.I]:
                return
            if not borrow:
                self._take(self._waiters.pop(0), self.I)
                continue
            # 5090 busy, 3090 free: give the 3090 to the first waiter that would otherwise
            # wait longer for the 5090 than the 3090 takes to render it.
            for i, w in enumerate(self._waiters):
                if self._remaining(self.V) + (i + 1) * self.est[self.V] > self.est[self.I]:
                    self._take(self._waiters.pop(i), self.I)
                    break
            return

    def _take(self, fut: asyncio.Future, server: str):
        log.info("image -> %s (waiting %d, backlog %d, video has %s)", server, len(self._waiters),
                 self._backlog(), self.loaded[self.V])
        self.busy[server] = True
        self.started[server] = time.monotonic()
        fut.set_result(server)

    def _release(self, server: str, seconds: float | None):
        if seconds:
            self.est[server] = 0.7 * self.est[server] + 0.3 * seconds
        self.busy[server] = False
        self._video_wait.set()
        self._dispatch()

    # ------------------------------------------------------------ slots
    @asynccontextmanager
    async def image_slot(self):
        """Yields the server name to render one Krea image on."""
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        self._dispatch()
        try:
            server = await fut
        except BaseException:
            if fut.done() and not fut.cancelled():        # assigned just as we were cancelled
                self._release(fut.result(), None)
            else:
                fut.cancel()
                self._dispatch()
            raise
        t, ok = time.monotonic(), False
        try:
            if server == self.V and self.loaded[self.V] != KREA:
                await self._free(self.V)                      # MiniMax out before Krea comes in
                self._set_loaded(self.V, KREA)
            yield server
            ok = True
        finally:
            self._release(server, time.monotonic() - t if ok else None)

    @asynccontextmanager
    async def video_slot(self):
        """Hold the 5090 for MiniMax: blocks new image borrows at once, waits for an image
        already rendering there, and empties Krea out of it first."""
        self.video_demand += 1
        held = False
        t = time.monotonic()
        try:
            while self.busy[self.V]:
                self._video_wait.clear()
                await self._video_wait.wait()
            self.busy[self.V] = held = True
            emptied = self.loaded[self.V] == KREA
            if emptied:
                await self._free(self.V)
            log.info("video slot after %.1f s wait%s", time.monotonic() - t, " (Krea emptied)" if emptied else "")
            self._set_loaded(self.V, MINIMAX)
            yield
        finally:
            self.video_demand -= 1
            if held:
                self.busy[self.V] = False
            self._video_wait.set()
            self._dispatch()
