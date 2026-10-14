"""Job queue with one lane per GPU server, so a Krea image never waits behind a video.

Jobs are persisted in the project's Job table for the Jobs tab and history; the work
itself is an async callable. Jobs left 'running' by a crash are marked failed on open.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from sqlmodel import select

from ..db import models as m
from ..db.project import Project
from ..events import bus

log = logging.getLogger(__name__)
Work = Callable[["JobCtx"], Awaitable[dict | None]]


class Cancelled(Exception):
    pass


@dataclass
class JobCtx:
    job_id: int
    project: Project
    queue: "JobQueue"

    def progress(self, fraction: float | None = None, message: str | None = None, **extra):
        with self.project.session() as s:
            j = s.get(m.Job, self.job_id)
            if fraction is not None:
                j.progress = max(0.0, min(1.0, fraction))
            if message is not None:
                j.message = message
            s.add(j)
        bus.publish("job", id=self.job_id, state="running", progress=fraction, message=message, **extra)

    def check(self):
        if self.job_id in self.queue.cancelled:
            raise Cancelled()


class JobQueue:
    def __init__(self, project: Project, lanes: dict[str, int] | list[str]):
        """lanes: name -> number of jobs that may run at once (a list means 1 each)."""
        self.project = project
        self.widths = dict(lanes) if isinstance(lanes, dict) else {lane: 1 for lane in lanes}
        self.queues: dict[str, asyncio.Queue] = {lane: asyncio.Queue() for lane in self.widths}
        self.running: dict[str, set[int]] = {lane: set() for lane in self.widths}
        self.tasks: dict[int, asyncio.Task] = {}
        self.workers: list[asyncio.Task] = []
        self.cancelled: set[int] = set()
        self.on_cancel: dict[int, Callable[[], Awaitable[None]]] = {}
        with project.session() as s:
            for j in s.exec(select(m.Job).where(m.Job.state.in_(["running", "queued"]))).all():
                j.state, j.message = "failed", "interrupted: the harness was restarted"
                s.add(j)

    def start(self):
        for lane, q in self.queues.items():
            for i in range(self.widths[lane]):
                self.workers.append(asyncio.create_task(self._worker(lane, q), name=f"lane-{lane}-{i}"))

    def pending(self, lane: str) -> int:
        """Jobs in a lane that are queued or running."""
        return self.queues[lane].qsize() + len(self.running[lane])

    async def stop(self, reason: str = "the harness was shut down"):
        """Interrupt running jobs (their on_cancel hooks stop ComfyUI work), stop the
        lanes, and record unfinished jobs as cancelled rather than crashed."""
        for jid, cb in list(self.on_cancel.items()):
            self.cancelled.add(jid)
            try:
                await asyncio.wait_for(cb(), 10)
            except Exception as e:
                log.warning("cancel hook for job %s failed: %s", jid, e)
        for w in self.workers:
            w.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        with self.project.session() as s:
            for j in s.exec(select(m.Job).where(m.Job.state.in_(["running", "queued"]))).all():
                j.state, j.message, j.finished = "cancelled", reason, m.now()
                s.add(j)

    def submit(self, kind: str, lane: str, work: Work, payload: dict[str, Any] | None = None) -> int:
        with self.project.session() as s:
            j = m.Job(kind=kind, server=lane, payload=payload or {})
            s.add(j)
            s.flush()
            jid = j.id
        self.queues[lane].put_nowait((jid, work))
        bus.publish("job", id=jid, state="queued", job_kind=kind, lane=lane, payload=payload or {})
        return jid

    async def cancel(self, job_id: int):
        self.cancelled.add(job_id)
        cb = self.on_cancel.get(job_id)
        if cb:
            await cb()
        t = self.tasks.get(job_id)
        if t and not t.done():
            t.cancel()          # also stops a job that is only waiting (for a GPU, a pick, ...)

    async def _worker(self, lane: str, q: asyncio.Queue):
        while True:
            jid, work = await q.get()
            if jid in self.cancelled:
                self._finish(jid, "cancelled", "cancelled before start")
                continue
            with self.project.session() as s:
                j = s.get(m.Job, jid)
                j.state, j.started = "running", m.now()
                s.add(j)
            bus.publish("job", id=jid, state="running", progress=0)
            task = asyncio.create_task(work(JobCtx(jid, self.project, self)), name=f"job-{jid}")
            self.tasks[jid] = task
            self.running[lane].add(jid)
            try:
                result = await task
                self._finish(jid, "done", "", result or {})
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():     # the lane itself is stopping
                    task.cancel()
                    raise
                self._finish(jid, "cancelled", "cancelled")
            except Cancelled:
                self._finish(jid, "cancelled", "cancelled")
            except Exception as e:
                log.error("job %s failed:\n%s", jid, traceback.format_exc())
                self._finish(jid, "failed", f"{type(e).__name__}: {e}")
            finally:
                self.on_cancel.pop(jid, None)
                self.tasks.pop(jid, None)
                self.running[lane].discard(jid)

    def _finish(self, jid: int, state: str, message: str, result: dict | None = None):
        with self.project.session() as s:
            j = s.get(m.Job, jid)
            j.state, j.message, j.finished = state, message[:4000], m.now()
            if state == "done":
                j.progress = 1.0
            if result:
                j.result = result
            s.add(j)
        bus.publish("job", id=jid, state=state, message=message, result=result or {})
