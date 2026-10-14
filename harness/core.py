"""The harness's domain layer. The HTTP API and Qwen's tools both call into this.

One Harness per process holds the servers, the workflow registry, the LLM and the
open project. Long work (renders, the final cut) runs as queued jobs; everything
else is a quick DB operation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from sqlmodel import select

from .comfy.client import ComfyClient
from .comfy.manifest import Registry, WorkflowError
from .comfy.runner import Runner
from .config import Settings, get_settings
from .db import deps
from .db import models as m
from .db.project import Project, next_id
from .events import bus
from .jobs.gpu_pool import GpuPool
from .jobs.queue import JobCtx, JobQueue
from . import styles
from .llm import media, roles
from .llm.client import LLM

log = logging.getLogger(__name__)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

IMAGE_WF, VIDEO_WF = "krea2", "minimax_h3"
ROLE_SLOT_ORDER = {"first_frame": 0, "last_frame": 1, "subject": 2, "style": 3}

DEFAULT_PROJECT_SETTINGS: dict[str, Any] = {
    "video_aspect": "16:9 (Widescreen)",
    "krea_style": None,                          # locked style name from the Krea style library
    "params": {IMAGE_WF: {}, VIDEO_WF: {}},     # user-locked overrides per workflow
    "iterate": {},                               # overrides of settings.yaml 'iterate'
}


class HarnessError(Exception):
    pass


class Harness:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.registry = Registry(s.workflows_root, s.preferred_workflows)
        self.clients = {n: ComfyClient(c["url"], n) for n, c in s.servers.items()}
        self.object_info: dict[str, dict] = {}
        self.runner = Runner(self.registry, self.clients, self.object_info, s.comfy_dir)
        self.llm = LLM(s.llm["base_url"], s.llm["model"], s.llm.get("reasoning_effort", "low"))
        self.project: Project | None = None
        self.queue: JobQueue | None = None
        self.picks: dict[int, tuple[asyncio.Event, list]] = {}    # job id -> (event, [chosen render id])
        self.krea_on_video: list[str] = ["video server not checked yet"]   # problems running Krea there
        g = s.gpu_sharing
        self.pool = GpuPool(free=lambda name: self.clients[name].free(),
                            can_borrow=self._can_borrow_video,
                            backlog=lambda: self.queue.pending("image") if self.queue else 0,
                            min_backlog=lambda: int(self.settings.gpu_sharing.get("min_backlog", 2)),
                            est={k: float(v) for k, v in (g.get("est_seconds") or {}).items()},
                            on_change=lambda: bus.publish("servers"))

    def _can_borrow_video(self) -> bool:
        return (self.settings.gpu_sharing.get("image_on_video_gpu", "auto") != "off"
                and not self.krea_on_video and "video" in self.object_info)

    def gpu_label(self, server: str) -> str:
        return self.settings.servers.get(server, {}).get("expect_gpu", server).replace("RTX ", "")

    # ------------------------------------------------------------ servers
    async def refresh_servers(self, full: bool = False) -> dict:
        """Ping servers. object_info (large) is fetched when a server first appears up,
        or on full=True (e.g. after installing nodes or models)."""
        status = {}
        changed = False
        for name, c in self.clients.items():
            up = await c.alive()
            status[name] = {"url": c.base, "up": up}
            if not up and name in self.object_info:
                del self.object_info[name]
                changed = True
            if up:
                try:
                    if full or name not in self.object_info:
                        self.object_info[name] = await c.object_info()
                        changed = True
                    st = await c.system_stats()
                    status[name]["gpu"] = ", ".join(d["name"] for d in st.get("devices", []))
                except Exception as e:
                    status[name]["error"] = str(e)
        if changed:
            if full:
                self.registry.reload()
            self.registry.check_all(self.object_info)
            self._check_krea_on_video()
        for name in self.clients:
            if name in status:
                status[name]["loaded"] = self.pool.loaded.get(name)
        status["llm"] = {"url": self.llm.base, "up": await self.llm.alive()}
        return status

    def _check_krea_on_video(self):
        """Krea normally targets the image server; check it also runs on the video one
        (same ComfyUI install, so normally yes) before images may borrow the 5090."""
        try:
            wv = self.registry.get(IMAGE_WF)
        except Exception as e:
            self.krea_on_video = [str(e)]
            return
        oi = self.object_info.get("video")
        self.krea_on_video = ["video server is down"] if oi is None else wv.problems_on(oi)

    def workflows_status(self) -> list[dict]:
        out = []
        for wf_id, vs in self.registry.versions.items():
            for v, wv in sorted(vs.items()):
                out.append({"key": wv.key, "label": wv.manifest.get("label"), "server": wv.manifest.get("server"),
                            "checked": wv.manifest.get("server") in self.object_info,
                            "usable": wv.usable, "problems": wv.problems,
                            "preferred": self.registry.preferred.get(wf_id) == v})
        for path, err in self.registry.errors.items():
            out.append({"key": path, "usable": False, "problems": [err]})
        return out

    # ------------------------------------------------------------ projects
    async def open_project(self, path: Path) -> Project:
        await self.close_project()
        self.project = Project.open(path)
        with self.project.session() as s:      # 'rendering' left by a job that died with the old process
            for sh in s.exec(select(m.Shot).where(m.Shot.status == "rendering")).all():
                sh.status = "done" if sh.final_render_id else "draft"
                s.add(sh)
        self._start_queue()
        return self.project

    async def create_project(self, name: str) -> Project:
        await self.close_project()
        self.project = Project.create(self.settings.projects_root, name)
        with self.project.session() as s:
            s.add(m.Meta(key="settings", value=json.dumps(DEFAULT_PROJECT_SETTINGS)))
        self._start_queue()
        return self.project

    def _start_queue(self):
        width = max(1, int(self.settings.gpu_sharing.get("image_jobs_in_flight", 3)))
        self.queue = JobQueue(self.project, {"image": width, "video": 1, "local": 1})
        self.queue.start()

    async def close_project(self):
        if self.queue:
            await self.queue.stop()
        if self.project:
            self.project.close()
        self.project, self.queue = None, None

    def p(self) -> Project:
        if self.project is None:
            raise HarnessError("no project is open")
        return self.project

    def project_settings(self) -> dict:
        with self.p().session() as s:
            row = s.get(m.Meta, "settings")
        cfg = json.loads(row.value) if row else {}
        merged = json.loads(json.dumps(DEFAULT_PROJECT_SETTINGS))
        for k, v in cfg.items():
            merged[k] = v
        return merged

    def set_project_settings(self, cfg: dict):
        with self.p().session() as s:
            s.merge(m.Meta(key="settings", value=json.dumps(cfg)))
        bus.publish("project_settings", settings=cfg)

    def iterate_cfg(self) -> dict:
        return {**self.settings.iterate, **self.project_settings().get("iterate", {})}

    def chat_event(self, text: str, attachments: list[dict]):
        """A non-conversational chat line (render finished, etc.) shown with its media.
        The director's history skips role 'event', so these cost no context."""
        with self.p().session() as s:
            row = m.ChatMessage(role="event", content=text, attachments=attachments)
            s.add(row)
            s.flush()
            rid = row.id
        bus.publish("chat_message", id=rid, role="event", content=text, attachments=attachments)

    # ------------------------------------------------------------ bible
    def bible_text(self, max_chars: int = 6000) -> str:
        with self.p().session() as s:
            rows = s.exec(select(m.BibleEntry).order_by(m.BibleEntry.kind, m.BibleEntry.id)).all()
        parts = [f"[{r.kind}] {r.title}: {r.body}".strip() for r in rows]
        text = "\n".join(parts)
        return text[:max_chars]

    def update_bible(self, kind: str, title: str, body: str, entry_id: int | None = None) -> m.BibleEntry:
        with self.p().session() as s:
            e = s.get(m.BibleEntry, entry_id) if entry_id else s.exec(
                select(m.BibleEntry).where(m.BibleEntry.kind == kind, m.BibleEntry.title == title)).first()
            if e is None:
                e = m.BibleEntry(kind=kind, title=title)
            e.kind, e.title, e.body, e.updated = kind, title, body, m.now()
            s.add(e)
            s.flush()
        bus.publish("bible", id=e.id)
        return e

    # ------------------------------------------------------------ references
    def search_references(self, text: str = "", kind: str | None = None, limit: int = 8) -> list[dict]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]
        out = []
        with self.p().session() as s:
            for ref in s.exec(select(m.Reference)).all():
                if kind and ref.kind != kind:
                    continue
                v = s.get(m.RefVersion, ref.current_version_id) if ref.current_version_id else None
                hay = {"name": ref.name.lower(), "tags": " ".join(ref.tags).lower(),
                       "cap": (v.caption if v else "").lower(), "desc": ref.description.lower()}
                score = sum(3 * (w in hay["name"]) + 2 * (w in hay["tags"]) + (w in hay["cap"]) + (w in hay["desc"])
                            for w in words) if words else 1
                if score:
                    out.append((score, self.ref_summary(s, ref)))
        out.sort(key=lambda x: -x[0])
        return [r for _, r in out[:limit]]

    def ref_summary(self, s, ref: m.Reference) -> dict:
        v = s.get(m.RefVersion, ref.current_version_id) if ref.current_version_id else None
        imp = deps.ref_impact(s, ref.id)
        return {"ref_id": ref.id, "name": ref.name, "kind": ref.kind, "tags": ref.tags,
                "description": ref.description, "current_version": v.id if v else None,
                "version_n": v.n if v else None, "caption": v.caption if v else "",
                "file_id": v.file_id if v else None,
                "used_in_shots": sorted({x["shot_id"] for x in imp.shots})}

    def create_reference(self, name: str, kind: str, description: str, tags: list[str] | None = None,
                         render: bool = True, enhance: bool = True) -> tuple[int, int | None]:
        """With enhance off, `description` is the exact Krea prompt (Qwen doesn't rewrite it)."""
        with self.p().session() as s:
            ref = m.Reference(name=name, kind=kind, description=description, tags=tags or [])
            s.add(ref)
            s.flush()
            ref_id = ref.id
        bus.publish("reference", id=ref_id)
        job = self.queue.submit("ref_image", "image", lambda ctx: self._ref_job(ctx, ref_id, None, None if enhance else description, enhance),
                                {"ref_id": ref_id, "name": name}) if render else None
        return ref_id, job

    def revise_reference(self, ref_id: int, change: str | None = None, prompt: str | None = None,
                         enhance: bool = True) -> int:
        """A new version from a change request (Qwen rewrites the prompt) or from an edited
        prompt (sent as-is unless enhance, which has Qwen polish it)."""
        change, prompt = (change or "").strip() or None, (prompt or "").strip() or None
        if not change and not prompt:
            raise HarnessError("describe a change or give a prompt")
        with self.p().session() as s:
            ref = s.get(m.Reference, ref_id)
            if ref is None:
                raise HarnessError(f"no reference {ref_id}")
            name = ref.name
        return self.queue.submit("ref_image", "image", lambda ctx: self._ref_job(ctx, ref_id, change, prompt, enhance),
                                 {"ref_id": ref_id, "name": name, "change": change or "edited prompt"})

    def import_reference_file(self, src: Path, name: str, kind: str, description: str = "",
                              ref_id: int | None = None) -> int:
        """A user-supplied image becomes a reference version (source=uploaded)."""
        proj = self.p()
        dest = proj.media("uploads") / f"{int(time.time() * 1000)}_{src.name}"
        shutil.copy2(src, dest)
        with proj.session() as s:
            if ref_id is None:
                ref = m.Reference(name=name, kind=kind, description=description)
                s.add(ref)
                s.flush()
            else:
                ref = s.get(m.Reference, ref_id)
            n = 1 + max([v.n for v in s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref.id)).all()] or [0])
            f = proj.register_file(s, dest, "image", "ref_version")
            v = m.RefVersion(id=next_id(s, m.RefVersion), ref_id=ref.id, n=n, file_id=f.id, source="uploaded",
                             caption=description)
            s.add(v)
            s.flush()
            f.owner_id = v.id
            ref.current_version_id = v.id
            s.add_all([f, ref])
            vid = v.id
        bus.publish("reference", id=ref.id)
        return vid

    async def _ref_job(self, ctx: JobCtx, ref_id: int, change: str | None, user_prompt: str | None = None,
                       enhance: bool = True) -> dict:
        proj = self.p()
        wv = self.registry.get(IMAGE_WF)
        if not wv.usable:
            raise HarnessError(f"{wv.key} is not usable: {wv.problems[:3]}")
        it = self.iterate_cfg()
        # Critic rounds rewrite the prompt, so a verbatim prompt gets one render.
        rounds = 1 + (int(it.get("max_image_iterations", 0)) if it.get("auto", True) and enhance else 0)
        pset = self.project_settings()
        with proj.session() as s:
            ref = s.get(m.Reference, ref_id)
            base = s.get(m.RefVersion, ref.current_version_id) if ref.current_version_id else None
            name, kind, desc = ref.name, ref.kind, ref.description
            prev_prompt = base.prompt if base else None
        bible = self.bible_text()
        style = styles.style_line(pset.get("krea_style"))
        ctx.progress(0.02, "writing image prompt")
        tm = {"write_s": 0.0, "wait_gpu_s": 0.0, "render_s": 0.0, "critic_s": 0.0}
        t0 = time.monotonic()
        if user_prompt and not enhance:
            prompt = styles.enforce_style(user_prompt, style)
        elif user_prompt:
            prompt = await roles.write_krea_prompt(
                self.llm, bible=bible, style=style, name=name, kind=kind, description=desc, previous_prompt=user_prompt,
                change=change or "Polish this prompt for Krea; keep every edit the user made.")
        else:
            prompt = await roles.write_krea_prompt(self.llm, bible=bible, style=style, name=name, kind=kind,
                                                   description=desc, previous_prompt=prev_prompt if change else None,
                                                   change=change)
        tm["write_s"] += time.monotonic() - t0
        best: tuple[float, int] | None = None
        critique_txt = None
        for i in range(rounds):
            ctx.check()
            if i:
                ctx.progress(i / rounds, f"revising prompt (round {i + 1})")
                t0 = time.monotonic()
                prompt = await roles.write_krea_prompt(self.llm, bible=bible, style=style, name=name, kind=kind, description=desc,
                                                       previous_prompt=prompt, critique=critique_txt)
                tm["write_s"] += time.monotonic() - t0
            params = wv.resolve_params({**pset["params"].get(IMAGE_WF, {}), "prompt": prompt, "ref_kind": kind},
                                       context={"video_aspect": pset["video_aspect"]})
            ctx.progress((i + 0.05) / rounds, f"waiting for a GPU for {name}")
            tag = f"ref{ref_id}_{int(time.time())}"
            t0 = time.monotonic()
            async with self.pool.image_slot() as server:
                tm["wait_gpu_s"] += time.monotonic() - t0
                ctx.check()
                ctx.progress((i + 0.1) / rounds, f"rendering {name} on {self.gpu_label(server)} (round {i + 1}/{rounds})")
                self.queue.on_cancel[ctx.job_id] = self.clients[server].interrupt
                t0 = time.monotonic()
                res = await self.runner.render(wv, "t2i", params, dest_dir=proj.media("refs"), tag=tag, server=server,
                                               on_progress=_comfy_progress(ctx, i / rounds, 1 / rounds))
                tm["render_s"] += time.monotonic() - t0
            self.queue.on_cancel.pop(ctx.job_id, None)
            params["_server"] = server
            img = res.outputs["image"][0]
            crit: dict = {}
            if rounds > 1:
                t0 = time.monotonic()
                part, _ = media.image_part(img)
                crit = await roles.critique(self.llm, intent=f"{kind} reference '{name}': {desc}",
                                            prompt=prompt, media=[part])
                tm["critic_s"] += time.monotonic() - t0
            with proj.session() as s:
                n = 1 + max([v.n for v in s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref_id)).all()] or [0])
                f = proj.register_file(s, img, "image", "ref_version")
                v = m.RefVersion(id=next_id(s, m.RefVersion), ref_id=ref_id, n=n, file_id=f.id, prompt=prompt,
                                 params=_jsonable(params), workflow=wv.key, seed=params.get("seed"), critic=crit,
                                 parent_id=base.id if base else None)
                s.add(v)
                s.flush()
                f.owner_id = v.id
                s.add(f)
                vid = v.id
            bus.publish("ref_version", ref_id=ref_id, version_id=vid, score=crit.get("score"))
            score = float(crit.get("score", 0) or 0)
            if best is None or score > best[0]:
                best = (score, vid)
            if rounds == 1 or score >= float(it.get("accept_score", 7)):
                break
            critique_txt = crit.get("fix") or "; ".join(crit.get("issues", []))
        # caption + make current
        with proj.session() as s:
            v = s.get(m.RefVersion, best[1])
            img = proj.abspath(s.get(m.FileRec, v.file_id))
        part, _ = media.image_part(img, 512)
        cap = await roles.caption(self.llm, image_part=part, name=name, kind=kind)
        with proj.session() as s:
            v = s.get(m.RefVersion, best[1])
            v.caption = cap.get("caption", "")[:400]
            ref = s.get(m.Reference, ref_id)
            if cap.get("tags"):
                ref.tags = sorted(set(ref.tags) | {str(t).lower() for t in cap["tags"]})[:12]
            old_current = ref.current_version_id
            in_use = deps.ref_impact(s, ref_id).shots if old_current else []
            if not in_use:
                ref.current_version_id = v.id       # nothing depends on it yet: just switch
            s.add_all([v, ref])
        result = {"ref_id": ref_id, "version_id": best[1], "score": best[0], **{k: round(v, 1) for k, v in tm.items()}}
        if in_use:
            result["needs_decision"] = {"shots": in_use, "message":
                f"'{name}' is used by {len(in_use)} shot(s). Choose which shots switch to the new version."}
            bus.publish("ref_impact", ref_id=ref_id, version_id=best[1], shots=in_use)
        bus.publish("reference", id=ref_id)
        with proj.session() as s:
            fid = s.get(m.RefVersion, best[1]).file_id
        note = f"Reference '{name}' ready" + (f" (critic {best[0]:.0f}/10)" if best[0] else "")
        if in_use:
            note += f" - used in {len(in_use)} shot(s): choose which switch to it in the References tab"
        self.chat_event(note, [{"type": "ref_version", "id": best[1], "ref_id": ref_id, "file_id": fid,
                                "label": f"{name} v?"}])
        return result

    def apply_ref_version(self, ref_id: int, version_id: int, shot_ids: list[int] | None) -> list[int]:
        with self.p().session() as s:
            changed = deps.replace_ref_version(s, ref_id, version_id, shot_ids)
        bus.publish("reference", id=ref_id)
        for sid in changed:
            bus.publish("shot", id=sid)
        return changed

    # ------------------------------------------------------------ shots
    def upsert_shot(self, shot_id: int | None = None, **fields) -> int:
        allowed = {"title", "description", "dialogue", "audio_notes", "mode", "duration_s", "prompt", "params",
                   "order", "status", "workflow"}
        with self.p().session() as s:
            shot = s.get(m.Shot, shot_id) if shot_id else None
            if shot is None:
                last = s.exec(select(m.Shot).order_by(m.Shot.order.desc())).first()
                shot = m.Shot(order=(last.order + 1) if last else 1)
            for k, v in fields.items():
                if k in allowed and v is not None:
                    if k == "duration_s":
                        v = max(2.0, min(15.0, float(v)))
                    setattr(shot, k, v)
            s.add(shot)
            s.flush()
            sid = shot.id
        bus.publish("shot", id=sid)
        return sid

    def assign_ref(self, shot_id: int, ref_id: int | None = None, version_id: int | None = None,
                   role: str = "subject", slot: int | None = None) -> list[dict]:
        """Put a reference in a shot. Without version_id the slot follows the reference's
        current version; if it has none yet (still rendering) the slot is pending and the
        shot's render waits for it. Slots are re-packed so first/last frames lead and
        <Picture k> numbering stays dense."""
        with self.p().session() as s:
            if version_id is not None:
                v = s.get(m.RefVersion, version_id)
                if v is None:
                    raise HarnessError(f"no reference version {version_id}")
                ref_id = v.ref_id
            ref = s.get(m.Reference, ref_id)
            if ref is None:
                raise HarnessError(f"no reference {ref_id}")
            if version_id is None:
                version_id = ref.current_version_id         # may be None: pending
            slots = s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id)).all()
            if any(x.ref_id == ref_id for x in slots):
                raise HarnessError(f"'{ref.name}' is already in this shot")
            if len(slots) >= 9:
                raise HarnessError("a shot holds at most 9 pictures")
            if role in ("first_frame", "last_frame"):
                same = next((x for x in slots if x.role == role), None)
                if same:
                    raise HarnessError(f"this shot already has a {role.replace('_', ' ')} "
                                       f"('{s.get(m.Reference, same.ref_id).name}'); use role subject or remove it first")
            s.add(m.ShotSlot(shot_id=shot_id, ref_id=ref_id, ref_version_id=version_id, role=role, slot=slot or 99))
            s.flush()
            self._repack(s, shot_id)
        bus.publish("shot", id=shot_id)
        return self.shot_slots(shot_id)

    def remove_slot(self, shot_id: int, slot_id: int):
        with self.p().session() as s:
            x = s.get(m.ShotSlot, slot_id)
            if x and x.shot_id == shot_id:
                s.delete(x)
                s.flush()
                self._repack(s, shot_id)
        bus.publish("shot", id=shot_id)

    def _repack(self, s, shot_id: int):
        slots = s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id)).all()
        slots.sort(key=lambda x: (ROLE_SLOT_ORDER.get(x.role, 5), x.slot, x.id))
        for i, x in enumerate(slots, 1):
            x.slot = i
            s.add(x)

    def shot_slots(self, shot_id: int) -> list[dict]:
        out = []
        with self.p().session() as s:
            for x in s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id).order_by(m.ShotSlot.slot)).all():
                ref = s.get(m.Reference, x.ref_id)
                v = s.get(m.RefVersion, x.ref_version_id) if x.ref_version_id else None
                out.append({"slot_id": x.id, "slot": x.slot, "role": x.role, "ref_id": ref.id, "name": ref.name,
                            "kind": ref.kind, "description": ref.description, "pending": v is None,
                            "version_id": v.id if v else None, "version_n": v.n if v else None,
                            "caption": v.caption if v else "", "file_id": v.file_id if v else None,
                            "outdated": bool(v and ref.current_version_id not in (None, v.id))})
        return out

    async def _resolve_pending_slots(self, ctx: JobCtx, shot_id: int, timeout_s: float = 1800) -> list[dict]:
        """Wait for references that are still rendering, then pin their versions."""
        deadline = time.time() + timeout_s
        while True:
            ctx.check()
            waiting = []
            with self.p().session() as s:
                for x in s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id)).all():
                    if x.ref_version_id is not None:
                        continue
                    ref = s.get(m.Reference, x.ref_id)
                    if ref.current_version_id:
                        x.ref_version_id = ref.current_version_id
                        s.add(x)
                        continue
                    active = [j for j in s.exec(select(m.Job).where(m.Job.kind == "ref_image",
                                                                     m.Job.state.in_(["queued", "running"]))).all()
                              if j.payload.get("ref_id") == ref.id]
                    if not active:
                        raise HarnessError(f"reference '{ref.name}' has no image and none is rendering - "
                                           "render or upload it, or remove it from the shot")
                    waiting.append(ref.name)
            if not waiting:
                return self.shot_slots(shot_id)
            if time.time() > deadline:
                raise HarnessError(f"gave up waiting for references: {', '.join(waiting)}")
            ctx.progress(0.0, f"waiting for reference images: {', '.join(waiting)}")
            await asyncio.sleep(3)

    def active_shot_jobs(self) -> dict[int, m.Job]:
        """shot id -> its queued or running render job."""
        with self.p().session() as s:
            jobs = s.exec(select(m.Job).where(m.Job.kind == "shot", m.Job.state.in_(["queued", "running"]))
                          .order_by(m.Job.id)).all()
        out: dict[int, m.Job] = {}
        for j in jobs:
            out.setdefault(j.payload.get("shot_id"), j)
        return out

    def render_shot(self, shot_id: int, rewrite_prompt: bool = False, change: str | None = None,
                    seeds: int | None = None, enhance: bool = True) -> int:
        """With enhance off the shot's saved prompt goes to MiniMax verbatim and Qwen never rewrites it."""
        busy = self.active_shot_jobs().get(shot_id)
        if busy:
            raise HarnessError(f"shot {shot_id} is already {'rendering' if busy.state == 'running' else 'queued'} "
                               f"(job {busy.id})")
        if not enhance:
            with self.p().session() as s:
                shot = s.get(m.Shot, shot_id)
                if shot is None or not shot.prompt.strip():
                    raise HarnessError("write a video prompt first, or tick Enhance prompt")
            rewrite_prompt, change = False, None
        return self.queue.submit("shot", "video",
                                 lambda ctx: self._shot_job(ctx, shot_id, rewrite_prompt, change, seeds, enhance),
                                 {"shot_id": shot_id, "change": change})

    def choose_preview(self, job_id: int, render_id: int):
        ev = self.picks.get(job_id)
        if not ev:
            raise HarnessError("that job is not waiting for a pick")
        ev[1].append(render_id)
        ev[0].set()

    async def _shot_job(self, ctx: JobCtx, shot_id: int, rewrite: bool, change: str | None,
                        seeds: int | None, enhance: bool = True) -> dict:
        proj = self.p()
        it = self.iterate_cfg()
        pset = self.project_settings()
        with proj.session() as s:
            shot = s.get(m.Shot, shot_id)
            wv = self.registry.get_key(shot.workflow) if shot.workflow else self.registry.get(VIDEO_WF)
            shot_d = shot.model_dump()
        if not wv.usable:
            raise HarnessError(f"{wv.key} is not usable: {wv.problems[:3]}")
        slots = await self._resolve_pending_slots(ctx, shot_id)
        pics = []
        with proj.session() as s:
            for x in slots:
                pics.append(proj.abspath(s.get(m.FileRec, x["file_id"])))
        mode = effective_mode(shot_d["mode"], [x["role"] for x in slots])
        if mode != shot_d["mode"]:
            with proj.session() as s:
                sh = s.get(m.Shot, shot_id)
                sh.mode = mode
                s.add(sh)
            self.chat_event(f"Shot '{shot_d['title']}': {_mode_note(mode)} (mode {shot_d['mode']} -> {mode}; "
                            "same model" + (", all pictures still used as references)" if slots else ")"), [])
            shot_d["mode"] = mode
        duration = shot_d["duration_s"]
        bible = self.bible_text()
        prev_clip = self._previous_clip_note(shot_id)
        prompt = shot_d["prompt"]
        rounds = 1 + (int(it.get("max_video_iterations", 0)) if it.get("auto", True) and enhance else 0)
        critique_txt = None
        picked_id = None
        accept = float(it.get("accept_score", 7))
        versions = [x["version_id"] for x in slots]
        # Hold the 5090 for the whole MiniMax section (previews, pick, review rounds): no image
        # borrows it meanwhile, and Krea is emptied out of it first if it was there.
        # The upscale (stage 2) happens at the final cut.
        async with self.pool.video_slot():
            for rnd in range(rounds):
                ctx.check()
                if not prompt or rewrite or change or critique_txt:
                    ctx.progress(0.01, "writing video prompt" + (f" (round {rnd + 1})" if rnd else ""))
                    prompt = await roles.write_h3_prompt(
                        self.llm, bible=bible, shot=shot_d, mode=mode, duration=duration, slots=slots,
                        style=styles.get(pset.get("krea_style")),
                        header=wv.prompt_header(mode, duration), previous_clip=prev_clip,
                        previous_prompt=prompt or None, critique=critique_txt, change=change)
                    change, rewrite = None, False
                    with proj.session() as s:
                        sh = s.get(m.Shot, shot_id)
                        sh.prompt = prompt
                        s.add(sh)
                    bus.publish("shot", id=shot_id)
                if not prompt.strip():
                    raise HarnessError(f"shot {shot_id}: the video prompt is empty")
                ihash = deps.input_hash(m.Shot(**{**shot_d, "prompt": prompt, "mode": mode}), versions, wv.key)
                base = {**pset["params"].get(VIDEO_WF, {}), **(shot_d.get("params") or {}),
                        "prompt": prompt, "duration_s": duration, "aspect_ratio": pset["video_aspect"]}
                n = int(seeds or base.get("preview_seeds") or wv.params["preview_seeds"]["default"])
                more = rnd < rounds - 1
                previews: list[tuple[int, dict, Any]] = []
                self.queue.on_cancel[ctx.job_id] = self.clients[wv.manifest["server"]].interrupt
                for i in range(n):
                    ctx.check()
                    params = wv.resolve_params({k: v for k, v in base.items() if k != "seed"})
                    ctx.progress(0.05 + 0.8 * i / n, f"preview {i + 1}/{n} (seed {params['seed']})"
                                 + (f", round {rnd + 1}" if rnd else ""))
                    tag = f"shot{shot_id}_p{int(time.time())}_{i}"
                    try:
                        res = await self.runner.render(wv, "stage1", params, mode=mode, pictures=pics,
                                                       dest_dir=proj.media("previews"), tag=tag,
                                                       on_progress=_comfy_progress(ctx, 0.05 + 0.8 * i / n, 0.8 / n))
                    except WorkflowError as e:
                        raise HarnessError(f"shot {shot_id}: {e}")
                    rid = self._save_render(shot_id, "preview", wv.key, mode, prompt, params, slots, res, ihash=ihash)
                    previews.append((rid, params, res))
                    bus.publish("render", id=rid, shot_id=shot_id, stage="preview")
                # --- seed hunt: the critic picks and scores (a lone preview is scored only if
                # a review round may follow); the user may override the pick
                pick_idx, crit = 0, {}
                if n > 1 or more:
                    ctx.progress(0.87, "comparing previews" if n > 1 else "reviewing the preview")
                    parts = []
                    for k, (_, _, res) in enumerate(previews, 1):
                        vp, _ = media.video_part(res.outputs["preview"][0], proj.media("proxies"))
                        parts += ([{"type": "text", "text": f"Candidate {k}:"}] if n > 1 else []) + [vp]
                    try:
                        crit = await roles.critique(self.llm, intent=shot_d["description"], prompt=prompt,
                                                    media=parts, candidates=n if n > 1 else 0)
                        pick_idx = max(0, min(n - 1, int(crit.get("pick", 1)) - 1))
                    except Exception as e:
                        log.warning("critic failed: %s", e)
                if n > 1:
                    pick_idx = await self._await_user_pick(ctx, shot_id, previews, pick_idx, crit,
                                                           float(it.get("seedhunt_auto_proceed_s", 10)))
                picked_id, params, _ = previews[pick_idx]
                with proj.session() as s:
                    r = s.get(m.Render, picked_id)
                    r.picked, r.critic = True, crit
                    s.add(r)
                    sh = s.get(m.Shot, shot_id)
                    sh.final_render_id, sh.status = picked_id, "done"
                    s.add(sh)
                    pfid = r.file_id
                bus.publish("render", id=picked_id, shot_id=shot_id, stage="preview")
                bus.publish("shot", id=shot_id)
                score = float(crit.get("score", 0) or 0) if crit else None
                self.chat_event(f"Shot '{shot_d['title']}' ready (preview, seed {params.get('seed')}"
                                + (f", critic {score:.0f}/10" if score is not None else "")
                                + "); it is upscaled at the final cut",
                                [{"type": "render", "id": picked_id, "shot_id": shot_id, "file_id": pfid,
                                  "label": shot_d["title"], "video": True}])
                if not more or score is None or score >= accept:
                    break
                critique_txt = crit.get("fix") or "; ".join(crit.get("issues", [])) or "the critic scored it low"
                self.chat_event(f"Shot '{shot_d['title']}': critic scored {score:.0f}/10 (below {accept:g}), "
                                "rewriting the prompt and running another seed hunt", [])
        return {"shot_id": shot_id, "final_render_id": picked_id}

    async def _upscale(self, ctx: JobCtx, preview_id: int, start: float, span: float) -> int:
        """Stage 2 of a preview from its saved latents. Call inside pool.video_slot()."""
        proj = self.p()
        with proj.session() as s:
            r = s.get(m.Render, preview_id)
            wv = self.registry.get_key(r.workflow)
            pics = [proj.abspath(s.get(m.FileRec, s.get(m.RefVersion, v).file_id)) for v in r.slot_versions]
            lat = {k: proj.abspath(s.get(m.FileRec, fid)) for k, fid in
                   (("video_latent", r.latent_video_file_id), ("audio_latent", r.latent_audio_file_id))}
            shot_id, mode, prompt, params, ihash = r.shot_id, r.mode, r.prompt, dict(r.params), r.input_hash
            slots = [{"version_id": v} for v in r.slot_versions]
        self.queue.on_cancel[ctx.job_id] = self.clients[wv.manifest["server"]].interrupt
        try:
            res = await self.runner.render(wv, "stage2", params, mode=mode, pictures=pics, dest_dir=proj.media("renders"),
                                           tag=f"shot{shot_id}_final_{int(time.time())}", handoff=lat,
                                           on_progress=_comfy_progress(ctx, start, span))
        except WorkflowError as e:
            raise HarnessError(f"shot {shot_id}: {e}")
        finally:
            self.queue.on_cancel.pop(ctx.job_id, None)
        final_id = self._save_render(shot_id, "final", wv.key, mode, prompt, params, slots, res,
                                     parent=preview_id, ihash=ihash)
        bus.publish("render", id=final_id, shot_id=shot_id, stage="final")
        return final_id

    async def _await_user_pick(self, ctx, shot_id, previews, pick_idx, crit, wait_s) -> int:
        ev = asyncio.Event()
        chosen: list[int] = []
        self.picks[ctx.job_id] = (ev, chosen)
        bus.publish("seedhunt", job_id=ctx.job_id, shot_id=shot_id, renders=[p[0] for p in previews],
                    pick=previews[pick_idx][0], reason=crit.get("pick_reason", ""), wait_s=wait_s)
        with self.p().session() as s:
            atts = [{"type": "preview", "id": rid, "shot_id": shot_id, "job_id": ctx.job_id, "video": True,
                     "file_id": s.get(m.Render, rid).file_id, "picked": k == pick_idx, "label": f"preview {k + 1}"}
                    for k, (rid, _, _) in enumerate(previews)]
        why = crit.get("pick_reason", "")
        who = "Qwen picks" if crit else "Defaulted to (critic unavailable)"
        self.chat_event(f"Seed hunt: {who} preview {pick_idx + 1}" + (f" - {why}" if why else "")
                        + (f". Upscaling in {wait_s:.0f}s unless you pick another." if wait_s > 0
                           else ". Pick one to upscale."), atts)
        ctx.progress(0.68, f"Qwen picked preview {pick_idx + 1}; "
                     + (f"upscaling in {wait_s:.0f}s unless you choose another" if wait_s > 0 else "waiting for your pick"))
        try:
            if wait_s > 0:
                await asyncio.wait_for(ev.wait(), timeout=wait_s)
            else:
                while not ev.is_set():
                    ctx.check()
                    await asyncio.sleep(0.5)
        except asyncio.TimeoutError:
            pass
        finally:
            self.picks.pop(ctx.job_id, None)
        if chosen:
            ids = [p[0] for p in previews]
            if chosen[-1] in ids:
                return ids.index(chosen[-1])
        return pick_idx

    def _save_render(self, shot_id, stage, wf, mode, prompt, params, slots, res, parent=None, ihash="") -> int:
        """ihash: the input hash of what was actually rendered (the shot row may have been
        edited while the job ran)."""
        proj = self.p()
        with proj.session() as s:
            versions = [x["version_id"] for x in slots]
            r = m.Render(id=next_id(s, m.Render), shot_id=shot_id, stage=stage, parent_id=parent, seed=params.get("seed"),
                         workflow=wf,
                         mode=mode, prompt=prompt, params=_jsonable(params), slot_versions=versions,
                         input_hash=ihash or deps.input_hash(s.get(m.Shot, shot_id), versions, wf))
            s.add(r)
            s.flush()
            main = res.outputs.get("preview" if stage == "preview" else "video", [None])[0]
            if main:
                r.file_id = proj.register_file(s, main, "preview" if stage == "preview" else "video", "render", r.id).id
            for key, attr in (("video_latent", "latent_video_file_id"), ("audio_latent", "latent_audio_file_id")):
                if res.outputs.get(key):
                    setattr(r, attr, proj.register_file(s, res.outputs[key][0], "latent", "render", r.id).id)
            for p in res.leftovers:
                proj.register_file(s, p, "leftover", "render", r.id, location="comfy_output")
            s.add(r)
            return r.id

    def _previous_clip_note(self, shot_id: int) -> str | None:
        with self.p().session() as s:
            shot = s.get(m.Shot, shot_id)
            prev = s.exec(select(m.Shot).where(m.Shot.order < shot.order).order_by(m.Shot.order.desc())).first()
            if not prev:
                return None
            return f"shot '{prev.title}': {prev.description[:400]}"

    # ------------------------------------------------------------ final cut
    def final_cut(self, name: str | None = None, loudnorm: bool = True, upscale: bool = True) -> int:
        """upscale=False joins the timeline previews as they are (a quick watch-through)."""
        return self.queue.submit("final_cut", "video" if upscale else "local",
                                 lambda ctx: self._final_cut_job(ctx, name, loudnorm, upscale), {"upscale": upscale})

    def cut_sources(self) -> list[dict]:
        """Per shot in timeline order: the render to use now and, for a preview, the upscaled
        child to reuse or whether one must be made."""
        proj = self.p()
        out = []
        with proj.session() as s:
            for sh in s.exec(select(m.Shot).order_by(m.Shot.order)).all():
                r = s.get(m.Render, sh.final_render_id) if sh.final_render_id else None
                if not (r and r.file_id):
                    continue
                d = {"shot_id": sh.id, "title": sh.title, "render_id": r.id, "upscaled_id": None, "can_upscale": False}
                if r.stage == "final":
                    d["upscaled_id"] = r.id
                else:
                    child = s.exec(select(m.Render).where(m.Render.parent_id == r.id, m.Render.stage == "final",
                                                          m.Render.file_id.is_not(None)).order_by(m.Render.id.desc())).first()
                    if child and proj.abspath(s.get(m.FileRec, child.file_id)).is_file():
                        d["upscaled_id"] = child.id
                    lat = [s.get(m.FileRec, x) for x in (r.latent_video_file_id, r.latent_audio_file_id) if x]
                    d["can_upscale"] = len(lat) == 2 and all(proj.abspath(f).is_file() for f in lat)
                out.append(d)
        return out

    async def _final_cut_job(self, ctx: JobCtx, name: str | None, loudnorm: bool, upscale: bool = True) -> dict:
        proj = self.p()
        src = self.cut_sources()
        if not src:
            raise HarnessError("no shots have a clip in the timeline yet")
        if upscale:
            todo = [d for d in src if d["upscaled_id"] is None]
            missing = [d["title"] for d in todo if not d["can_upscale"]]
            if missing:
                self.chat_event("Final cut: no latents to upscale " + ", ".join(f"'{t}'" for t in missing)
                                + " - using the preview" + ("s" if len(missing) > 1 else ""), [])
            todo = [d for d in todo if d["can_upscale"]]
            if todo:
                async with self.pool.video_slot():
                    for k, d in enumerate(todo):
                        ctx.check()
                        ctx.progress(0.9 * k / len(todo), f"upscaling '{d['title']}' ({k + 1}/{len(todo)})")
                        d["upscaled_id"] = await self._upscale(ctx, d["render_id"], 0.9 * k / len(todo), 0.9 / len(todo))
                        bus.publish("shot", id=d["shot_id"])
        clips = []
        with proj.session() as s:
            for d in src:
                r = s.get(m.Render, (d["upscaled_id"] if upscale else None) or d["render_id"])
                clips.append(proj.abspath(s.get(m.FileRec, r.file_id)))
        if not upscale:
            name = (name or proj.name) + "_preview"
        biggest = max((media.probe(c) for c in clips), key=lambda i: i["width"] * i["height"])   # previews may be mixed in
        w, h = biggest["width"], biggest["height"]
        out = proj.root / "exports" / f"{re.sub(r'[^\w-]+', '_', name or proj.name)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y"]
        filters, labels = [], []
        for i, c in enumerate(clips):
            cmd += ["-i", str(c)]
        for i, c in enumerate(clips):
            info = media.probe(c)
            filters.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                           f"fps=48,format=yuv420p,setsar=1[v{i}]")
            if info["has_audio"]:
                filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
            else:
                filters.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{info['duration']:.3f}[a{i}]")
            labels.append(f"[v{i}][a{i}]")
        filters.append("".join(labels) + f"concat=n={len(clips)}:v=1:a=1[v][a0]")
        filters.append("[a0]loudnorm=I=-16:TP=-1.5:LRA=11[a]" if loudnorm else "[a0]anull[a]")
        cmd += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                "-preset", "medium", "-crf", "17", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)]
        ctx.progress(0.92 if upscale else 0.1, f"joining {len(clips)} clips")
        proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE, creationflags=NO_WINDOW)
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise HarnessError("ffmpeg failed: " + err.decode("utf-8", "replace")[-1500:])
        with proj.session() as s:
            f = proj.register_file(s, out, "export", "export")
            fid = f.id
        bus.publish("export", file_id=fid, path=str(out))
        return {"file_id": fid, "path": str(out), "clips": len(clips)}


def effective_mode(mode: str, roles: list[str]) -> str:
    """The mode the slot roles imply. A first/last-frame picture only acts as a frame when
    the mode's guide is on, so the roles win over a mismatched mode."""
    first, last = "first_frame" in roles, "last_frame" in roles
    if first and last:
        return "fl2va"
    if first:
        return "i2va"
    if last:
        return "l2va"
    if mode in ("i2va", "fl2va", "l2va"):
        return "ref2va" if roles else "t2va"
    if mode == "t2va" and roles:
        return "ref2va"
    return mode


def _mode_note(mode: str) -> str:
    """What a mode does to the pictures, for the user (modes only rewire frame guides)."""
    return {"i2va": "using <Picture 1> as the exact first frame",
            "l2va": "using <Picture 1> as the exact last frame",
            "fl2va": "using <Picture 1> as the exact first frame and <Picture 2> as the exact last frame",
            }.get(mode, "no first/last-frame picture any more")


def _comfy_progress(ctx: JobCtx, start: float, span: float):
    last = [0.0]

    def cb(e):
        if e["type"] != "progress" or not e.get("max"):
            return
        now = time.time()
        if now - last[0] < 0.5 and e["value"] != e["max"]:
            return
        last[0] = now
        ctx.progress(start + span * e["value"] / e["max"], None, node=e.get("node"), step=e["value"], steps=e["max"])
    return cb


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))
