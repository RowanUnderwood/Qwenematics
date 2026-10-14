"""The director's tools: JSON schemas for Qwen plus handlers that call the Harness.

Results are short JSON so they don't flood the context. Media is never returned in a
tool result; `view` queues an attachment for the next model call instead.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from sqlmodel import select

from ..db import deps
from ..db import models as m

KINDS = ["character", "character_closeup", "location", "establishing", "prop", "first_frame", "concept"]
ROLES_ = ["subject", "first_frame", "last_frame"]
MODES = ["ref2va", "i2va", "fl2va", "l2va", "t2va"]
BIBLE_KINDS = ["plot", "character", "location", "style", "audio", "dialogue", "note"]
SHOT_FIELDS = {"title", "description", "dialogue", "audio_notes", "mode", "duration_s"}


def _fn(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": required or []}}}


TOOLS = [
    _fn("get_bible", "Read the full project bible (plot, characters, locations, style, audio notes).", {}),
    _fn("update_bible", "Create or replace one bible entry, matched by kind+title.", {
        "kind": {"type": "string", "enum": BIBLE_KINDS}, "title": {"type": "string"},
        "body": {"type": "string", "description": "The full text of the entry (replaces the old body)."}},
        ["kind", "title", "body"]),
    _fn("search_references", "Find existing reference images to REUSE. Always call before create_reference.", {
        "text": {"type": "string", "description": "Words describing what you need, e.g. 'old fisherman full body'."},
        "kind": {"type": "string", "enum": KINDS}}, ["text"]),
    _fn("get_reference", "Details of one reference: versions, captions, which shots use it.", {
        "ref_id": {"type": "integer"}}, ["ref_id"]),
    _fn("create_reference", "Design a NEW reference image and queue its render (Krea 2).", {
        "name": {"type": "string", "description": "Short unique name, e.g. 'Mara - full body'."},
        "kind": {"type": "string", "enum": KINDS},
        "description": {"type": "string", "description": "Everything it must show: identity, clothing, colours, setting, pose, framing."},
        "tags": {"type": "array", "items": {"type": "string"}},
        "reuse_checked": {"type": "boolean", "description": "true only after search_references found nothing reusable."}},
        ["name", "kind", "description", "reuse_checked"]),
    _fn("revise_reference", "Make a new version of a reference with a change (keeps the old version).", {
        "ref_id": {"type": "integer"}, "change": {"type": "string"}}, ["ref_id", "change"]),
    _fn("list_shots", "The shot list in order with status.", {}),
    _fn("get_shot", "Full details of one shot, including its pictures and current video prompt.", {
        "shot_id": {"type": "integer"}}, ["shot_id"]),
    _fn("upsert_shot", "Create a shot (omit shot_id) or update fields of an existing one.", {
        "shot_id": {"type": "integer"}, "title": {"type": "string"},
        "description": {"type": "string", "description": "What happens on screen, in order, in plain words."},
        "dialogue": {"type": "string", "description": "Spoken lines, verbatim, with who says them."},
        "audio_notes": {"type": "string"}, "mode": {"type": "string", "enum": MODES},
        "duration_s": {"type": "number", "description": "2-15 seconds"},
        "after_shot_id": {"type": "integer", "description": "Insert right after this shot (new shots only)."}}),
    _fn("assign_ref", "Put a reference into a shot. role: subject (default), first_frame or last_frame "
                       "(at most one first_frame and one last_frame per shot).", {
        "shot_id": {"type": "integer"}, "ref_id": {"type": "integer"},
        "role": {"type": "string", "enum": ROLES_}}, ["shot_id", "ref_id"]),
    _fn("unassign_ref", "Remove a reference from a shot.", {
        "shot_id": {"type": "integer"}, "ref_id": {"type": "integer"}}, ["shot_id", "ref_id"]),
    _fn("render_shot", "Queue the video render of a shot (writes its prompt, seed-hunts; the upscale happens at the final cut).", {
        "shot_id": {"type": "integer"},
        "change": {"type": "string", "description": "Optional change to make versus the last render."},
        "rewrite_prompt": {"type": "boolean"}}, ["shot_id"]),
    _fn("view", "Look at an item yourself. It is attached to your next turn only.", {
        "item": {"type": "string", "description": "'ref:<ref_id>' for a reference's current image, or 'shot:<shot_id>' for a shot's final clip."}},
        ["item"]),
    _fn("job_status", "Status of queued and running render jobs.", {}),
    _fn("ask_user", "Ask the user a question and stop until they answer.", {
        "question": {"type": "string"}}, ["question"]),
]


class ToolBox:
    """Executes tool calls for one agent turn. `pending_media` collects view() requests."""

    def __init__(self, harness):
        self.h = harness
        self.pending_media: list[str] = []
        self.asked: str | None = None
        self.handlers: dict[str, Callable[..., Awaitable[Any] | Any]] = {
            t["function"]["name"]: getattr(self, "t_" + t["function"]["name"]) for t in TOOLS}

    async def run(self, name: str, args: dict) -> str:
        fn = self.handlers.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            out = fn(**args)
            if hasattr(out, "__await__"):
                out = await out
            return json.dumps(out, default=str)[:6000]
        except TypeError as e:
            return json.dumps({"error": f"bad arguments for {name}: {e}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # ---------------------------------------------------------------- bible
    def t_get_bible(self):
        return {"bible": self.h.bible_text(20000) or "(empty)"}

    def t_update_bible(self, kind, title, body):
        e = self.h.update_bible(kind, title, body)
        return {"ok": True, "entry_id": e.id}

    # ---------------------------------------------------------------- references
    def t_search_references(self, text, kind=None):
        hits = self.h.search_references(text, kind)
        return {"matches": [{k: r[k] for k in ("ref_id", "name", "kind", "caption", "description", "used_in_shots", "version_n")}
                            for r in hits]} if hits else {"matches": [], "note": "nothing similar exists"}

    def t_get_reference(self, ref_id):
        with self.h.p().session() as s:
            ref = s.get(m.Reference, ref_id)
            if not ref:
                return {"error": f"no reference {ref_id}"}
            out = self.h.ref_summary(s, ref)
            out["versions"] = [{"version_id": v.id, "n": v.n, "caption": v.caption, "score": (v.critic or {}).get("score")}
                               for v in s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref_id)).all()]
        return out

    def t_create_reference(self, name, kind, description, tags=None, reuse_checked=False):
        if not reuse_checked:
            hits = self.h.search_references(f"{name} {description}", kind)
            return {"not_created": True, "reason": "check for reusable references first",
                    "candidates": [{k: r[k] for k in ("ref_id", "name", "kind", "caption")} for r in hits]}
        ref_id, job = self.h.create_reference(name, kind, description, tags)
        return {"ref_id": ref_id, "job_id": job, "status": "rendering - it can be assigned to shots once it finishes"}

    def t_revise_reference(self, ref_id, change):
        job = self.h.revise_reference(ref_id, change)
        with self.h.p().session() as s:
            imp = deps.ref_impact(s, ref_id)
        out = {"job_id": job, "status": "rendering a new version"}
        if imp.shots:
            out["warning"] = (f"used in {len(imp.shots)} shot(s) ({imp.rendered_count} with finished clips). "
                              "The user will choose which shots switch to the new version.")
        return out

    # ---------------------------------------------------------------- shots
    def t_list_shots(self):
        wf = self.h.registry.get("minimax_h3").key
        jobs = self.h.active_shot_jobs()
        with self.h.p().session() as s:
            rows = s.exec(select(m.Shot).order_by(m.Shot.order)).all()
            return {"shots": [{"shot_id": x.id, "title": x.title, "mode": x.mode, "duration_s": x.duration_s,
                               "status": ("rendering" if jobs[x.id].state == "running" else "queued") if x.id in jobs else x.status,
                               "pictures": len(deps.slot_versions(s, x.id)),
                               "stale": x.id not in jobs and deps.is_stale(s, x, x.workflow or wf)} for x in rows]}

    def t_get_shot(self, shot_id):
        with self.h.p().session() as s:
            x = s.get(m.Shot, shot_id)
            if not x:
                return {"error": f"no shot {shot_id}"}
            d = {k: getattr(x, k) for k in ("id", "title", "description", "dialogue", "audio_notes", "mode",
                                             "duration_s", "status", "final_render_id")}
            d["prompt"] = x.prompt[:3000]
        d["pictures"] = [{k: v for k, v in sl.items() if k in ("slot", "role", "ref_id", "name", "kind", "caption", "outdated")}
                         for sl in self.h.shot_slots(shot_id)]
        return d

    def _busy_note(self, shot_id) -> dict:
        j = self.h.active_shot_jobs().get(shot_id)
        return {"warning": f"shot {shot_id} is {'rendering' if j.state == 'running' else 'queued'} (job {j.id}); "
                           "this change makes that render stale"} if j else {}

    def t_upsert_shot(self, shot_id=None, after_shot_id=None, **fields):
        # only the schema's fields: prompt, status, params... are not Qwen's to set here
        ignored = sorted(k for k in fields if k not in SHOT_FIELDS)
        fields = {k: v for k, v in fields.items() if k in SHOT_FIELDS}
        with self.h.p().session() as s:
            if shot_id is not None and s.get(m.Shot, shot_id) is None:
                return {"error": f"no shot {shot_id}"}
            if shot_id is None:
                title = str(fields.get("title") or "").strip().lower()
                dup = title and next((x for x in s.exec(select(m.Shot)).all() if x.title.strip().lower() == title), None)
                if dup:
                    return {"error": f"shot {dup.id} already has the title '{dup.title}'. Pass shot_id={dup.id} "
                                     "to update it, or choose another title for a new shot."}
                if after_shot_id is not None:
                    a = s.get(m.Shot, after_shot_id)
                    if a is None:
                        return {"error": f"no shot {after_shot_id} to insert after"}
                    nxt = s.exec(select(m.Shot).where(m.Shot.order > a.order).order_by(m.Shot.order)).first()
                    fields["order"] = (a.order + nxt.order) / 2 if nxt else a.order + 1
        sid = self.h.upsert_shot(shot_id, **fields)
        return {"shot_id": sid, **({"ignored": ignored} if ignored else {}),
                **(self._busy_note(shot_id) if shot_id is not None else {})}

    def t_assign_ref(self, shot_id, ref_id, role="subject"):
        slots = self.h.assign_ref(shot_id, ref_id=ref_id, role=role)
        return {"pictures": [{"slot": x["slot"], "role": x["role"], "name": x["name"],
                              **({"pending": True} if x["pending"] else {})} for x in slots], **self._busy_note(shot_id)}

    def t_unassign_ref(self, shot_id, ref_id):
        for x in self.h.shot_slots(shot_id):
            if x["ref_id"] == ref_id:
                self.h.remove_slot(shot_id, x["slot_id"])
                return {"ok": True, **self._busy_note(shot_id)}
        return {"error": "that reference is not in the shot"}

    def t_render_shot(self, shot_id, change=None, rewrite_prompt=False):
        slots = self.h.shot_slots(shot_id)
        with self.h.p().session() as s:
            shot = s.get(m.Shot, shot_id)
            mode = shot.mode
        if mode != "t2va" and not slots:
            return {"error": f"mode {mode} needs pictures; assign references first or use t2va"}
        pending = [x["name"] for x in slots if x["pending"]]
        job = self.h.render_shot(shot_id, rewrite_prompt=rewrite_prompt, change=change)
        out = {"job_id": job, "status": "queued on the video GPU"}
        if pending:
            out["note"] = f"it will start once these references finish rendering: {', '.join(pending)}"
        return out

    # ---------------------------------------------------------------- misc
    def t_view(self, item):
        self.pending_media.append(item)
        return {"ok": True, "note": f"{item} will be attached to your next turn"}

    def t_job_status(self):
        with self.h.p().session() as s:
            jobs = s.exec(select(m.Job).where(m.Job.state.in_(["queued", "running"])).order_by(m.Job.id)).all()
            recent = s.exec(select(m.Job).where(m.Job.state.in_(["done", "failed"])).order_by(m.Job.id.desc()).limit(5)).all()
        fmt = lambda j: {"job_id": j.id, "kind": j.kind, "state": j.state, "progress": round(j.progress, 2),
                         "message": j.message[:200], "payload": j.payload}
        return {"active": [fmt(j) for j in jobs], "recent": [fmt(j) for j in recent]}

    def t_ask_user(self, question):
        self.asked = question
        return {"ok": True, "note": "The question is shown to the user. End your turn now."}
