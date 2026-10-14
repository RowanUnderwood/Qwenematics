"""Dependency questions the UI and Qwen ask before changing or deleting things.

- input_hash / stale_shots: has a shot changed since its final render?
- ref_impact: which shots and finished clips use a reference (any version)?
- replace_ref_version: point chosen shots at a new version (they become stale)
- *_delete_plan + apply_delete: cascade deletes that list every file before removing it
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from sqlmodel import Session, select

from . import models as m

# params that change the picture but are not inputs a user edits
_HASH_IGNORE = {"seed", "upscale_seed", "preview_seeds"}


def slot_versions(s: Session, shot_id: int) -> list[int]:
    slots = s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id).order_by(m.ShotSlot.slot)).all()
    return [x.ref_version_id for x in slots]


def input_hash(shot: m.Shot, versions: list[int], workflow: str) -> str:
    payload = {
        "workflow": workflow, "mode": shot.mode, "duration": round(shot.duration_s, 3),
        "prompt": shot.prompt.strip(),
        "params": {k: v for k, v in sorted((shot.params or {}).items()) if k not in _HASH_IGNORE},
        "slots": versions,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def shot_hash(s: Session, shot: m.Shot, workflow: str) -> str:
    return input_hash(shot, slot_versions(s, shot.id), workflow)


def is_stale(s: Session, shot: m.Shot, workflow: str) -> bool:
    if not shot.final_render_id:
        return False
    r = s.get(m.Render, shot.final_render_id)
    return r is not None and r.input_hash != shot_hash(s, shot, workflow)


@dataclass
class Impact:
    ref_id: int
    shots: list[dict] = field(default_factory=list)   # {shot_id, title, slot, version_id, has_final, final_uses_it}

    @property
    def rendered_count(self) -> int:
        return sum(1 for x in self.shots if x["final_uses_it"])


def ref_impact(s: Session, ref_id: int) -> Impact:
    """Shots that use this reference (any version, or pending), and whether a finished clip depends on it."""
    imp = Impact(ref_id)
    for slot in s.exec(select(m.ShotSlot).where(m.ShotSlot.ref_id == ref_id)).all():
        shot = s.get(m.Shot, slot.shot_id)
        final = s.get(m.Render, shot.final_render_id) if shot.final_render_id else None
        imp.shots.append({"shot_id": shot.id, "title": shot.title, "order": shot.order, "slot": slot.slot,
                          "version_id": slot.ref_version_id, "has_final": final is not None,
                          "final_uses_it": bool(final and slot.ref_version_id in (final.slot_versions or []))})
    imp.shots.sort(key=lambda x: x["order"])
    return imp


def replace_ref_version(s: Session, ref_id: int, new_version_id: int, shot_ids: list[int] | None = None) -> list[int]:
    """Point slots that use this reference at new_version_id (all shots, or only shot_ids).
    Shots not listed keep their old version - that is the 'fork' choice. Returns changed shot ids."""
    changed = []
    for slot in s.exec(select(m.ShotSlot).where(m.ShotSlot.ref_id == ref_id)).all():
        if shot_ids is not None and slot.shot_id not in shot_ids:
            continue
        if slot.ref_version_id != new_version_id:
            slot.ref_version_id = new_version_id
            s.add(slot)
            changed.append(slot.shot_id)
    ref = s.get(m.Reference, ref_id)
    ref.current_version_id = new_version_id
    s.add(ref)
    return sorted(set(changed))


# ---------------------------------------------------------------- deletes
@dataclass
class DeletePlan:
    rows: list = field(default_factory=list)          # ORM objects to delete
    files: list[m.FileRec] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)   # reasons a plain delete would break things
    summary: list[str] = field(default_factory=list)

    def merge(self, other: "DeletePlan"):
        self.rows += [r for r in other.rows if r not in self.rows]
        self.files += [f for f in other.files if f not in self.files]
        self.blocked_by += other.blocked_by
        self.summary += other.summary


def _files_owned(s: Session, owner_type: str, owner_id: int) -> list[m.FileRec]:
    return s.exec(select(m.FileRec).where(m.FileRec.owner_type == owner_type,
                                          m.FileRec.owner_id == owner_id)).all()


def render_delete_plan(s: Session, render_id: int) -> DeletePlan:
    r = s.get(m.Render, render_id)
    plan = DeletePlan(rows=[r], files=_files_owned(s, "render", r.id))
    plan.summary.append(f"render #{r.id} ({r.stage}, seed {r.seed})")
    for child in s.exec(select(m.Render).where(m.Render.parent_id == r.id)).all():
        plan.merge(render_delete_plan(s, child.id))
    return plan


def ref_version_delete_plan(s: Session, version_id: int, cascade: bool = False) -> DeletePlan:
    v = s.get(m.RefVersion, version_id)
    plan = DeletePlan(rows=[], files=_files_owned(s, "ref_version", v.id))
    plan.summary.append(f"reference version #{v.id} (v{v.n})")
    slots = s.exec(select(m.ShotSlot).where(m.ShotSlot.ref_version_id == v.id)).all()
    for slot in slots:
        shot = s.get(m.Shot, slot.shot_id)
        if cascade:
            plan.rows.append(slot)
            plan.summary.append(f"  removes it from shot '{shot.title or shot.id}' slot {slot.slot}")
        else:
            plan.blocked_by.append(f"used by shot '{shot.title or shot.id}' slot {slot.slot}")
    plan.rows.append(v)
    return plan


def ref_delete_plan(s: Session, ref_id: int, cascade: bool = False) -> DeletePlan:
    ref = s.get(m.Reference, ref_id)
    plan = DeletePlan()
    for v in s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref_id)).all():
        plan.merge(ref_version_delete_plan(s, v.id, cascade))
    for slot in s.exec(select(m.ShotSlot).where(m.ShotSlot.ref_id == ref_id, m.ShotSlot.ref_version_id.is_(None))).all():
        shot = s.get(m.Shot, slot.shot_id)
        if cascade:
            plan.rows.append(slot)
        else:
            plan.blocked_by.append(f"waiting in shot '{shot.title or shot.id}' slot {slot.slot}")
    plan.rows.append(ref)
    plan.summary.insert(0, f"reference '{ref.name}'")
    return plan


def shot_delete_plan(s: Session, shot_id: int, orphan_refs: bool = False) -> DeletePlan:
    """Delete a shot with its slots and renders. With orphan_refs, also delete reference
    versions no other shot uses."""
    shot = s.get(m.Shot, shot_id)
    plan = DeletePlan(rows=[], files=_files_owned(s, "shot", shot.id))
    plan.summary.append(f"shot '{shot.title or shot.id}'")
    for r in s.exec(select(m.Render).where(m.Render.shot_id == shot_id, m.Render.parent_id.is_(None))).all():
        plan.merge(render_delete_plan(s, r.id))
    slots = s.exec(select(m.ShotSlot).where(m.ShotSlot.shot_id == shot_id)).all()
    plan.rows += slots
    if orphan_refs:
        for slot in slots:
            if slot.ref_version_id is None:
                continue
            others = s.exec(select(m.ShotSlot).where(m.ShotSlot.ref_version_id == slot.ref_version_id,
                                                      m.ShotSlot.shot_id != shot_id)).all()
            if not others:
                plan.merge(ref_version_delete_plan(s, slot.ref_version_id, cascade=True))
    plan.rows.append(shot)
    return plan


def orphan_plan(s: Session) -> DeletePlan:
    """Clean-up: unpicked previews of finished shots, their latents, and unreferenced
    reference versions that are not the current version of their reference."""
    plan = DeletePlan()
    for shot in s.exec(select(m.Shot).where(m.Shot.final_render_id.is_not(None))).all():
        final = s.get(m.Render, shot.final_render_id)
        for r in s.exec(select(m.Render).where(m.Render.shot_id == shot.id, m.Render.stage == "preview")).all():
            if final and r.id == final.parent_id:
                plan.files += [f for f in _files_owned(s, "render", r.id) if f.kind == "latent"]
                continue
            plan.merge(render_delete_plan(s, r.id))
    used = {x.ref_version_id for x in s.exec(select(m.ShotSlot)).all() if x.ref_version_id}
    current = {r.current_version_id for r in s.exec(select(m.Reference)).all()}
    for v in s.exec(select(m.RefVersion)).all():
        if v.id not in used and v.id not in current:
            plan.merge(ref_version_delete_plan(s, v.id))
    return plan


def apply_delete(project, s: Session, plan: DeletePlan, force: bool = False) -> list[str]:
    if plan.blocked_by and not force:
        raise ValueError("blocked: " + "; ".join(plan.blocked_by))
    for row in plan.rows:
        if isinstance(row, m.Render):
            for shot in s.exec(select(m.Shot).where(m.Shot.final_render_id == row.id)).all():
                shot.final_render_id = None
                s.add(shot)
        if isinstance(row, m.RefVersion):
            ref = s.get(m.Reference, row.ref_id)
            if ref and ref.current_version_id == row.id:
                ref.current_version_id = None
                s.add(ref)
        s.delete(row)
        s.flush()
    # rows first: ref versions and renders point at their files
    return project.delete_files(s, plan.files)
