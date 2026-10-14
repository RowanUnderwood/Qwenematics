"""HTTP + websocket API for the GUI.   Run:  python -m harness.server  (then open http://127.0.0.1:8765)"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from PIL import Image
from pydantic import BaseModel
from sqlmodel import select

from . import procs, styles
from .config import ROOT, load_state, save_state
from .core import DEFAULT_PROJECT_SETTINGS, Harness, HarnessError
from .db import deps
from .db import models as m
from .db.project import Project, reveal_in_explorer
from .events import bus
from .llm.agent import Director

log = logging.getLogger("harness")
H: Harness
D: Director
SERVER = None          # the uvicorn.Server, so /api/shutdown can stop it
PORT = 8765


@asynccontextmanager
async def lifespan(app: FastAPI):
    global H, D
    H = Harness()
    D = Director(H)
    async def first_refresh():
        try:
            await H.refresh_servers(full=True)      # ~10 s per ComfyUI: don't hold up the UI
            bus.publish("servers")
        except Exception as e:
            log.warning("server refresh failed: %s", e)
    asyncio.create_task(first_refresh())
    last = _state().get("last_project")
    if last and Path(last, "project.db").is_file():
        try:
            await H.open_project(Path(last))
        except Exception as e:
            log.warning("could not reopen %s: %s", last, e)
    yield
    await H.close_project()


app = FastAPI(title="Qwenematics", lifespan=lifespan)


@app.exception_handler(HarnessError)
async def _harness_error(_, exc: HarnessError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse({"error": str(exc)}, status_code=400)


def proj() -> Project:
    if H.project is None:
        raise HTTPException(409, "no project is open")
    return H.project


# ------------------------------------------------------------------ events
@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q = bus.subscribe()
    try:
        while True:
            msg = await q.get()
            await sock.send_text(bus.encode(msg))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        bus.unsubscribe(q)


# ------------------------------------------------------------------ status / servers
@app.get("/api/ping")
async def ping():
    return {"app": "qwenematics"}


@app.get("/api/status")
async def status(full: bool = False):
    return {"servers": await H.refresh_servers(full), "workflows": H.workflows_status(),
            "gpu_pool": {**H.pool.status(), "krea_on_video_problems": H.krea_on_video[:5]},
            "project": {"name": H.project.name, "path": str(H.project.root)} if H.project else None,
            "director_busy": D.busy}


@app.post("/api/servers/{name}/launch")
async def launch(name: str):
    s = H.settings
    path = Path(s.llm["launcher"]) if name == "llm" else s.server_launcher(name)
    if not path or not path.exists():
        raise HTTPException(404, f"no launcher for {name}")
    subprocess.Popen(["cmd.exe", "/c", "start", "", str(path)], cwd=str(path.parent))
    return {"ok": True, "launcher": str(path)}


class ShutdownReq(BaseModel):
    comfy: bool = False          # also stop the harness's ComfyUI instances (image + video)
    llm: bool = False            # also stop Ninfer (Qwen)


@app.post("/api/shutdown")
async def shutdown(req: ShutdownReq):
    """Stop jobs cleanly, optionally stop the GPU servers, then exit and free the port."""
    async def run():
        done = []
        try:
            await H.close_project()                     # interrupts running renders, saves job states
            done.append("project closed")
            s = H.settings
            if req.comfy:
                for name, cfg in s.servers.items():
                    port = httpx.URL(cfg["url"]).port
                    done.append(await asyncio.to_thread(procs.stop_port, port, s.server_launcher(name)))
            if req.llm and s.llm.get("stopper"):
                await asyncio.to_thread(procs.run_bat, Path(s.llm["stopper"]))
                if s.llm.get("launcher"):
                    await asyncio.to_thread(procs.close_windows, Path(s.llm["launcher"]))
                done.append("Ninfer stopped")
        except Exception as e:
            log.exception("shutdown step failed")
            done.append(f"error: {e}")
        log.info("shutdown: %s", "; ".join(done))
        bus.publish("shutdown", steps=done)
        await asyncio.sleep(0.5)                        # let the event reach the browser
        if SERVER is not None:
            SERVER.should_exit = True
    asyncio.create_task(run())
    return {"ok": True}


class GpuSharingIn(BaseModel):
    image_on_video_gpu: str | None = None      # auto | off
    min_backlog: int | None = None
    image_jobs_in_flight: int | None = None


@app.get("/api/gpu_sharing")
async def get_gpu_sharing():
    return {**H.settings.gpu_sharing, "pool": H.pool.status(), "krea_on_video_problems": H.krea_on_video[:5]}


@app.put("/api/gpu_sharing")
async def put_gpu_sharing(body: GpuSharingIn):
    """GUI overrides of settings.yaml gpu_sharing, kept in the harness state file."""
    if body.image_on_video_gpu not in (None, "auto", "off"):
        raise HTTPException(400, "image_on_video_gpu must be auto or off")
    cur = load_state(H.settings.projects_root).get("gpu_sharing") or {}
    cur.update({k: v for k, v in body.model_dump().items() if v is not None})
    save_state(H.settings.projects_root, gpu_sharing=cur)
    H.pool._dispatch()                          # a newly allowed borrow can start now
    bus.publish("servers")
    return await get_gpu_sharing()


@app.post("/api/settings/restore_defaults")
async def restore_defaults():
    """Reset what the Settings tab controls: GPU sharing (back to settings.yaml) and, if a
    project is open, its video aspect, iterate and parameter overrides. The locked style
    (Bible tab), the bible, shots and renders are untouched."""
    save_state(H.settings.projects_root, gpu_sharing=None)
    H.pool._dispatch()
    if H.project:
        keep = {"krea_style": H.project_settings().get("krea_style")}
        H.set_project_settings({**json.loads(json.dumps(DEFAULT_PROJECT_SETTINGS)), **keep})
    bus.publish("servers")
    return {"gpu_sharing": H.settings.gpu_sharing, "project": H.project_settings() if H.project else None}


@app.get("/api/workflows")
async def workflows():
    out = []
    for wf_id, vs in H.registry.versions.items():
        for v, wv in vs.items():
            oi = H.object_info.get(wv.manifest.get("server"))
            params = {}
            for k, spec in wv.params.items():
                if spec.get("type") == "text":
                    continue
                d = {kk: vv for kk, vv in spec.items() if kk not in ("targets", "choices_from")}
                ch = wv.choices(k, oi)
                if spec.get("type") == "lora" and oi:
                    ch = wv.choices(k, oi) or _lora_choices(wv, k, oi)
                if ch:
                    d["choices"] = ch
                params[k] = d
            out.append({"key": wv.key, "id": wf_id, "version": v, "label": wv.manifest.get("label"),
                        "usable": wv.usable, "problems": wv.problems, "params": params,
                        "modes": {k: mm.get("label", k) for k, mm in wv.modes.items()}})
    return out


def _lora_choices(wv, name, oi):
    t = wv.params[name]["targets"][0]
    for g in wv.graphs.values():
        node = g.get(t["node"])
        if node:
            spec = oi.get(node["class_type"], {}).get("input", {}).get("required", {}).get(t["input"])
            if spec and isinstance(spec[0], list):
                return spec[0]
    return None


# ------------------------------------------------------------------ projects
class NameIn(BaseModel):
    name: str


class PathIn(BaseModel):
    path: str


@app.get("/api/projects")
async def list_projects():
    return Project.list(H.settings.projects_root)


def _state() -> dict:
    return load_state(H.settings.projects_root)


def _remember(p: Project | None):
    save_state(H.settings.projects_root, last_project=str(p.root) if p else None)


@app.post("/api/projects")
async def create_project(body: NameIn):
    p = await H.create_project(body.name)
    _remember(p)
    bus.publish("project", name=p.name)
    return {"name": p.name, "path": str(p.root)}


@app.post("/api/projects/open")
async def open_project(body: PathIn):
    p = await H.open_project(Path(body.path))
    _remember(p)
    bus.publish("project", name=p.name)
    return {"name": p.name, "path": str(p.root)}


@app.post("/api/project/snapshot")
async def snapshot(body: NameIn):
    dest = proj().snapshot(H.settings.projects_root, body.name)
    return {"path": str(dest)}


@app.post("/api/project/archive")
async def archive():
    p = proj()
    z = p.archive(p.root.parent / f"{p.root.name}_archive.zip")
    return {"path": str(z)}


@app.get("/api/project/delete_plan")
async def project_delete_plan():
    return proj().delete_plan()


@app.delete("/api/project")
async def delete_project():
    p = proj()
    await H.queue.stop()
    H.queue = None
    failed = p.delete_everything()
    H.project = None
    _remember(None)
    bus.publish("project", name=None)
    return {"failed": failed}


@app.get("/api/project/settings")
async def get_psettings():
    return H.project_settings()


@app.put("/api/project/settings")
async def put_psettings(cfg: dict[str, Any]):
    H.set_project_settings(cfg)
    return H.project_settings()


# ------------------------------------------------------------------ styles
@app.get("/api/styles")
async def list_styles():
    return {"styles": styles.catalog(), "bases": styles.bases(),
            "current": H.project_settings().get("krea_style") if H.project else None}


@app.get("/api/styles/{base}/{style_id}/image")
async def style_image(base: str, style_id: str, size: int | None = None):
    try:
        path = await styles.image(base, style_id, size)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except httpx.HTTPError as e:
        raise HTTPException(502, f"could not fetch the sample image: {e}")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=604800"})


class StyleReq(BaseModel):
    name: str | None = None      # None clears the lock


@app.put("/api/project/style")
async def set_style(req: StyleReq):
    if req.name and not styles.get(req.name):
        raise HTTPException(404, f"no style named {req.name!r}")
    cfg = H.project_settings()
    cfg["krea_style"] = styles.get(req.name)["name"] if req.name else None
    H.set_project_settings(cfg)
    H.chat_event(f"Project style locked to '{cfg['krea_style']}' - new images use it; existing references keep their look "
                 "until regenerated." if cfg["krea_style"] else "Project style unlocked.", [])
    return {"current": cfg["krea_style"]}


# ------------------------------------------------------------------ bible
class BibleIn(BaseModel):
    kind: str
    title: str
    body: str
    id: int | None = None


@app.get("/api/bible")
async def bible():
    with proj().session() as s:
        return s.exec(select(m.BibleEntry).order_by(m.BibleEntry.kind, m.BibleEntry.id)).all()


@app.put("/api/bible")
async def put_bible(b: BibleIn):
    return H.update_bible(b.kind, b.title, b.body, b.id)


@app.delete("/api/bible/{entry_id}")
async def del_bible(entry_id: int):
    with proj().session() as s:
        e = s.get(m.BibleEntry, entry_id)
        if e:
            s.delete(e)
    bus.publish("bible", id=entry_id)
    return {"ok": True}


# ------------------------------------------------------------------ references
@app.get("/api/references")
async def references():
    out = []
    with proj().session() as s:
        for ref in s.exec(select(m.Reference).order_by(m.Reference.id)).all():
            d = H.ref_summary(s, ref)
            d["versions"] = [{"id": v.id, "n": v.n, "file_id": v.file_id, "caption": v.caption, "prompt": v.prompt,
                              "seed": v.seed, "workflow": v.workflow, "source": v.source,
                              "score": (v.critic or {}).get("score"), "critic": v.critic, "created": v.created}
                             for v in s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref.id)
                                             .order_by(m.RefVersion.n)).all()]
            out.append(d)
    return out


class RefIn(BaseModel):
    name: str
    kind: str
    description: str
    tags: list[str] = []
    enhance: bool = True                   # off: description is the exact Krea prompt


@app.post("/api/references")
async def create_ref(r: RefIn):
    ref_id, job = H.create_reference(r.name, r.kind, r.description, r.tags, enhance=r.enhance)
    return {"ref_id": ref_id, "job_id": job}


class ReviseIn(BaseModel):
    change: str | None = None
    prompt: str | None = None              # an edited Krea prompt
    enhance: bool = True


@app.post("/api/references/{ref_id}/revise")
async def revise_ref(ref_id: int, c: ReviseIn):
    return {"job_id": H.revise_reference(ref_id, c.change, c.prompt, c.enhance)}


@app.post("/api/references/upload")
async def upload_ref(file: UploadFile = File(...), name: str = Form(...), kind: str = Form("character"),
                     description: str = Form(""), ref_id: int | None = Form(None)):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / Path(file.filename or "upload.png").name
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        vid = H.import_reference_file(tmp, name, kind, description, ref_id)
    return {"version_id": vid}


@app.get("/api/references/{ref_id}/impact")
async def ref_impact(ref_id: int):
    with proj().session() as s:
        imp = deps.ref_impact(s, ref_id)
    return {"shots": imp.shots, "rendered": imp.rendered_count}


class ApplyIn(BaseModel):
    version_id: int
    shot_ids: list[int] | None = None


@app.post("/api/references/{ref_id}/apply_version")
async def apply_version(ref_id: int, a: ApplyIn):
    return {"changed_shots": H.apply_ref_version(ref_id, a.version_id, a.shot_ids)}


def _plan_json(plan: deps.DeletePlan, p: Project) -> dict:
    return {"summary": plan.summary, "blocked_by": plan.blocked_by,
            "files": [str(p.abspath(f)) for f in plan.files],
            "bytes": sum(f.bytes for f in plan.files)}


@app.get("/api/references/{ref_id}/delete_plan")
async def ref_delete_plan(ref_id: int, cascade: bool = False):
    p = proj()
    with p.session() as s:
        return _plan_json(deps.ref_delete_plan(s, ref_id, cascade), p)


@app.delete("/api/references/{ref_id}")
async def delete_ref(ref_id: int, cascade: bool = False):
    p = proj()
    with p.session() as s:
        failed = deps.apply_delete(p, s, deps.ref_delete_plan(s, ref_id, cascade))
    bus.publish("reference", id=ref_id, deleted=True)
    bus.publish("shot", id=None)
    return {"failed": failed}


@app.delete("/api/ref_versions/{version_id}")
async def delete_ref_version(version_id: int, cascade: bool = False):
    p = proj()
    with p.session() as s:
        v = s.get(m.RefVersion, version_id)
        ref_id = v.ref_id if v else None
        failed = deps.apply_delete(p, s, deps.ref_version_delete_plan(s, version_id, cascade))
        ref = s.get(m.Reference, ref_id) if ref_id else None
        if ref and ref.current_version_id is None:
            last = s.exec(select(m.RefVersion).where(m.RefVersion.ref_id == ref_id).order_by(m.RefVersion.n.desc())).first()
            ref.current_version_id = last.id if last else None
            s.add(ref)
    bus.publish("reference", id=ref_id)
    return {"failed": failed}


@app.post("/api/references/{ref_id}/current/{version_id}")
async def set_current(ref_id: int, version_id: int):
    with proj().session() as s:
        ref = s.get(m.Reference, ref_id)
        ref.current_version_id = version_id
        s.add(ref)
    bus.publish("reference", id=ref_id)
    return {"ok": True}


# ------------------------------------------------------------------ shots
def _shot_json(s, x: m.Shot, wf: str, jobs: dict[int, m.Job]) -> dict:
    d = x.model_dump()
    d["slots"] = H.shot_slots(x.id)
    j = jobs.get(x.id)
    d["job"] = {"id": j.id, "state": j.state, "progress": j.progress, "message": j.message} if j else None
    d["stale"] = not j and deps.is_stale(s, x, x.workflow or wf)    # a running job replaces the clip anyway
    d["renders"] = [{"id": r.id, "stage": r.stage, "seed": r.seed, "file_id": r.file_id, "picked": r.picked,
                     "parent_id": r.parent_id, "score": (r.critic or {}).get("score"), "critic": r.critic,
                     "created": r.created, "prompt": r.prompt}
                    for r in s.exec(select(m.Render).where(m.Render.shot_id == x.id).order_by(m.Render.id)).all()]
    return d


@app.get("/api/shots")
async def shots():
    wf = H.registry.get("minimax_h3").key
    jobs = H.active_shot_jobs()
    with proj().session() as s:
        return [_shot_json(s, x, wf, jobs) for x in s.exec(select(m.Shot).order_by(m.Shot.order)).all()]


@app.post("/api/shots")
async def create_shot(fields: dict[str, Any]):
    return {"shot_id": H.upsert_shot(None, **fields)}


@app.patch("/api/shots/{shot_id}")
async def update_shot(shot_id: int, fields: dict[str, Any]):
    return {"shot_id": H.upsert_shot(shot_id, **fields)}


@app.post("/api/shots/reorder")
async def reorder(ids: list[int]):
    with proj().session() as s:
        for i, sid in enumerate(ids, 1):
            x = s.get(m.Shot, sid)
            if x:
                x.order = float(i)
                s.add(x)
    bus.publish("shot", id=None)
    return {"ok": True}


class SlotIn(BaseModel):
    ref_id: int | None = None
    version_id: int | None = None
    role: str = "subject"


@app.post("/api/shots/{shot_id}/slots")
async def add_slot(shot_id: int, sl: SlotIn):
    return H.assign_ref(shot_id, ref_id=sl.ref_id, version_id=sl.version_id, role=sl.role)


@app.delete("/api/shots/{shot_id}/slots/{slot_id}")
async def del_slot(shot_id: int, slot_id: int):
    H.remove_slot(shot_id, slot_id)
    return {"ok": True}


class RenderIn(BaseModel):
    change: str | None = None
    rewrite_prompt: bool = False
    seeds: int | None = None
    enhance: bool = True                   # off: the saved prompt goes through verbatim


@app.post("/api/shots/{shot_id}/render")
async def render(shot_id: int, r: RenderIn):
    return {"job_id": H.render_shot(shot_id, r.rewrite_prompt, r.change, r.seeds, r.enhance)}


@app.post("/api/shots/{shot_id}/final/{render_id}")
async def set_final(shot_id: int, render_id: int):
    with proj().session() as s:
        x, r = s.get(m.Shot, shot_id), s.get(m.Render, render_id)
        if x is None or r is None or r.shot_id != shot_id:
            raise HarnessError(f"render {render_id} does not belong to shot {shot_id}")
        x.final_render_id = render_id
        s.add(x)
    bus.publish("shot", id=shot_id)
    return {"ok": True}


@app.get("/api/shots/{shot_id}/delete_plan")
async def shot_delete_plan(shot_id: int, orphan_refs: bool = False):
    p = proj()
    with p.session() as s:
        return _plan_json(deps.shot_delete_plan(s, shot_id, orphan_refs), p)


@app.delete("/api/shots/{shot_id}")
async def delete_shot(shot_id: int, orphan_refs: bool = False):
    p = proj()
    with p.session() as s:
        failed = deps.apply_delete(p, s, deps.shot_delete_plan(s, shot_id, orphan_refs), force=True)
    bus.publish("shot", id=shot_id, deleted=True)
    return {"failed": failed}


@app.delete("/api/renders/{render_id}")
async def delete_render(render_id: int):
    p = proj()
    with p.session() as s:
        r = s.get(m.Render, render_id)
        sid = r.shot_id if r else None
        failed = deps.apply_delete(p, s, deps.render_delete_plan(s, render_id))
    bus.publish("shot", id=sid)
    return {"failed": failed}


# ------------------------------------------------------------------ jobs
@app.get("/api/jobs")
async def jobs(limit: int = 50):
    with proj().session() as s:
        return s.exec(select(m.Job).order_by(m.Job.id.desc()).limit(limit)).all()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: int):
    await H.queue.cancel(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/pick/{render_id}")
async def pick(job_id: int, render_id: int):
    H.choose_preview(job_id, render_id)
    return {"ok": True}


# ------------------------------------------------------------------ chat
class ChatIn(BaseModel):
    text: str
    attachments: list[dict] = []


@app.get("/api/chat")
async def chat_history(limit: int = 200):
    with proj().session() as s:
        rows = s.exec(select(m.ChatMessage).order_by(m.ChatMessage.id.desc()).limit(limit)).all()
    return list(reversed(rows))


@app.post("/api/chat")
async def chat(c: ChatIn):
    proj()
    if D.busy:
        raise HTTPException(409, "Qwen is still answering")

    async def run():
        try:
            await D.turn(c.text, c.attachments)
        except Exception as e:
            log.exception("chat turn failed")
            bus.publish("chat_error", error=f"{type(e).__name__}: {e}")
    asyncio.create_task(run())
    return {"ok": True}


# ------------------------------------------------------------------ files
def _file(file_id: int) -> tuple[m.FileRec, Path]:
    p = proj()
    with p.session() as s:
        f = s.get(m.FileRec, file_id)
    if f is None:
        raise HTTPException(404, "no such file")
    return f, p.abspath(f)


# File ids restart at 1 in every project (and can be reused after a delete), so the
# same URL can mean a different file: make the browser revalidate (cheap 304s).
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/api/files/{file_id}")
async def get_file(file_id: int):
    _, path = _file(file_id)
    if not path.exists():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(path, headers=NO_CACHE)


@app.get("/api/files/{file_id}/thumb")
async def thumb(file_id: int, size: int = 320):
    f, path = _file(file_id)
    if not path.exists():
        raise HTTPException(404, "file missing on disk")
    tdir = proj().media("thumbs")
    stamp = int(path.stat().st_mtime)
    out = tdir / f"{file_id}_{stamp}_{size}.jpg"
    if not out.exists():
        src = path
        if path.suffix.lower() in (".mp4", ".webm", ".mov"):
            frame = tdir / f"{file_id}_{stamp}_frame.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(path), "-frames:v", "1", str(frame)],
                           capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            src = frame
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            im.save(out, "JPEG", quality=85)
    return FileResponse(out, headers=NO_CACHE)


@app.post("/api/files/{file_id}/reveal")
async def reveal(file_id: int):
    _, path = _file(file_id)
    reveal_in_explorer(path)
    return {"ok": True}


class SaveAsIn(BaseModel):
    dest: str


@app.post("/api/files/{file_id}/save_as")
async def save_as(file_id: int, body: SaveAsIn):
    _, path = _file(file_id)
    dest = Path(body.dest)
    if dest.is_dir():
        dest = dest / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return {"path": str(dest)}


# ------------------------------------------------------------------ cleanup / export
@app.get("/api/cleanup/plan")
async def cleanup_plan():
    p = proj()
    with p.session() as s:
        return _plan_json(deps.orphan_plan(s), p)


@app.post("/api/cleanup")
async def cleanup():
    p = proj()
    with p.session() as s:
        plan = deps.orphan_plan(s)
        failed = deps.apply_delete(p, s, plan, force=True)
    shutil.rmtree(p.media("proxies"), ignore_errors=True)
    shutil.rmtree(p.media("thumbs"), ignore_errors=True)
    bus.publish("shot", id=None)
    bus.publish("reference", id=None)
    return {"failed": failed, "removed": len(plan.files)}


class CutIn(BaseModel):
    name: str | None = None
    loudnorm: bool = True
    upscale: bool = True                   # off: join the timeline previews as they are


@app.post("/api/final_cut")
async def final_cut(c: CutIn):
    return {"job_id": H.final_cut(c.name, c.loudnorm, c.upscale)}


@app.get("/api/final_cut/plan")
async def final_cut_plan():
    return H.cut_sources()


@app.get("/api/exports")
async def exports():
    with proj().session() as s:
        return s.exec(select(m.FileRec).where(m.FileRec.kind == "export").order_by(m.FileRec.id.desc())).all()


# ------------------------------------------------------------------ frontend
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


APP_IDS = {"qwenematics", "qwen-video-harness"}   # the old id, so a copy started before the rename is found


def main():
    global SERVER
    import uvicorn
    import webbrowser
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = f"http://127.0.0.1:{PORT}"
    if procs.port_in_use(PORT):
        try:
            running = httpx.get(f"{url}/api/ping", timeout=5).json().get("app") in APP_IDS
        except (httpx.HTTPError, ValueError):
            running = False
        if running:
            print(f"Qwenematics is already running at {url} - opening it. "
                  "Use the Shut down button (top right) to stop it.")
            webbrowser.open(url)
            raise SystemExit(0)
        who = procs.listener(PORT) or {}
        print(f"Port {PORT} is used by another program ({who.get('name', '?')}, pid {who.get('pid', '?')}). "
              "Close it and start Qwenematics again.")
        raise SystemExit(1)
    SERVER = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info",
                                          timeout_graceful_shutdown=3))   # don't wait on open browser tabs
    if "--open" in __import__("sys").argv:
        import threading
        threading.Timer(2.5, webbrowser.open, (url,)).start()
    SERVER.run()


if __name__ == "__main__":
    main()
