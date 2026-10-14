from pathlib import Path

import pytest
from sqlmodel import select

from harness.db import deps
from harness.db import models as m
from harness.db.project import Project


def _img(p: Path, name: str) -> Path:
    f = p / name
    f.write_bytes(b"x" * 10)
    return f


@pytest.fixture
def proj(tmp_path):
    p = Project.create(tmp_path, "Test Film")
    yield p
    p.close()


def _setup(proj):
    """Two shots sharing one character ref; shot 1 has a final render."""
    with proj.session() as s:
        ref = m.Reference(name="Aria", kind="character")
        s.add(ref); s.flush()
        f = proj.register_file(s, _img(proj.media("refs"), "aria_v1.png"), "image", "ref_version")
        v1 = m.RefVersion(ref_id=ref.id, n=1, file_id=f.id, prompt="aria")
        s.add(v1); s.flush()
        f.owner_id = v1.id
        ref.current_version_id = v1.id
        shots = []
        for i in (1, 2):
            sh = m.Shot(order=i, title=f"shot {i}", prompt=f"p{i}")
            s.add(sh); s.flush()
            s.add(m.ShotSlot(shot_id=sh.id, ref_id=ref.id, ref_version_id=v1.id, slot=1))
            shots.append(sh)
        s.flush()
        wf = "minimax_h3@v251"
        r = m.Render(shot_id=shots[0].id, stage="final", workflow=wf, slot_versions=[v1.id],
                     input_hash=deps.shot_hash(s, shots[0], wf))
        s.add(r); s.flush()
        rf = proj.register_file(s, _img(proj.media("renders"), "shot1.mp4"), "video", "render", r.id)
        shots[0].final_render_id = r.id
        s.add(shots[0])
        return ref.id, v1.id, [x.id for x in shots], r.id


def test_create_open_list(tmp_path, proj):
    assert proj.name == "Test Film"
    assert (proj.root / "media" / "refs").is_dir()
    again = Project.open(proj.root)
    assert again.name == "Test Film"
    again.close()
    assert Project.list(tmp_path)[0]["folder"] == "Test_Film"


def test_impact_and_staleness(proj):
    ref_id, v1, shots, _ = _setup(proj)
    wf = "minimax_h3@v251"
    with proj.session() as s:
        imp = deps.ref_impact(s, ref_id)
        assert [x["shot_id"] for x in imp.shots] == shots
        assert imp.rendered_count == 1
        assert not deps.is_stale(s, s.get(m.Shot, shots[0]), wf)

        f2 = proj.register_file(s, _img(proj.media("refs"), "aria_v2.png"), "image", "ref_version")
        v2 = m.RefVersion(ref_id=ref_id, n=2, file_id=f2.id, parent_id=v1)
        s.add(v2); s.flush()
        changed = deps.replace_ref_version(s, ref_id, v2.id, shot_ids=[shots[0]])   # fork: shot 2 keeps v1
        assert changed == [shots[0]]
        assert deps.is_stale(s, s.get(m.Shot, shots[0]), wf)
        assert deps.slot_versions(s, shots[1]) == [v1]

        # seeds don't make a shot stale, prompt edits do
        sh2 = s.get(m.Shot, shots[1])
        h = deps.shot_hash(s, sh2, wf)
        sh2.params = {"seed": 5}
        assert deps.shot_hash(s, sh2, wf) == h
        sh2.prompt = "changed"
        assert deps.shot_hash(s, sh2, wf) != h


def test_delete_ref_blocked_then_cascade(proj):
    ref_id, v1, shots, _ = _setup(proj)
    with proj.session() as s:
        plan = deps.ref_delete_plan(s, ref_id)
        assert len(plan.blocked_by) == 2
        with pytest.raises(ValueError):
            deps.apply_delete(proj, s, plan)
        plan = deps.ref_delete_plan(s, ref_id, cascade=True)
        files = [proj.abspath(f) for f in plan.files]
        assert all(p.exists() for p in files)
        assert deps.apply_delete(proj, s, plan) == []
        assert not any(p.exists() for p in files)
        assert s.exec(select(m.ShotSlot)).all() == []
        assert s.get(m.Reference, ref_id) is None


def test_delete_shot_removes_render_files(proj):
    ref_id, v1, shots, r_id = _setup(proj)
    with proj.session() as s:
        plan = deps.shot_delete_plan(s, shots[0], orphan_refs=True)
        # the ref version is still used by shot 2, so it must survive
        assert not any(isinstance(r, m.RefVersion) for r in plan.rows)
        deps.apply_delete(proj, s, plan)
        assert s.get(m.Render, r_id) is None
        assert not (proj.media("renders") / "shot1.mp4").exists()
        assert s.get(m.RefVersion, v1) is not None


def test_snapshot_archive_delete(tmp_path, proj):
    _setup(proj)
    outside = tmp_path / "comfy_output_file.mp4"
    outside.write_bytes(b"y")
    with proj.session() as s:
        proj.register_file(s, outside, "video", "render", 1, location="comfy_output")
    snap = proj.snapshot(tmp_path / "snaps", "Test Film copy")
    sp = Project.open(snap)
    assert sp.name == "Test Film copy"
    with sp.session() as s:
        assert len(s.exec(select(m.Shot)).all()) == 2
    sp.close()
    z = proj.archive(tmp_path / "arch" / "film.zip")
    import zipfile
    names = zipfile.ZipFile(z).namelist()
    assert "project.db" in names and "media/refs/aria_v1.png" in names
    plan = proj.delete_plan()
    assert str(outside) in plan["outside_files"]
    assert proj.delete_everything() == []
    assert not proj.root.exists() and not outside.exists()


def test_ids_are_never_reused(proj):
    from harness.db.project import next_id
    with proj.session() as s:
        a = proj.register_file(s, _img(proj.media("refs"), "a.png"), "image")
        b = proj.register_file(s, _img(proj.media("refs"), "b.png"), "image")
        ref = m.Reference(name="R")
        s.add(ref); s.flush()
        v = m.RefVersion(id=next_id(s, m.RefVersion), ref_id=ref.id, n=1, file_id=b.id)
        s.add(v); s.flush()
        top_file, top_version = b.id, v.id
        s.delete(v); s.flush(); s.delete(b)
    with proj.session() as s:
        c = proj.register_file(s, _img(proj.media("refs"), "c.png"), "image")
        assert c.id > top_file and c.id != a.id
        assert next_id(s, m.RefVersion) > top_version


def test_verbatim_prompt_keeps_text_but_enforces_style():
    from harness.styles import enforce_style
    line = "Style: Ink. Black ink wash."
    assert enforce_style("A fox in snow.", line) == "A fox in snow.\n\n" + line
    assert enforce_style("A fox.\nStyle: photo\nmore", line) == "A fox.\n" + line + "\nmore"
    assert enforce_style("A fox.", None) == "A fox."


def _preview_with_latents(proj, s, shot_id, name):
    r = m.Render(shot_id=shot_id, stage="preview", workflow="minimax_h3@v251")
    s.add(r); s.flush()
    r.file_id = proj.register_file(s, _img(proj.media("previews"), f"{name}.mp4"), "preview", "render", r.id).id
    r.latent_video_file_id = proj.register_file(s, _img(proj.media("previews"), f"{name}_v.latent"), "latent", "render", r.id).id
    r.latent_audio_file_id = proj.register_file(s, _img(proj.media("previews"), f"{name}_a.latent"), "latent", "render", r.id).id
    s.add(r); s.flush()
    return r


def test_cleanup_keeps_timeline_preview_latents(proj):
    _, _, shots, _ = _setup(proj)
    with proj.session() as s:
        keep = _preview_with_latents(proj, s, shots[1], "keep")
        drop = _preview_with_latents(proj, s, shots[1], "drop")
        sh = s.get(m.Shot, shots[1])
        sh.final_render_id = keep.id
        s.add(sh); s.flush()
        plan = deps.orphan_plan(s)
        ids = {f.id for f in plan.files}
        assert keep.latent_video_file_id not in ids and keep.file_id not in ids
        assert drop.latent_video_file_id in ids and drop in plan.rows
