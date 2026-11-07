"""Image-only projects: the mode, the director's tools and the image pipeline. No GPU, no Ninfer."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from sqlmodel import select

from harness import server
from harness.core import DEFAULT_PROJECT_SETTINGS, Harness, HarnessError
from harness.db import models as m
from harness.db.project import Project
from harness.llm import roles
from harness.llm.agent import Director
from harness.llm.client import Reply
from harness.llm.tools import IMAGE_TOOL_NAMES, TOOLS, ToolBox, tools_for
from test_agent import ScriptedLLM, call


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness()
    monkeypatch.setattr(h.settings, "raw", {**h.settings.raw, "projects_root": str(tmp_path)})
    return h


def _names(tools):
    return {t["function"]["name"] for t in tools}


# ------------------------------------------------------------------ the mode
def test_mode_row_survives_settings_writes_and_snapshot(harness, tmp_path):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        assert h.image_only()
        h.set_project_settings(json.loads(json.dumps(DEFAULT_PROJECT_SETTINGS)))     # what a save or a reset does
        assert h.image_only()
        snap = Project.open(h.p().snapshot(tmp_path, "Pics copy"))
        with snap.session() as s:
            assert s.get(m.Meta, "mode").value == "image"
        snap.close()
        await h.close_project()
    asyncio.run(go())


def test_project_without_the_row_is_full(harness):
    async def go():
        await harness.create_project("Film")
        assert not harness.image_only()
        with pytest.raises(HarnessError):
            harness.upgrade_project()
        await harness.close_project()
        assert not harness.image_only()                 # no project open
    asyncio.run(go())


def test_upgrade_unlocks_video(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        with pytest.raises(HarnessError):
            h.upsert_shot(None, title="x")
        h.upgrade_project()
        assert not h.image_only()
        assert h.upsert_shot(None, title="x")
        await h.close_project()
    asyncio.run(go())


def test_video_entry_points_refuse(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        for fn in (lambda: h.upsert_shot(None, title="x"), lambda: h.assign_ref(1, ref_id=1),
                   lambda: h.render_shot(1), lambda: h.final_cut()):
            with pytest.raises(HarnessError, match="image-only"):
                fn()
        await h.close_project()
    asyncio.run(go())


def test_routes_report_the_guard_and_the_mode(harness, monkeypatch):
    async def no_servers(full=False):
        return {}

    monkeypatch.setattr(harness, "refresh_servers", no_servers)
    monkeypatch.setattr(server, "H", harness, raising=False)      # normally set by the app's lifespan
    monkeypatch.setattr(server, "D", Director(harness), raising=False)

    async def go():
        await harness.create_project("Pics", image_only=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as c:
            r = await c.post("/api/shots", json={"title": "x"})
            assert r.status_code == 400 and "image-only" in r.json()["error"]
            assert (await c.post("/api/final_cut", json={})).status_code == 400
            assert (await c.get("/api/shots")).json() == []     # the UI still loads these
            assert (await c.get("/api/exports")).status_code == 200
            assert (await c.get("/api/status")).json()["project"]["mode"] == "image"
            assert (await c.post("/api/project/upgrade")).status_code == 200
            assert (await c.get("/api/status")).json()["project"]["mode"] == "full"
            assert (await c.post("/api/project/upgrade")).status_code == 400
        await harness.close_project()
    asyncio.run(go())


# ------------------------------------------------------------------ the director
def test_image_director_tools_prompt_and_state(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        h.update_bible("note", "Secret", "BIBLE-TEXT")
        submitted = []
        h.queue.submit = lambda kind, lane, work, payload=None: submitted.append(kind) or len(submitted)
        h.llm = ScriptedLLM([
            Reply(tool_calls=[call("render_shot", shot_id=1), call("get_bible")]),     # tools it wasn't offered
            Reply(tool_calls=[call("create_reference", name="Poster", kind="concept", description="A harbour")]),
            Reply(content="Queued."),
        ])
        await Director(h).turn("Make a poster.")
        assert _names(h.llm.tools[0]) == IMAGE_TOOL_NAMES
        system = h.llm.seen[0][0]["content"]
        assert system.startswith("You are an image designer")
        assert "BIBLE" not in system and "SHOTS" not in system and "video aspect" not in system
        tool_results = [x["content"] for x in h.llm.seen[1] if x["role"] == "tool"]
        assert all("not available" in r for r in tool_results)
        assert submitted == ["ref_image"]               # created without a reuse check
        with h.p().session() as s:
            assert s.exec(m.Shot.__table__.select()).all() == []
        await h.close_project()
    asyncio.run(go())


def test_image_tools_list_aspects_and_hide_video(harness):
    tools = tools_for(True, ["1:1 (Square)", "3:2 (Photo)"])
    create = next(t for t in tools if t["function"]["name"] == "create_reference")["function"]["parameters"]["properties"]
    assert create["aspect_ratio"]["enum"] == ["1:1 (Square)", "3:2 (Photo)"]
    assert "first_frame" not in create["kind"]["enum"] and "reuse_checked" not in create
    assert "enum" not in next(t for t in tools_for(True) if t["function"]["name"] == "create_reference")[
        "function"]["parameters"]["properties"]["aspect_ratio"]       # image server not seen yet


def test_image_toolbox_hides_shots(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        ref_id, _ = h.create_reference("Poster", "concept", "A harbour", render=False)
        box = ToolBox(h)
        assert "error" in json.loads(await box.run("view", {"item": "shot:1"}))
        assert "used_in_shots" not in json.loads(await box.run("get_reference", {"ref_id": ref_id}))
        hit = json.loads(await box.run("search_references", {"text": "Poster"}))["matches"][0]
        assert "used_in_shots" not in hit
        await h.close_project()
    asyncio.run(go())


def test_full_project_director_is_unchanged(harness):
    async def go():
        h = harness
        await h.create_project("Film")
        h.llm = ScriptedLLM([Reply(content="Hello.")])
        await Director(h).turn("Hi")
        assert h.llm.tools[0] is TOOLS and "make_variations" not in _names(TOOLS)
        system = h.llm.seen[0][0]["content"]
        assert system.startswith(roles.prompt_file("director.md")) and "video aspect" in system and "SHOTS" in system
        result = json.loads(await ToolBox(h).run("make_variations", {"ref_id": 1}))
        assert "not available" in result["error"]
        await h.close_project()
    asyncio.run(go())


# ------------------------------------------------------------------ the pipeline
@pytest.fixture
def pipeline(harness, monkeypatch, tmp_path):
    """A harness whose Krea renders and Qwen calls are stubbed. `runs` holds each run's params."""
    h, runs, written = harness, [], []

    async def render(wv, graph, params, *, dest_dir, tag, server, on_progress=None):
        img = Path(dest_dir) / f"{tag}_{len(runs)}.png"
        Image.new("RGB", (8, 8)).save(img)
        runs.append(params)
        return SimpleNamespace(outputs={"image": [img]})

    async def write(llm, **kw):
        written.append(kw)
        return "the prompt"

    async def critique(llm, **kw):
        return {"score": 9}

    async def caption(llm, **kw):
        return {"caption": "cap", "tags": []}

    monkeypatch.setattr(h.runner, "render", render)
    monkeypatch.setattr(roles, "write_krea_prompt", write)
    monkeypatch.setattr(roles, "critique", critique)
    monkeypatch.setattr(roles, "caption", caption)

    def capture():
        """Jobs run when `flush` is awaited, as the queue's image lane would run them."""
        work = []
        h.queue.submit = lambda kind, lane, fn, payload=None: work.append(fn) or len(work)
        ctx = SimpleNamespace(job_id=1, progress=lambda *a, **k: None, check=lambda: None)

        async def flush():
            jobs, work[:] = list(work), []
            await asyncio.gather(*(fn(ctx) for fn in jobs))
        return flush

    return SimpleNamespace(h=h, runs=runs, written=written, capture=capture)


def _versions(h, ref_id):
    with h.p().session() as s:
        ref = s.get(m.Reference, ref_id)
        vs = [x for x in s.exec(m.RefVersion.__table__.select()).all() if x.ref_id == ref_id]
        return ref.current_version_id, vs


def test_explicit_aspect_wins_and_revision_keeps_frame_and_seed(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Poster", "prop", "A boat", aspect="3:2 (Photo)")
        await flush()
        assert p.runs[0]["aspect_ratio"] == "3:2 (Photo)"       # a prop is otherwise 1:1
        assert p.written[0]["frame"] == "3:2 (Photo)" and p.written[0]["image_only"]
        p.h.revise_reference(ref_id, "make it red")
        await flush()
        assert p.runs[1]["aspect_ratio"] == "3:2 (Photo)"
        assert p.runs[1]["seed"] == p.runs[0]["seed"]
        await p.h.close_project()
    asyncio.run(go())


def test_concept_kind_uses_the_default_frame(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        p.h.create_reference("Idea", "concept", "Something")
        await flush()
        assert p.runs[0]["aspect_ratio"] == "16:9 (Widescreen)"
        await p.h.close_project()
    asyncio.run(go())


def test_a_seed_locked_in_settings_beats_the_kept_seed(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Poster", "concept", "A boat")
        await flush()
        cfg = p.h.project_settings()
        cfg["params"]["krea2"] = {"seed": 1234}
        p.h.set_project_settings(cfg)
        p.h.revise_reference(ref_id, "make it red")
        await flush()
        assert p.runs[1]["seed"] == 1234
        await p.h.close_project()
    asyncio.run(go())


def test_variations_differ_and_leave_the_current_version(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Poster", "prop", "A boat", aspect="3:2 (Photo)")
        await flush()
        current, _ = _versions(p.h, ref_id)
        cfg = p.h.project_settings()
        cfg["params"]["krea2"] = {"seed": 1234}           # even with a locked seed
        p.h.set_project_settings(cfg)
        written_before = len(p.written)
        assert len(p.h.make_variations(ref_id, 3)) == 3
        await flush()
        variations = p.runs[1:]
        assert len({x["seed"] for x in variations}) == 3 and 1234 not in {x["seed"] for x in variations}
        assert all(x["aspect_ratio"] == "3:2 (Photo)" and x["prompt"] == "the prompt" for x in variations)
        assert len(p.written) == written_before            # verbatim: no prompt was rewritten
        now, versions = _versions(p.h, ref_id)
        assert now == current
        assert sorted(v.n for v in versions) == [1, 2, 3, 4]
        await p.h.close_project()
    asyncio.run(go())


def test_variations_need_a_finished_image_and_a_sane_count(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        ref_id, _ = p.h.create_reference("Poster", "concept", "A boat", render=False)
        with pytest.raises(HarnessError, match="no finished render"):
            p.h.make_variations(ref_id)
        for bad in (0, 9):
            with pytest.raises(HarnessError):
                p.h.make_variations(ref_id, bad)
        await p.h.close_project()
    asyncio.run(go())


def test_bad_aspect_and_first_frame_are_refused(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        p.h.object_info["image"] = {"ResolutionSelector": {"input": {"required": {"aspect_ratio": [["1:1 (Square)"]]}}}}
        with pytest.raises(HarnessError, match="unknown aspect"):
            p.h.create_reference("Poster", "concept", "A boat", aspect="4:5")
        with pytest.raises(HarnessError, match="first_frame"):
            p.h.create_reference("Poster", "first_frame", "A boat")
        await p.h.close_project()
    asyncio.run(go())


def test_full_project_ignores_aspect_and_picks_a_new_seed(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Film")
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Boat", "prop", "A boat", aspect="3:2 (Photo)")
        await flush()
        assert p.runs[0]["aspect_ratio"] == "1:1 (Square)"
        assert not p.written[0]["image_only"]
        p.h.revise_reference(ref_id, "make it red")
        await flush()
        assert p.runs[1]["seed"] != p.runs[0]["seed"]
        with pytest.raises(HarnessError):
            p.h.make_variations(ref_id)
        await p.h.close_project()
    asyncio.run(go())


def test_accepted_change_sticks_for_later_revisions(pipeline, monkeypatch):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "concept", "A pixie with a hat")
        await flush()
        calls = _critic_says(monkeypatch, {"score": 9, "change_visible": True}, {"score": 9, "change_visible": True})
        p.h.revise_reference(ref_id, "make her an orb")
        await flush()
        p.h.revise_reference(ref_id, "make it brighter")
        await flush()
        assert "make her an orb" not in calls[0]["intent"]
        assert "overrides the above): make her an orb" in calls[1]["intent"]
        assert "overrides the above): make her an orb" in p.written[-1]["description"]
        await p.h.close_project()
    asyncio.run(go())


def test_rejected_change_is_not_kept(pipeline, monkeypatch):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "concept", "A pixie")
        await flush()
        _critic_says(monkeypatch, *[{"score": 8, "change_visible": False, "fix": "no"}] * 3)
        p.h.revise_reference(ref_id, "make her an orb")
        await flush()
        with p.h.p().session() as s:
            assert s.get(m.Reference, ref_id).description == "A pixie"
        await p.h.close_project()
    asyncio.run(go())


def test_digest_describes_the_current_look(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        p.h.create_reference("Mara", "character", "OLD DESCRIPTION")
        await flush()
        digest = p.h.image_digest()
        assert "Mara: cap" in digest and "OLD DESCRIPTION" not in digest      # the stub captions everything "cap"
        await p.h.close_project()
    asyncio.run(go())


def test_render_tools_say_qwen_will_not_be_woken(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        p.capture()
        box = ToolBox(p.h)
        made = json.loads(await box.run("create_reference", {"name": "N", "kind": "concept", "description": "D"}))
        revised = json.loads(await box.run("revise_reference", {"ref_id": made["ref_id"], "change": "x"}))
        assert "won't be told" in made["note"] and "won't be told" in revised["note"]
        await p.h.close_project()
    asyncio.run(go())


def test_digest_lists_finished_images_only(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        done, _ = p.h.create_reference("Mara", "character", "An old fisherwoman in a yellow coat")
        await flush()
        p.h.create_reference("Rendering", "prop", "Not finished", render=False)
        digest = p.h.image_digest()
        assert "Mara" in digest and "Rendering" not in digest
        assert "Mara" not in p.h.image_digest(exclude=done)
        p.h.update_bible("note", "Secret", "BIBLE-TEXT")
        p.h.revise_reference(done, "red coat")
        await flush()
        assert "BIBLE-TEXT" not in p.written[-1]["bible"]
        await p.h.close_project()
    asyncio.run(go())


def test_writer_and_critic_use_the_image_prompts(monkeypatch):
    async def go():
        seen = []

        class LLM:
            async def chat(self, msgs, **kw):
                seen.append(msgs)
                return Reply(content="a prompt")

            async def json_call(self, msgs, **kw):
                seen.append(msgs)
                return {"score": 8, "caption": "c", "tags": []}

        await roles.write_krea_prompt(LLM(), bible="[prop] Boat: red", name="N", kind="concept", description="D",
                                      image_only=True, frame="1:1 (Square)")
        assert seen[-1][0]["content"] == roles.prompt_file("krea_writer_image.md")
        assert "EXISTING IMAGES" in seen[-1][1]["content"] and "Frame: 1:1 (Square)" in seen[-1][1]["content"]
        await roles.critique(LLM(), intent="i", prompt="p", media=[], image_only=True)
        assert seen[-1][0]["content"] == roles.prompt_file("critic_image.md")
        await roles.caption(LLM(), image_part={"type": "text", "text": ""}, name="N", kind="concept", image_only=True)
        assert "film" not in seen[-1][0]["content"][1]["text"]
        for name in ("krea_writer_image.md", "critic_image.md", "director_image.md"):
            assert "video model" not in roles.prompt_file(name) and "seed hunt" not in roles.prompt_file(name).lower()
    asyncio.run(go())


# ------------------------------------------------------------------ a change that doesn't show up
def _critic_says(monkeypatch, *replies):
    calls, replies = [], list(replies)

    async def critique(llm, **kw):
        calls.append(kw)
        return await real_cap(dict(replies.pop(0)), kw.get("change"))
    monkeypatch.setattr(roles, "critique", critique)
    return calls


async def real_cap(reply, change):
    """The real critique() against a critic LLM that gives this reply."""
    class LLM:
        async def json_call(self, msgs, **kw):
            return reply
    return await REAL_CRITIQUE(LLM(), intent="i", prompt="p", media=[], change=change)


REAL_CRITIQUE = roles.critique


def test_critic_gets_the_change_and_a_missing_change_cannot_pass():
    async def go():
        seen = []

        class LLM:
            async def json_call(self, msgs, **kw):
                seen.append(msgs)
                return {"score": 9, "change_visible": False, "fix": ""}

        out = await REAL_CRITIQUE(LLM(), intent="i", prompt="p", media=[], change="add a bubble")
        assert out["score"] == roles.MISSING_CHANGE_SCORE and "add a bubble" in out["fix"]
        assert "REQUESTED CHANGE: add a bubble" in seen[0][1]["content"][-1]["text"]
        untouched = await REAL_CRITIQUE(LLM(), intent="i", prompt="p", media=[])
        assert untouched["score"] == 9                 # no change asked for: nothing to enforce
    asyncio.run(go())


def test_missing_change_retries_with_the_change_and_the_notes(pipeline, monkeypatch):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "concept", "A fairy")
        await flush()
        calls = _critic_says(monkeypatch, {"score": 8, "change_visible": False, "fix": "no bubble yet"},
                             {"score": 8, "change_visible": True})
        written, runs = len(p.written), len(p.runs)
        p.h.revise_reference(ref_id, "put her in a bubble")
        await flush()
        assert len(p.runs) - runs == 2                  # would have stopped at one on the raw 8
        assert all(c["change"] == "put her in a bubble" for c in calls)
        retry = p.written[written + 1]
        assert retry["change"] == "put her in a bubble" and retry["critique"] == "no bubble yet"
        current, versions = _versions(p.h, ref_id)
        assert current == max(v.id for v in versions)  # the attempt that showed it wins
        await p.h.close_project()
    asyncio.run(go())


def test_user_is_told_when_the_change_never_shows(pipeline, monkeypatch):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "concept", "A fairy")
        await flush()
        _critic_says(monkeypatch, *[{"score": 8, "change_visible": False, "fix": "no bubble"}] * 3)
        runs = len(p.runs)
        p.h.revise_reference(ref_id, "put her in a bubble")
        await flush()
        assert len(p.runs) - runs == 3                  # all the configured rounds were used
        with p.h.p().session() as s:
            notes = [c.content for c in s.exec(m.ChatMessage.__table__.select()).all() if c.role == "event"]
        assert "still isn't visible after 3 attempt(s)" in notes[-1]
        assert "still isn't" not in notes[0]            # the first render had no change to check
        await p.h.close_project()
    asyncio.run(go())


def test_first_render_cannot_report_a_missing_change(pipeline, monkeypatch):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        _critic_says(monkeypatch, {"score": 9, "change_visible": False})     # seen live: the critic adds it unasked
        ref_id, _ = p.h.create_reference("Navi", "concept", "A fairy")
        await flush()
        with p.h.p().session() as s:
            notes = [c.content for c in s.exec(m.ChatMessage.__table__.select()).all() if c.role == "event"]
            assert s.get(m.Reference, ref_id).description == "A fairy"
        assert notes == ["Image 'Navi' ready (critic 9/10)"]
        await p.h.close_project()
    asyncio.run(go())


def test_director_prompts_steer_retries_away_from_variations():
    image = roles.prompt_file("director_image.md")
    assert "can't add anything that's missing" in image and "nothing wakes you" in image
    assert "nothing wakes you" in roles.prompt_file("director.md")
    make = next(t for t in tools_for(True) if t["function"]["name"] == "make_variations")
    assert "revise_reference" in make["function"]["description"]


# ------------------------------------------------------------------ Qwen flubbing tool calls
def _row(h, role="assistant", content="", calls=()):
    with h.p().session() as s:
        s.add(m.ChatMessage(role=role, content=content, tool_calls=list(calls)))


def _used(name="revise_reference", args='{"ref_id": 1, "change": "x"}', result='{"job_id": 1}'):
    return {"name": name, "arguments": args, "result": result}


def test_history_keeps_recent_tool_calls_in_short_valid_form(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        _row(h, "user", "make it")
        long = "x" * 2000
        _row(h, content="Done.\n[tools used: revise_reference]",
             calls=[_used(args=json.dumps({"ref_id": 1, "change": long}), result="r" * 2000),
                    _used(args='{"ref_id": 1, "change": "cut mid-JSO')])     # saved arguments can be truncated
        msgs = Director(h).history()
        ids = [x["tool_calls"][0]["id"] for x in msgs if x.get("tool_calls")]
        assert len(ids) == 2 and [x["tool_call_id"] for x in msgs if x["role"] == "tool"] == ids
        for x in msgs:
            if x.get("tool_calls"):
                args = json.loads(x["tool_calls"][0]["function"]["arguments"])     # always valid JSON
                assert len(args.get("change", "")) <= 303
            if x["role"] == "tool":
                assert len(x["content"]) <= 200
        assert msgs[-1] == {"role": "assistant", "content": "Done."}               # marker gone, calls come first
        await h.close_project()
    asyncio.run(go())


def test_history_caps_how_many_turns_keep_their_calls(harness):
    from harness.llm import agent

    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        for i in range(agent.HISTORY_TOOL_TURNS + 3):
            _row(h, "user", f"ask {i}")
            _row(h, content=f"reply {i}", calls=[_used()])
        msgs = Director(h).history()
        assert sum(1 for x in msgs if x.get("tool_calls")) == agent.HISTORY_TOOL_TURNS
        assert sum(1 for x in msgs if x.get("content", "").startswith("reply")) == agent.HISTORY_TOOL_TURNS + 3
        await h.close_project()
    asyncio.run(go())


def _fake_queue(h):
    h.queue.submit = lambda kind, lane, work, payload=None: 1


def test_false_render_claim_is_sent_back_and_the_real_call_goes_through(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        _fake_queue(h)
        ref_id, _ = h.create_reference("Poster", "concept", "A boat", render=False)
        h.llm = ScriptedLLM([
            Reply(content="Changed it. It's rendering now.\n[tools used: revise_reference]"),
            Reply(tool_calls=[call("revise_reference", ref_id=ref_id, change="red")]),
            Reply(content="Queued the red version."),
        ])
        assert await Director(h).turn("Make it red") == "Queued the red version."
        assert "Harness check" in h.llm.seen[1][-1]["content"]
        with h.p().session() as s:
            row = [c for c in s.exec(m.ChatMessage.__table__.select()).all() if c.role == "assistant"][-1]
        assert [t["name"] for t in row.tool_calls] == ["revise_reference"]
        await h.close_project()
    asyncio.run(go())


def test_repeated_false_claim_is_flagged_to_the_user_and_not_looped(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        h.llm = ScriptedLLM([Reply(content="It's rendering."), Reply(content="Yes, it's rendering. [tools used: x]")])
        out = await Director(h).turn("Make it red")
        assert out == "Yes, it's rendering.\n\n(Note: no render was actually started.)"
        assert not h.llm.replies
        await h.close_project()
    asyncio.run(go())


def test_true_statements_about_rendering_are_left_alone(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        with h.p().session() as s:
            s.add(m.Job(kind="ref_image", server="image", payload={"ref_id": 1}, state="running"))
        h.llm = ScriptedLLM([Reply(content="It's still rendering.")])              # one is: nothing to challenge
        assert await Director(h).turn("Is it done?") == "It's still rendering."
        with h.p().session() as s:
            for j in s.exec(select(m.Job)).all():
                j.state = "done"
                s.add(j)
        h.llm = ScriptedLLM([Reply(content="The frame looks good to me.")])        # no claim at all
        assert await Director(h).turn("Thoughts?") == "The frame looks good to me."
        await h.close_project()
    asyncio.run(go())


def test_duplicate_render_is_refused_but_variations_do_not_block(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        ref_id, _ = h.create_reference("Poster", "concept", "A boat", render=False)
        other, _ = h.create_reference("Other", "concept", "A tree", render=False)
        with h.p().session() as s:
            s.add(m.Job(kind="ref_image", server="image", payload={"ref_id": ref_id, "change": "red"}, state="queued"))
            s.add(m.Job(kind="ref_image", server="image", payload={"ref_id": other, "change": "variation"}, state="running"))
        submitted = []
        h.queue.submit = lambda kind, lane, work, payload=None: submitted.append(payload) or 99
        box = ToolBox(h)
        out = json.loads(await box.run("revise_reference", {"ref_id": ref_id, "change": "blue"}))
        assert out["not_started"] and "already rendering" in out["reason"] and not submitted
        assert json.loads(await box.run("make_variations", {"ref_id": ref_id}))["not_started"]
        assert json.loads(await box.run("revise_reference", {"ref_id": other, "change": "blue"}))["job_id"] == 99
        await h.close_project()
    asyncio.run(go())


def test_the_ui_can_still_queue_a_revision_during_a_render(harness):
    async def go():
        h = harness
        await h.create_project("Pics", image_only=True)
        ref_id, _ = h.create_reference("Poster", "concept", "A boat", render=False)
        with h.p().session() as s:
            s.add(m.Job(kind="ref_image", server="image", payload={"ref_id": ref_id}, state="running"))
        h.queue.submit = lambda kind, lane, work, payload=None: 5
        assert h.revise_reference(ref_id, "blue") == 5                             # the refusal is for Qwen's tool
        await h.close_project()
    asyncio.run(go())


def test_revision_can_change_the_frame_and_later_revisions_keep_it(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "character_closeup", "A fairy")
        await flush()
        assert p.runs[0]["aspect_ratio"] == "3:4 (Portrait Standard)"
        out = json.loads(await ToolBox(p.h).run("revise_reference", {
            "ref_id": ref_id, "change": "add Link", "aspect_ratio": "16:9 (Widescreen)"}))
        assert "error" not in out
        await flush()
        assert p.runs[1]["aspect_ratio"] == "16:9 (Widescreen)" and p.written[-1]["frame"] == "16:9 (Widescreen)"
        assert p.runs[1]["seed"] == p.runs[0]["seed"]
        p.h.revise_reference(ref_id, "add a tree")
        await flush()
        assert p.runs[2]["aspect_ratio"] == "16:9 (Widescreen)"
        await p.h.close_project()
    asyncio.run(go())


def test_frame_only_revision_gets_a_change_and_full_projects_ignore_aspect(pipeline):
    async def go():
        p = pipeline
        await p.h.create_project("Pics", image_only=True)
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Navi", "concept", "A fairy")
        await flush()
        p.h.revise_reference(ref_id, aspect="1:1 (Square)")
        await flush()
        assert "1:1 (Square)" in p.written[-1]["change"] and p.runs[-1]["aspect_ratio"] == "1:1 (Square)"
        await p.h.close_project()

        await p.h.create_project("Film")
        flush = p.capture()
        ref_id, _ = p.h.create_reference("Boat", "prop", "A boat")
        await flush()
        p.h.revise_reference(ref_id, "red", aspect="16:9 (Widescreen)")
        await flush()
        assert p.runs[-1]["aspect_ratio"] == "1:1 (Square)"
        await p.h.close_project()
    asyncio.run(go())


def test_revise_schema_offers_the_frame_only_in_image_mode():
    image = next(t for t in tools_for(True, ["1:1 (Square)"]) if t["function"]["name"] == "revise_reference")
    assert image["function"]["parameters"]["properties"]["aspect_ratio"]["enum"] == ["1:1 (Square)"]
    full = next(t for t in TOOLS if t["function"]["name"] == "revise_reference")
    assert "aspect_ratio" not in full["function"]["parameters"]["properties"]
