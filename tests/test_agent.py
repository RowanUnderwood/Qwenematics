"""Director loop with a scripted LLM: no GPU, no Ninfer."""
import asyncio
import json

import pytest

from harness.core import Harness
from harness.db import models as m
from harness.llm.agent import Director
from harness.llm.client import Reply, ToolCall


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.base = "fake"

    async def chat(self, messages, **kw):
        self.seen.append([dict(x) for x in messages])
        return self.replies.pop(0)

    async def context_window(self, default=65536):
        return 65536

    async def alive(self):
        return True


def call(tool_name, **args):
    return ToolCall(id=f"c_{tool_name}", name=tool_name, arguments=json.dumps(args))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness()
    monkeypatch.setattr(h.settings, "raw", {**h.settings.raw, "projects_root": str(tmp_path)})
    return h


def test_director_creates_shot_and_enforces_reuse(harness):
    async def go():
        h = harness
        await h.create_project("Agent Test")
        submitted = []
        h.queue.submit = lambda kind, lane, work, payload=None: submitted.append((kind, payload)) or len(submitted)
        h.llm = ScriptedLLM([
            Reply(tool_calls=[call("update_bible", kind="character", title="Mara",
                                   body="A fisherwoman in her 60s, yellow oilskin coat.")]),
            Reply(tool_calls=[call("upsert_shot", title="Harbour dawn", description="Mara mends a net.",
                                   duration_s=20)]),
            # creating without the reuse check is refused and returns candidates
            Reply(tool_calls=[call("create_reference", name="Mara - full body", kind="character",
                                   description="Mara, yellow oilskin", reuse_checked=False)]),
            Reply(tool_calls=[call("search_references", text="Mara fisherwoman"),
                              call("create_reference", name="Mara - full body", kind="character",
                                   description="Mara, yellow oilskin", reuse_checked=True)]),
            Reply(content="Planned shot 1 and queued Mara's reference."),
        ])
        d = Director(h)
        out = await d.turn("Make a short scene of an old fisherwoman at dawn.")
        assert out.startswith("Planned shot 1")
        with h.p().session() as s:
            shots = s.exec(m.Shot.__table__.select()).all()
            assert len(shots) == 1 and shots[0].duration_s == 15.0      # clamped to 15
            refs = s.exec(m.Reference.__table__.select()).all()
            assert len(refs) == 1                                         # only the checked create went through
            chat = s.exec(m.ChatMessage.__table__.select()).all()
            assert [c.role for c in chat] == ["user", "assistant"]
        assert submitted and submitted[0][0] == "ref_image"
        # the refused create told the model why
        tool_msgs = [x for x in h.llm.seen[3] if x.get("role") == "tool"]
        assert "check for reusable references first" in tool_msgs[-1]["content"]
        # project state in the system prompt reflects tool changes
        assert "Harbour dawn" in h.llm.seen[-1][0]["content"]
        assert "Mara" in h.llm.seen[-1][0]["content"]

        # second turn sees the first turn as plain text history, not tool traffic
        h.llm = ScriptedLLM([Reply(content="ok")])
        await d.turn("thanks")
        msgs = h.llm.seen[0]
        assert [x["role"] for x in msgs] == ["system", "user", "assistant", "user"]
        assert "tools used:" in msgs[2]["content"]
        await h.close_project()
    asyncio.run(go())


def test_assign_ref_repacks_frames_first(harness, tmp_path):
    async def go():
        h = harness
        await h.create_project("Slots")
        img = tmp_path / "a.png"
        from PIL import Image
        Image.new("RGB", (64, 64)).save(img)
        v_subject = h.import_reference_file(img, "Mara", "character")
        v_frame = h.import_reference_file(img, "Opening", "first_frame")
        sid = h.upsert_shot(None, title="s", mode="i2va")
        h.assign_ref(sid, version_id=v_subject)
        slots = h.assign_ref(sid, version_id=v_frame, role="first_frame")
        assert [(x["slot"], x["role"]) for x in slots] == [(1, "first_frame"), (2, "subject")]
        await h.close_project()
    asyncio.run(go())


def test_pending_slot_resolves_when_reference_renders(harness, tmp_path):
    async def go():
        h = harness
        await h.create_project("Pending")
        ref_id, _ = h.create_reference("Mara", "character", "a fisherwoman", render=False)
        sid = h.upsert_shot(None, title="s")
        slots = h.assign_ref(sid, ref_id=ref_id)
        assert slots[0]["pending"] and slots[0]["file_id"] is None

        class Ctx:
            job_id = 0
            def check(self): pass
            def progress(self, *a, **k): pass

        # no image and nothing rendering -> clear error instead of waiting forever
        import pytest as _pt
        from harness.core import HarnessError
        with _pt.raises(HarnessError, match="has no image"):
            await h._resolve_pending_slots(Ctx(), sid)
        # once an image exists the slot pins to it
        from PIL import Image
        img = tmp_path / "m.png"; Image.new("RGB", (32, 32)).save(img)
        vid = h.import_reference_file(img, "Mara", "character", ref_id=ref_id)
        slots = await h._resolve_pending_slots(Ctx(), sid)
        assert slots[0]["version_id"] == vid and not slots[0]["pending"]
        await h.close_project()
    asyncio.run(go())


def test_effective_mode():
    from harness.core import effective_mode
    assert effective_mode("ref2va", ["first_frame", "subject"]) == "i2va"
    assert effective_mode("ref2va", ["first_frame", "last_frame"]) == "fl2va"
    assert effective_mode("i2va", ["last_frame"]) == "l2va"
    assert effective_mode("i2va", ["subject"]) == "ref2va"
    assert effective_mode("t2va", ["subject"]) == "ref2va"
    assert effective_mode("i2va", []) == "t2va"
    assert effective_mode("ref2va", ["subject"]) == "ref2va"


def test_upsert_shot_tool_guards(harness):
    async def go():
        h = harness
        await h.create_project("Guards")
        from harness.llm.tools import ToolBox
        box = ToolBox(h)
        out = json.loads(await box.run("upsert_shot", {"title": "The airship", "prompt": "", "status": "rendering"}))
        assert out["ignored"] == ["prompt", "status"]
        sid = out["shot_id"]
        dup = json.loads(await box.run("upsert_shot", {"title": "the airship ", "description": "again"}))
        assert "already has the title" in dup["error"]
        assert "no shot 99" in json.loads(await box.run("upsert_shot", {"title": "x", "after_shot_id": 99}))["error"]
        with h.p().session() as s:
            assert len(s.exec(m.Shot.__table__.select()).all()) == 1
            assert s.get(m.Shot, sid).status == "draft"
        # a second render of a shot that is already queued is refused (UI and Qwen alike)
        h.queue.submit("shot", "video", lambda ctx: asyncio.sleep(3600), {"shot_id": sid})
        from harness.core import HarnessError
        with pytest.raises(HarnessError, match="already"):
            h.render_shot(sid, enhance=True)
        out = json.loads(await box.run("upsert_shot", {"shot_id": sid, "description": "new"}))
        assert "stale" in out["warning"]
        await h.close_project()
    asyncio.run(go())


def test_empty_director_reply_is_nudged(harness):
    async def go():
        h = harness
        await h.create_project("Nudge")
        h.llm = ScriptedLLM([Reply(tool_calls=[call("list_shots")]), Reply(content=""),
                             Reply(content="I listed the shots.")])
        assert await Director(h).turn("hi") == "I listed the shots."
        h.llm = ScriptedLLM([Reply(content=""), Reply(content="")])
        assert "without a reply" in await Director(h).turn("hi")
        await h.close_project()
    asyncio.run(go())


def test_video_prompt_writer_retries_then_fails_on_empty():
    from harness.llm import roles
    from harness.llm.client import LLMError
    shot = {"title": "t", "description": "d"}
    llm = ScriptedLLM([Reply(content="", finish_reason="length"), Reply(content="[Shot 1] ok")])
    assert asyncio.run(roles.write_h3_prompt(llm, bible="", shot=shot, mode="t2va", duration=5, slots=[],
                                             header=None)) == "[Shot 1] ok"
    llm = ScriptedLLM([Reply(content=""), Reply(content="  ")])
    with pytest.raises(LLMError, match="empty video prompt"):
        asyncio.run(roles.write_h3_prompt(llm, bible="", shot=shot, mode="t2va", duration=5, slots=[], header=None))


def test_cut_sources_reuse_upscales(harness, tmp_path):
    async def go():
        h = harness
        await h.create_project("Cut")
        p = h.p()
        def f(name):
            x = p.media("previews") / name; x.write_bytes(b"x"); return x
        a, b = h.upsert_shot(None, title="a"), h.upsert_shot(None, title="b")
        with p.session() as s:
            ids = []
            for sid in (a, b):
                r = m.Render(shot_id=sid, stage="preview", workflow="minimax_h3@v251")
                s.add(r); s.flush()
                r.file_id = p.register_file(s, f(f"{sid}.mp4"), "preview", "render", r.id).id
                r.latent_video_file_id = p.register_file(s, f(f"{sid}v.latent"), "latent", "render", r.id).id
                r.latent_audio_file_id = p.register_file(s, f(f"{sid}a.latent"), "latent", "render", r.id).id
                s.get(m.Shot, sid).final_render_id = r.id
                ids.append(r.id)
            up = m.Render(shot_id=a, stage="final", parent_id=ids[0])
            s.add(up); s.flush()
            up.file_id = p.register_file(s, f("a_final.mp4"), "video", "render", up.id).id
            up_id = up.id
        src = h.cut_sources()
        assert [(d["render_id"], d["upscaled_id"], d["can_upscale"]) for d in src] == \
            [(ids[0], up_id, True), (ids[1], None, True)]
        await h.close_project()
    asyncio.run(go())


def test_one_first_frame_per_shot(harness, tmp_path):
    async def go():
        h = harness
        await h.create_project("Frames")
        from PIL import Image
        img = tmp_path / "f.png"; Image.new("RGB", (32, 32)).save(img)
        v1 = h.import_reference_file(img, "Square", "location")
        v2 = h.import_reference_file(img, "Concept", "concept")
        sid = h.upsert_shot(None, title="s")
        h.assign_ref(sid, version_id=v1, role="first_frame")
        from harness.core import HarnessError
        with pytest.raises(HarnessError, match="already has a first frame .'Square'"):
            h.assign_ref(sid, version_id=v2, role="first_frame")
        assert [x["role"] for x in h.assign_ref(sid, version_id=v2, role="last_frame")] == ["first_frame", "last_frame"]
        await h.close_project()
    asyncio.run(go())
