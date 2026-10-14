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
