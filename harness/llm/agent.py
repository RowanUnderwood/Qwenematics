"""The director: a tool-using chat loop over the project, with context budgeting.

Context each call = director.md + a freshly generated compact PROJECT STATE + the chat
history + any media for this turn. Recent turns keep their tool calls in shortened form:
with only text to go on, Qwen copies "I did it" replies without calling anything. Older
turns are plain text. Old media becomes text placeholders. When history passes
`summarize_at` of the context window, the oldest half is folded into a summary message.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from sqlmodel import select

from ..db import deps
from ..db import models as m
from ..events import bus
from . import media, roles
from .tools import ToolBox, tools_for

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 16
CHARS_PER_TOKEN = 3.5
HISTORY_TOOL_TURNS = 6            # how many recent tool-using turns keep their tool calls
HISTORY_ARG_CHARS = 300           # long strings in those calls are cut to this
HISTORY_RESULT_CHARS = 200
RENDER_TOOLS = {"create_reference", "revise_reference", "make_variations", "render_shot"}
TOOLS_MARKER = re.compile(r"\s*\[tools used:[^\]]*\]")     # older replies copied this from a former history format
RENDER_CLAIM = re.compile(r"\b(rendering|re-?queued|queued)\b", re.I)


def _short_args(arguments: str) -> str:
    """Valid JSON with long strings cut. Saved arguments may already be cut mid-JSON."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return '{"note": "arguments omitted"}'
    if not isinstance(args, dict):
        return '{"note": "arguments omitted"}'
    return json.dumps({k: v[:HISTORY_ARG_CHARS] + "..." if isinstance(v, str) and len(v) > HISTORY_ARG_CHARS else v
                       for k, v in args.items()})


class Director:
    def __init__(self, harness):
        self.h = harness
        self.busy = False
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- state
    def project_state(self) -> str:
        h = self.h
        pset = h.project_settings()
        image_only = h.image_only()
        lines = [f"PROJECT: {h.p().name}" + ("" if image_only else f"   video aspect: {pset['video_aspect']}")]
        if pset.get("krea_style"):
            lines.append(f"LOCKED VISUAL STYLE: {pset['krea_style']} (chosen by the user in the "
                         f"{'Images' if image_only else 'Bible'} tab). Every image is rendered in it automatically; "
                         "don't propose another style unless the user asks"
                         + ("" if image_only else ", and keep bible style notes consistent with it")
                         + ". Only the user can change it.")
        if not image_only:
            bible = h.bible_text(2500)
            lines.append("BIBLE (abridged; get_bible for all):\n" + (bible or "(empty)"))
            wf = h.registry.get("minimax_h3").key
            jobs = h.active_shot_jobs()
        with h.p().session() as s:
            if not image_only:
                shots = s.exec(select(m.Shot).order_by(m.Shot.order)).all()
                lines.append(f"SHOTS ({len(shots)}):" if shots else "SHOTS: none yet")
                for i, x in enumerate(shots, 1):
                    pics = ", ".join(f"P{sl['slot']}={sl['name']}" + ("(old ver)" if sl["outdated"] else "")
                                     for sl in h.shot_slots(x.id))
                    j = jobs.get(x.id)
                    status = ("rendering" if j.state == "running" else "queued") if j else x.status
                    stale = " STALE" if not j and deps.is_stale(s, x, x.workflow or wf) else ""
                    lines.append(f"  #{i} shot_id={x.id} '{x.title}' {x.mode} {x.duration_s:g}s [{status}{stale}] "
                                 f"{x.description[:120]} | pictures: {pics or '-'}")
            refs = s.exec(select(m.Reference).order_by(m.Reference.id)).all()
            label = "IMAGES" if image_only else "REFERENCES"
            lines.append(f"{label} ({len(refs)}):" if refs else f"{label}: none yet")
            for r in refs[-60:]:
                v = s.get(m.RefVersion, r.current_version_id) if r.current_version_id else None
                cap = (v.caption if v else "rendering...")[:110]
                lines.append(f"  ref_id={r.id} [{r.kind}] {r.name} v{v.n if v else 0}: {cap}")
            jobs = s.exec(select(m.Job).where(m.Job.state.in_(["queued", "running"]))).all()
            if jobs:
                lines.append("ACTIVE JOBS: " + "; ".join(f"#{j.id} {j.kind} {j.state} {int(j.progress * 100)}% {j.message[:60]}"
                                                         for j in jobs))
        return "\n".join(lines)

    def _system(self) -> str:
        name = "director_image.md" if self.h.image_only() else "director.md"
        return roles.prompt_file(name) + "\n\n# PROJECT STATE\n" + self.project_state()

    # ---------------------------------------------------------------- media
    def media_parts(self, items: list[dict | str]) -> tuple[list[dict], list[str]]:
        """Attachments -> (message parts, text labels)."""
        h, proj = self.h, self.h.p()
        parts, labels = [], []
        llm = h.settings.llm
        with proj.session() as s:
            for it in items:
                if isinstance(it, str):
                    kind, _, ident = it.partition(":")
                    it = {"type": {"ref": "ref", "shot": "shot"}.get(kind, kind), "id": int(ident)}
                t, i = it["type"], it["id"]
                f, label, is_video = None, "", False
                if t == "ref":
                    ref = s.get(m.Reference, i)
                    if ref and ref.current_version_id:
                        v = s.get(m.RefVersion, ref.current_version_id)
                        f, label = s.get(m.FileRec, v.file_id), f"reference '{ref.name}' v{v.n} (ref_id={ref.id})"
                elif t == "ref_version":
                    v = s.get(m.RefVersion, i)
                    ref = s.get(m.Reference, v.ref_id) if v else None
                    if v:
                        f, label = s.get(m.FileRec, v.file_id), f"reference '{ref.name}' v{v.n} (ref_id={ref.id})"
                elif t in ("shot", "render"):
                    r = None
                    if t == "shot":
                        sh = s.get(m.Shot, i)
                        r = s.get(m.Render, sh.final_render_id) if sh and sh.final_render_id else None
                    else:
                        r = s.get(m.Render, i)
                    if r and r.file_id:
                        sh = s.get(m.Shot, r.shot_id)
                        f, is_video = s.get(m.FileRec, r.file_id), True
                        label = f"{'final clip' if r.stage == 'final' else 'preview'} of shot '{sh.title}' (shot_id={sh.id}, render {r.id})"
                if f is None:
                    labels.append(f"[{t} {i}: nothing to show yet]")
                    continue
                path = proj.abspath(f)
                if is_video:
                    p, _ = media.clip_parts(path, proj.media, llm)
                    parts += [{"type": "text", "text": label}] + p
                else:
                    p, _ = media.image_part(path, llm.get("proxy_long_edge", 768))
                    parts += [{"type": "text", "text": label}, p]
                labels.append(f"[attached: {label}]")
        return parts, labels

    # ---------------------------------------------------------------- history
    def history(self) -> list[dict]:
        with self.h.p().session() as s:
            rows = s.exec(select(m.ChatMessage).where(m.ChatMessage.summarized == False)  # noqa: E712
                          .order_by(m.ChatMessage.id)).all()
        with_calls = {r.id for r in rows if r.role == "assistant" and r.tool_calls}
        keep_calls = set(sorted(with_calls)[-HISTORY_TOOL_TURNS:])
        msgs = []
        for r in rows:
            if r.role == "system_note":
                msgs.append({"role": "user", "content": f"[Summary of the earlier conversation]\n{r.content}"})
                msgs.append({"role": "assistant", "content": "Understood."})
            elif r.role == "user":
                labels = " ".join(a.get("label", "") for a in r.attachments)
                msgs.append({"role": "user", "content": (r.content + (f"\n{labels}" if labels else "")).strip()})
            elif r.role == "assistant":
                if r.id in keep_calls:
                    for i, t in enumerate(r.tool_calls):
                        cid = f"old{r.id}_{i}"
                        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function", "function": {
                            "name": t.get("name", ""), "arguments": _short_args(t.get("arguments", ""))}}]})
                        msgs.append({"role": "tool", "tool_call_id": cid, "content": t.get("result", "")[:HISTORY_RESULT_CHARS]})
                text = TOOLS_MARKER.sub("", r.content or "").strip()
                if text:
                    msgs.append({"role": "assistant", "content": text})
        return msgs

    async def maybe_summarize(self):
        window = await self.h.llm.context_window()
        budget = window * float(self.h.settings.llm.get("summarize_at", 0.6))
        hist = self.history()
        used = sum(len(json.dumps(x)) for x in hist) / CHARS_PER_TOKEN + 6000
        if used < budget or len(hist) < 8:
            return
        with self.h.p().session() as s:
            rows = s.exec(select(m.ChatMessage).where(m.ChatMessage.summarized == False)  # noqa: E712
                          .order_by(m.ChatMessage.id)).all()
        old = rows[: len(rows) // 2]
        text = "\n".join(f"{r.role.upper()}: {r.content[:1500]}" for r in old if r.content)
        project, listed = ("image-project", "images") if self.h.image_only() else ("film-project", "shots, references")
        reply = await self.h.llm.chat([
            {"role": "system", "content": f"Summarise this {project} conversation for the director's memory. "
                                          "Keep decisions, user preferences, rejected ideas, open questions and promises. "
                                          f"Do not restate what the project state already lists ({listed}). Max 300 words."},
            {"role": "user", "content": text}], max_tokens=800, effort="none")
        with self.h.p().session() as s:
            s.add(m.ChatMessage(role="system_note", content=reply.content.strip(), summary_of=[r.id for r in old]))
            for r in old:
                r.summarized = True
                s.add(r)
        bus.publish("chat_summarized", count=len(old))

    # ---------------------------------------------------------------- turn
    async def turn(self, text: str, attachments: list[dict] | None = None) -> str:
        if self.busy:
            raise RuntimeError("Qwen is still answering the previous message")
        self.busy = True
        self._task = asyncio.current_task()
        try:
            return await self._turn(text, attachments or [])
        finally:
            self.busy, self._task = False, None

    def stop(self) -> bool:
        """Cancel the running turn (the Stop button). Tool calls already made stand."""
        if self._task and not self._task.done():
            self._task.cancel()
            return True
        return False

    async def _nudge(self, messages: list[dict], calls: list[dict], on_delta) -> str:
        """Qwen ended its turn with no text (often reasoning ran out of tokens): ask once
        for a short report so the user is never left with an empty reply."""
        try:
            reply = await self.h.llm.chat(messages + [{"role": "user", "content":
                                          "Briefly tell the user what you just did and what is next."}],
                                          effort="none", max_tokens=1500, on_delta=on_delta)
            if reply.content.strip():
                return reply.content.strip()
        except Exception as e:
            log.warning("nudge failed: %s", e)
        used = ", ".join(sorted({c["name"] for c in calls})) or "none"
        return f"(Qwen ended its turn without a reply. Tools used: {used}.)"

    def _false_claim(self, text: str, calls: list[dict]) -> bool:
        """The reply says something is rendering, but nothing was started and nothing is running."""
        if any(c["name"] in RENDER_TOOLS for c in calls) or not (RENDER_CLAIM.search(text) or TOOLS_MARKER.search(text)):
            return False
        with self.h.p().session() as s:
            return not s.exec(select(m.Job).where(m.Job.state.in_(["queued", "running"]))).first()

    async def _turn(self, text: str, attachments: list[dict]) -> str:
        h = self.h
        await self.maybe_summarize()
        parts, labels = self.media_parts(attachments)
        att_rows = [{**a, "label": lb} for a, lb in zip(attachments, labels)]
        with h.p().session() as s:
            um = m.ChatMessage(role="user", content=text, attachments=att_rows)
            s.add(um)
            s.flush()
            uid = um.id
        bus.publish("chat_message", id=uid, role="user", content=text, attachments=att_rows)

        history = self.history()[:-1]          # the user row we just saved is re-added with media below
        user_content: Any = text
        if parts:
            user_content = parts + [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": self._system()},
                    *history, {"role": "user", "content": user_content}]

        box = ToolBox(h)
        if box.image_only:
            await h.ensure_object_info("image")
        tools = tools_for(box.image_only, h.image_aspects() if box.image_only else None)
        all_calls: list[dict] = []

        async def on_delta(kind, chunk):
            bus.publish("chat_delta", stream=kind, text=chunk)

        try:
            final = await self._rounds(messages, tools, box, all_calls, on_delta)
        except asyncio.CancelledError:
            self._save_reply("(Stopped by you.)", all_calls)
            raise
        return self._save_reply(final, all_calls)

    async def _rounds(self, messages: list[dict], tools: list[dict], box: ToolBox, all_calls: list[dict], on_delta) -> str:
        h, final, challenged = self.h, "", False
        for _ in range(MAX_TOOL_ROUNDS):
            reply = await h.llm.chat(messages, tools=tools, on_delta=on_delta, max_tokens=12000)
            if not reply.tool_calls:
                final = reply.content.strip()
                if self._false_claim(final, all_calls):
                    if not challenged:
                        challenged = True
                        log.warning("director claimed a render without calling a tool; asking it to correct itself")
                        bus.publish("chat_reset")      # the claim was streamed; the corrected reply replaces it
                        messages += [reply.as_message(), {"role": "user", "content":
                                     "[Harness check, not the user] You said a render is under way, but you called no "
                                     "render tool this turn and no job is running, so nothing is rendering. If a render "
                                     "is needed, call the tool now. If not, reply again without saying it's rendering. "
                                     "Don't mention this check."}]
                        continue
                    final = TOOLS_MARKER.sub("", final).strip() + "\n\n(Note: no render was actually started.)"
                final = TOOLS_MARKER.sub("", final).strip()
                if not final:
                    log.warning("director ended with no text (finish_reason=%s)", reply.finish_reason)
                    final = await self._nudge(messages, all_calls, on_delta)
                break
            messages.append(reply.as_message())
            for tc in reply.tool_calls:
                try:
                    args = tc.args()
                except Exception as e:
                    result = json.dumps({"error": str(e)})
                else:
                    bus.publish("tool_call", name=tc.name, args=args)
                    result = await box.run(tc.name, args)
                    bus.publish("tool_result", name=tc.name, result=result[:2000])
                all_calls.append({"name": tc.name, "arguments": tc.arguments[:2000], "result": result[:2000]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if box.asked:
                final = box.asked
                break
            if box.pending_media:
                vparts, vlabels = self.media_parts(box.pending_media)
                box.pending_media = []
                messages.append({"role": "user", "content": vparts + [{"type": "text", "text":
                                 "Here is what you asked to view: " + " ".join(vlabels)}]})
            # refresh the state line so the model sees what its tools changed
            messages[0]["content"] = self._system()
        else:
            final = "(I stopped after many tool calls - tell me how to continue.)"
        return final

    def _save_reply(self, final: str, all_calls: list[dict]) -> str:
        with self.h.p().session() as s:
            am = m.ChatMessage(role="assistant", content=final, tool_calls=all_calls)
            s.add(am)
            s.flush()
            aid = am.id
        bus.publish("chat_message", id=aid, role="assistant", content=final, tool_calls=all_calls)
        return final
