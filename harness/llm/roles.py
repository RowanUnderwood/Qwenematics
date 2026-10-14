"""The specialist roles: Krea prompt writer, H3 prompt writer, critic, captioner.

Each call is a fresh, short conversation (system prompt + one task), so these never
eat into the director's context. The writers get the bible text the director keeps.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import styles as style_lib
from ..config import ROOT
from .client import LLM

PROMPTS = ROOT / "prompts"


@lru_cache(maxsize=None)
def prompt_file(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _poses() -> dict:
    return json.loads((PROMPTS / "knowledge" / "krea_poses.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _styles() -> dict:
    return json.loads((PROMPTS / "knowledge" / "krea_styles.json").read_text(encoding="utf-8"))


def lookup_pose(name: str) -> str:
    poses = _poses()
    key = next((k for k in poses if k.lower() == name.lower()), None) or \
        next((k for k in poses if name.lower() in k.lower()), None)
    if not key:
        return f"No pose named '{name}'. Available: {', '.join(poses)}. Or write your own block."
    return f"Pose: {key}\n\n{poses[key]['block']}"


def search_styles(query: str, limit: int = 8) -> list[str]:
    """Rank style names by keyword hits: name x3, categories x2, paragraph x1."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
    scored = []
    for k, v in _styles().items():
        hay = (k.lower(), " ".join(v.get("categories") or [v.get("family", "")]).lower(),
               (v.get("paragraph") or "").lower())
        score = sum(3 * (w in hay[0]) + 2 * (w in hay[1]) + (w in hay[2]) for w in words)
        if score:
            scored.append((-score, k))
    return [k for _, k in sorted(scored)[:limit]]


def lookup_style(name: str) -> str:
    styles = _styles()
    key = next((k for k in styles if k.lower() == name.lower()), None) or \
        next((k for k in styles if name.lower() in k.lower()), None)
    if not key:
        hits = search_styles(name)
        if hits:
            return (f"No style named exactly '{name}'. Closest matches: {', '.join(hits)}. "
                    "Look one of these up, or write your own paragraph with the style formula.")
        fams: dict[str, list[str]] = {}
        for k, v in styles.items():
            fams.setdefault(v.get("family", "?"), []).append(k)
        sample = "; ".join(f"{f}: {', '.join(n[:8])}" for f, n in fams.items())
        return f"No style named '{name}'. Some names by family - {sample}"
    para = styles[key].get("paragraph")
    if para:
        return f"Style: {key}. {para}"
    return (f"'{key}' ({styles[key].get('family')}) exists in the library but its paragraph is not loaded. "
            "Write a 60-150 word paragraph for it yourself using the style formula.")


WRITER_TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_pose", "description": "Get the descriptor block for a named pose.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "lookup_style", "description": "Get the tested style paragraph for a named style "
        "(288 styles: anime, illustration, photography, painting, comics, 3D, cinema...). "
        "A partial name or a few keywords (e.g. 'noir photography', 'watercolor') returns the closest names.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
]


def _clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```\w*\n|\n?```$", "", t).strip()
    return t


async def _with_lookups(llm: LLM, messages: list[dict], max_rounds: int = 3) -> str:
    for _ in range(max_rounds):
        reply = await llm.chat(messages, tools=WRITER_TOOLS, max_tokens=8000)
        if not reply.tool_calls:
            return _clean(reply.content)
        messages = messages + [reply.as_message()]
        for tc in reply.tool_calls:
            args = tc.args()
            out = lookup_pose(args.get("name", "")) if tc.name == "lookup_pose" else lookup_style(args.get("name", ""))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    reply = await llm.chat(messages + [{"role": "user", "content": "Now write the final prompt."}], max_tokens=8000)
    return _clean(reply.content)


async def write_krea_prompt(llm: LLM, *, bible: str, name: str, kind: str, description: str,
                            previous_prompt: str | None = None, change: str | None = None,
                            critique: str | None = None, style: str | None = None) -> str:
    """`style` is the project's locked Style line; it overrides any style in the bible
    and is written into the prompt verbatim whatever the model does."""
    task = [f"Reference to make: {name}", f"Kind: {kind}", f"What it must show: {description}"]
    if style:
        task.append("\nLOCKED PROJECT STYLE (chosen by the user; it overrides any style notes in the bible). "
                    "Use this exact line as the Style block, do not call lookup_style, and keep the other "
                    f"blocks (medium words, lighting, palette) consistent with it:\n{style}")
    if previous_prompt:
        task.append(f"\nPrevious prompt:\n{previous_prompt}")
    if change:
        task.append(f"\nRequested change: {change}")
    if critique:
        task.append(f"\nCritic's notes on the last result: {critique}")
    msgs = [{"role": "system", "content": prompt_file("krea_writer.md")},
            {"role": "user", "content": f"PROJECT BIBLE (style, characters, places):\n{bible or '(empty)'}\n\n"
                                        + "\n".join(task)}]
    return style_lib.enforce_style(await _with_lookups(llm, msgs), style)


async def write_h3_prompt(llm: LLM, *, bible: str, shot: dict, mode: str, duration: float,
                          slots: list[dict], header: str | None, previous_clip: str | None = None,
                          previous_prompt: str | None = None, critique: str | None = None,
                          change: str | None = None, style: dict | None = None) -> str:
    lines = [f"MODE: {mode}", f"DURATION: {duration:.2f} seconds"]
    if style:
        lines.append(f"VISUAL STYLE (locked by the user for the whole project; the reference pictures are in it): "
                     f"{style['name']} - {style['paragraph']}\nState this style in the opening style sentence "
                     "and keep the whole clip in it; never call it live-action unless the style is photographic.")
    full_ref = any(s["role"] == "subject" for s in slots)
    lines.append("FORMAT: B (full-reference, six sections)" if full_ref else "FORMAT: A (base, three fields)")
    if header and not full_ref:
        lines.append(f"FIRST LINE (copy exactly): {header}")
    if slots:
        lines.append("LOADED PICTURES:")
        for s in slots:
            lines.append(f"  <Picture {s['slot']}> role={s['role']} kind={s['kind']} - {s['name']}: {s['caption'] or s['description']}")
    else:
        lines.append("LOADED PICTURES: none")
    lines.append(f"\nSHOT: {shot.get('title', '')}\nWhat happens: {shot.get('description', '')}")
    if shot.get("dialogue"):
        lines.append(f"Dialogue (verbatim): {shot['dialogue']}")
    if shot.get("audio_notes"):
        lines.append(f"Sound/music notes: {shot['audio_notes']}")
    if previous_clip:
        lines.append(f"\nThe previous clip ends on: {previous_clip}")
    if previous_prompt:
        lines.append(f"\nPrevious prompt for this shot:\n{previous_prompt}")
    if change:
        lines.append(f"\nRequested change: {change}")
    if critique:
        lines.append(f"\nCritic's notes on the last render: {critique}")
    msgs = [{"role": "system", "content": prompt_file("h3_writer.md")},
            {"role": "user", "content": f"PROJECT BIBLE:\n{bible or '(empty)'}\n\n" + "\n".join(lines)}]
    reply = await llm.chat(msgs, max_tokens=10000)
    return _clean(reply.content)


async def critique(llm: LLM, *, intent: str, prompt: str, media: list[dict],
                   transcript: str | None = None, candidates: int = 0) -> dict[str, Any]:
    text = [f"INTENT: {intent}", f"PROMPT USED:\n{prompt[:6000]}"]
    if transcript:
        text.append(f"SPEECH TRANSCRIPT: {transcript}")
    if candidates:
        text.append(f"There are {candidates} candidates, shown in order 1..{candidates}. Score the best one and set 'pick'.")
    content = list(media) + [{"type": "text", "text": "\n\n".join(text)}]
    msgs = [{"role": "system", "content": prompt_file("critic.md")}, {"role": "user", "content": content}]
    out = await llm.json_call(msgs, max_tokens=5000)
    out.setdefault("score", 0)
    out.setdefault("issues", [])
    out.setdefault("fix", "")
    return out


async def caption(llm: LLM, *, image_part: dict, name: str, kind: str) -> dict[str, Any]:
    msgs = [{"role": "user", "content": [image_part, {"type": "text", "text":
            f"This is the reference image '{name}' (kind: {kind}) for a film project. Reply with only JSON: "
            '{"caption": "<one line, <=30 words, the concrete visual facts a reader needs to recognise it>", '
            '"tags": ["<3-8 short lowercase tags>"]}'}]}]
    try:
        return await llm.json_call(msgs, max_tokens=600, effort="none")
    except Exception:
        return {"caption": "", "tags": []}
