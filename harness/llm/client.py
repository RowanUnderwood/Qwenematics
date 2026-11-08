"""OpenAI-compatible chat client for Ninfer (or any phase-2 backend).

Streams /v1/chat/completions, accumulating content, reasoning_content and tool-call
deltas. Ninfer notes (H:/Ninfer Qwen/README.md): reasoning_effort per request must be
none | low | medium | xhigh; earlier turns' reasoning is not resent.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from PIL import Image

DeltaCb = Callable[[str, str], Awaitable[None] | None]   # (kind: "content"|"reasoning", text)


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON text as the model wrote it

    def args(self) -> dict:
        try:
            v = json.loads(self.arguments or "{}")
            return v if isinstance(v, dict) else {"value": v}
        except json.JSONDecodeError as e:
            raise LLMError(f"tool {self.name}: arguments are not valid JSON ({e.msg}): {self.arguments[:300]}")

    def as_message_part(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)

    def as_message(self) -> dict:
        m: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            m["tool_calls"] = [t.as_message_part() for t in self.tool_calls]
        return m


class LLM:
    def __init__(self, base_url: str, model: str, effort: str | None = "low", timeout: float = 900):
        self.base = base_url.rstrip("/")
        self.model = model
        self.effort = effort
        self.http = httpx.AsyncClient(base_url=self.base, timeout=httpx.Timeout(timeout, connect=5))
        self._context_window: int | None = None

    async def close(self):
        await self.http.aclose()

    async def alive(self) -> bool:
        try:
            return (await self.http.get("/v1/models", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    async def context_window(self, default: int = 65536) -> int:
        if self._context_window is None:
            try:
                data = (await self.http.get("/v1/models", timeout=5)).json().get("data", [{}])[0]
                n = (data.get("context_window") or data.get("max_model_len")
                     or (data.get("meta") or {}).get("n_ctx"))           # llama.cpp
                self._context_window = int(n or default)
            except (httpx.HTTPError, ValueError, IndexError, TypeError):
                return default
        return self._context_window

    async def probe(self, check_vision: bool = False) -> dict:
        """What the server offers: {up, models, model_found, vision, video, n_ctx, vision_ok, error}."""
        out: dict[str, Any] = {"up": False, "models": [], "model_found": False, "vision": None,
                               "video": None, "n_ctx": None, "vision_ok": None, "error": ""}
        try:
            r = await self.http.get("/v1/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data") or []
        except (httpx.HTTPError, ValueError) as e:
            out["error"] = f"{type(e).__name__}: {e}"[:300]
            return out
        out["up"] = True
        out["models"] = [m.get("id") for m in data if m.get("id")]
        m = next((m for m in data if m.get("id") == self.model), data[0] if data else {})
        out["model_found"] = self.model in out["models"]
        mods = (m.get("architecture") or {}).get("input_modalities")
        if mods:
            out["vision"], out["video"] = "image" in mods, "video" in mods
        out["n_ctx"] = m.get("context_window") or m.get("max_model_len") or (m.get("meta") or {}).get("n_ctx")
        try:                                                # llama.cpp only
            p = await self.http.get("/props", timeout=5)
            if p.status_code == 200:
                md = p.json().get("modalities") or {}
                out["vision"] = md.get("vision", out["vision"])
                out["video"] = md.get("video", out["video"])
        except (httpx.HTTPError, ValueError):
            pass
        if check_vision:
            out["vision_ok"], err = await self._vision_check()
            if err:
                out["error"] = err
        return out

    async def _vision_check(self) -> tuple[bool, str]:
        """Show the model a red square and ask its colour: proves images really get through."""
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (220, 20, 20)).save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        try:
            reply = await self.chat([{"role": "user", "content": [
                {"type": "text", "text": "What colour is this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": uri}}]}], effort="none", max_tokens=200)
        except (LLMError, httpx.HTTPError) as e:
            return False, f"vision check: {e}"[:300]
        text = reply.content.lower()
        return "red" in text, "" if "red" in text else f"vision check: model said {reply.content[:80]!r}"

    async def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
                   effort: str | None = None, max_tokens: int = 4096, temperature: float | None = None,
                   on_delta: DeltaCb | None = None, tool_choice: str | None = None,
                   response_format: dict | None = None) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": True,
                                "max_tokens": max_tokens, "stream_options": {"include_usage": True}}
        eff = effort if effort is not None else self.effort
        if eff:
            body["reasoning_effort"] = eff
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if response_format:
            body["response_format"] = response_format

        reply = Reply()
        calls: dict[int, dict] = {}
        async with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                text = (await r.aread()).decode("utf-8", "replace")
                raise LLMError(f"HTTP {r.status_code}: {text[:1500]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise LLMError(str(chunk["error"])[:1500])
                if chunk.get("usage"):
                    reply.usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    d = choice.get("delta", {})
                    if d.get("reasoning_content"):
                        reply.reasoning += d["reasoning_content"]
                        await _emit(on_delta, "reasoning", d["reasoning_content"])
                    if d.get("content"):
                        reply.content += d["content"]
                        await _emit(on_delta, "content", d["content"])
                    for tc in d.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        reply.finish_reason = choice["finish_reason"]
        reply.tool_calls = [ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"])
                            for i, c in sorted(calls.items()) if c["name"]]
        return reply

    async def json_call(self, messages: list[dict], *, effort: str | None = None,
                        max_tokens: int = 4000, retries: int = 2) -> dict:
        """Ask for a JSON object; tolerate code fences and retry once with the parse error."""
        msgs = list(messages)
        last = ""
        for _ in range(retries + 1):
            reply = await self.chat(msgs, effort=effort, max_tokens=max_tokens)
            last = reply.content
            obj = extract_json(last)
            if obj is not None:
                return obj
            msgs = msgs + [reply.as_message(),
                           {"role": "user", "content": "That was not a single valid JSON object. "
                                                       "Reply with only the JSON object, no prose."}]
        raise LLMError(f"no JSON object in reply: {last[:500]}")


def extract_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


async def _emit(cb, kind, text):
    if cb is None:
        return
    r = cb(kind, text)
    if asyncio.iscoroutine(r):
        await r
