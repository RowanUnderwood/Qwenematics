"""Async ComfyUI client: upload, queue, follow progress over the websocket, fetch outputs."""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import websockets

ProgressCb = Callable[[dict], Awaitable[None] | None]


class ComfyError(Exception):
    pass


@dataclass
class OutputFile:
    filename: str
    subfolder: str
    type: str           # "output" | "temp" | "input"
    node: str
    kind: str           # images | gifs | latents | audio ...
    extra: dict = field(default_factory=dict)   # the raw history record (VHS adds workflow/fullpath)

    @property
    def rel(self) -> str:
        return f"{self.subfolder}/{self.filename}" if self.subfolder else self.filename


class ComfyClient:
    def __init__(self, base_url: str, name: str = ""):
        self.base = base_url.rstrip("/")
        self.name = name
        self.client_id = uuid.uuid4().hex
        self.http = httpx.AsyncClient(base_url=self.base, timeout=httpx.Timeout(60, connect=5))

    async def close(self):
        await self.http.aclose()

    # ------------------------------------------------------------ info
    async def alive(self) -> bool:
        try:
            r = await self.http.get("/system_stats", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def system_stats(self) -> dict:
        r = await self.http.get("/system_stats")
        r.raise_for_status()
        return r.json()

    async def object_info(self) -> dict:
        r = await self.http.get("/object_info", timeout=120)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------ files
    async def upload(self, path: Path, subfolder: str = "harness", name: str | None = None,
                     overwrite: bool = True) -> str:
        """Upload into ComfyUI's input folder; returns the value a loader node expects."""
        name = name or path.name
        with open(path, "rb") as f:
            r = await self.http.post("/upload/image", files={"image": (name, f)},
                                     data={"subfolder": subfolder, "type": "input",
                                           "overwrite": "true" if overwrite else "false"})
        if r.status_code != 200:
            raise ComfyError(f"upload {path.name} failed: {r.status_code} {r.text[:300]}")
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    async def download(self, f: OutputFile, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        params = {"filename": f.filename, "subfolder": f.subfolder, "type": f.type}
        async with self.http.stream("GET", "/view", params=params, timeout=None) as r:
            if r.status_code != 200:
                raise ComfyError(f"download {f.rel} failed: {r.status_code}")
            with open(dest, "wb") as out:
                async for chunk in r.aiter_bytes(1 << 20):
                    out.write(chunk)
        return dest

    # ------------------------------------------------------------ run
    async def queue(self, prompt: dict) -> str:
        r = await self.http.post("/prompt", json={"prompt": prompt, "client_id": self.client_id})
        if r.status_code != 200:
            try:
                err = r.json()
            except ValueError:
                err = r.text
            raise ComfyError(f"prompt rejected: {json.dumps(err)[:2000]}")
        return r.json()["prompt_id"]

    async def interrupt(self):
        await self.http.post("/interrupt")

    async def free(self, unload_models: bool = True, free_memory: bool = True):
        await self.http.post("/free", json={"unload_models": unload_models, "free_memory": free_memory})

    async def history(self, prompt_id: str) -> dict | None:
        r = await self.http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        return r.json().get(prompt_id)

    async def run(self, prompt: dict, on_progress: ProgressCb | None = None,
                  timeout: float = 3 * 3600) -> tuple[str, list[OutputFile]]:
        """Queue a prompt and wait for it, reporting progress; returns (prompt_id, outputs)."""
        ws_url = self.base.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        async with websockets.connect(ws_url, max_size=None) as ws:
            prompt_id = await self.queue(prompt)
            await _emit(on_progress, {"type": "queued", "prompt_id": prompt_id})
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ComfyError(f"timed out waiting for {prompt_id}")
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
                except asyncio.TimeoutError:
                    # quiet socket: the job may have finished while we weren't listening
                    if await self.history(prompt_id):
                        break
                    continue
                if isinstance(msg, bytes):
                    continue        # binary preview frames
                m = json.loads(msg)
                data = m.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                t = m.get("type")
                if t == "progress":
                    await _emit(on_progress, {"type": "progress", "node": data.get("node"),
                                              "value": data.get("value"), "max": data.get("max")})
                elif t == "executing" and data.get("node"):
                    await _emit(on_progress, {"type": "node", "node": data["node"]})
                elif t == "execution_error":
                    raise ComfyError(f"{data.get('node_type')} ({data.get('node_id')}): "
                                     f"{data.get('exception_message', '').strip()}")
                elif t == "execution_interrupted":
                    raise ComfyError("interrupted")
                elif t == "execution_success" or (t == "executing" and data.get("node") is None):
                    break
        hist = await self.history(prompt_id) or {}
        status = hist.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(f"execution failed: {json.dumps(status.get('messages'))[:1500]}")
        return prompt_id, collect_outputs(hist)


def collect_outputs(hist: dict) -> list[OutputFile]:
    files = []
    for node_id, out in (hist.get("outputs") or {}).items():
        for kind, items in out.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "filename" in it:
                    files.append(OutputFile(it["filename"], it.get("subfolder", ""),
                                            it.get("type", "output"), node_id, kind, dict(it)))
    return files


async def _emit(cb: ProgressCb | None, event: dict):
    if cb is None:
        return
    r = cb(event)
    if asyncio.iscoroutine(r):
        await r
