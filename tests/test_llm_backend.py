"""Local/remote LLM backend switch: settings view, client probing, clip media mode. Offline."""
import asyncio
from pathlib import Path

import httpx
import pytest

from harness import server
from harness.config import normalize_llm_url
from harness.core import Harness
from harness.llm import media
from harness.llm.agent import Director
from harness.llm.client import LLM


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness()
    monkeypatch.setattr(h.settings, "raw", {**h.settings.raw, "projects_root": str(tmp_path)})
    async def offline(full=False):
        return {}
    monkeypatch.setattr(h, "refresh_servers", offline)
    return h


@pytest.mark.parametrize("raw,want", [
    ("192.168.2.74:8080/v1", "http://192.168.2.74:8080"),
    ("http://192.168.2.74:8080/v1/", "http://192.168.2.74:8080"),
    (" 10.0.0.5:9000 ", "http://10.0.0.5:9000"),
    ("https://host:1/", "https://host:1"),
])
def test_normalize_llm_url(raw, want):
    assert normalize_llm_url(raw) == want


def test_settings_view_switches_profile(harness):
    s = harness.settings
    local = s.llm
    assert local["backend"] == "local" and local.get("launcher")
    asyncio.run(harness.set_llm(backend="remote", remote_url="http://1.2.3.4:5/v1", remote_model="m2"))
    r = s.llm
    assert r["backend"] == "remote" and r["base_url"] == "http://1.2.3.4:5" and r["model"] == "m2"
    assert r["media_mode"] == "frames" and "launcher" not in r and "stopper" not in r
    assert r["reasoning_effort"] == local["reasoning_effort"]        # shared knobs carry over
    assert harness.llm.base == "http://1.2.3.4:5" and harness.llm.model == "m2"
    asyncio.run(harness.set_llm(backend="local"))
    assert s.llm["base_url"] == local["base_url"] and harness.llm.base == local["base_url"].rstrip("/")
    assert s.llm_remote["base_url"] == "http://1.2.3.4:5"           # the remote edits are kept


def _mock_llm(handler) -> LLM:
    c = LLM("http://x", "qwen36-genesis-hermes")
    c.http = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))
    return c


LLAMA_MODELS = {"object": "list", "data": [{"id": "qwen36-genesis-hermes", "meta": {"n_ctx": 131072},
                                            "architecture": {"input_modalities": ["text", "image"]}}]}


def test_context_window_and_probe_read_llama_cpp():
    def handler(req: httpx.Request):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json=LLAMA_MODELS)
        if req.url.path == "/props":
            return httpx.Response(200, json={"modalities": {"vision": True, "video": True}})
        return httpx.Response(404)

    async def go():
        c = _mock_llm(handler)
        assert await c.context_window() == 131072
        p = await c.probe()
        assert p["up"] and p["model_found"] and p["vision"] and p["video"] and p["n_ctx"] == 131072
        await c.close()
    asyncio.run(go())


def test_probe_down():
    def handler(req):
        raise httpx.ConnectError("refused")

    async def go():
        c = _mock_llm(handler)
        p = await c.probe(check_vision=True)
        assert not p["up"] and "ConnectError" in p["error"] and p["vision_ok"] is None
        await c.close()
    asyncio.run(go())


def test_clip_parts_follows_media_mode(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(media, "frame_parts", lambda p, d, n=8, edge=768: (seen.append(("frames", d.name)) or [{"f": 1}], 5))
    monkeypatch.setattr(media, "video_part", lambda p, d, e=768, f=2: (seen.append(("video", d.name)) or {"v": 1}, 7))
    md = lambda sub: tmp_path / sub
    assert media.clip_parts(Path("x.mp4"), md, {"media_mode": "frames"}) == ([{"f": 1}], 5)
    assert media.clip_parts(Path("x.mp4"), md, {}) == ([{"v": 1}], 7)
    assert seen == [("frames", "frames"), ("video", "proxies")]


def test_put_llm_refuses_while_busy(harness, monkeypatch):
    d = Director(harness)
    monkeypatch.setattr(server, "H", harness, raising=False)
    monkeypatch.setattr(server, "D", d, raising=False)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as c:
            assert (await c.put("/api/llm", json={"backend": "bogus"})).status_code == 400
            d.busy = True
            assert (await c.put("/api/llm", json={"backend": "remote"})).status_code == 409
            d.busy = False
            r = await c.put("/api/llm", json={"backend": "remote", "remote_url": "9.9.9.9:80/v1"})
            assert r.status_code == 200 and r.json()["active"]["url"] == "http://9.9.9.9:80"
            assert r.json()["active"]["media_mode"] == "frames"
    asyncio.run(go())
