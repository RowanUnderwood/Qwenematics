"""The Krea 2 style catalog and the project's locked style.

Paragraphs come from prompts/knowledge/krea_styles.json (merged from the Krea 2 Style
Library, https://matplinta.github.io/t2i-krea-2-style-library/). Its sample images
(one per style per base subject) are fetched on first view and cached under
cache/krea_styles/, with small JPEG thumbnails beside them.

A project can lock one style (project setting "krea_style"). The lock is enforced, not
just suggested: enforce_style() rewrites the Style line of every Krea prompt.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
from functools import lru_cache
from pathlib import Path

import httpx
from PIL import Image

from .config import ROOT

KNOW = ROOT / "prompts" / "knowledge"
CACHE = ROOT / "cache" / "krea_styles"
SITE = "https://matplinta.github.io/t2i-krea-2-style-library/"
_locks: dict[Path, asyncio.Lock] = {}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@lru_cache(maxsize=1)
def catalog() -> list[dict]:
    """[{id, name, family, categories, paragraph}] in the library's original order."""
    styles = json.loads((KNOW / "krea_styles.json").read_text(encoding="utf-8"))
    src = KNOW / "krea_styles.source.json"
    ids = {}
    if src.exists():
        ids = {s["name"]: s["id"] for s in json.loads(src.read_text(encoding="utf-8"))["styles"]}
    return [{"id": ids.get(k, slug(k)), "name": k, "family": v.get("family"),
             "categories": v.get("categories") or [v.get("family")], "paragraph": v.get("paragraph") or "",
             "has_image": k in ids}
            for k, v in styles.items()]


@lru_cache(maxsize=1)
def bases() -> list[dict]:
    """The sample subjects each style was rendered on: [{id, label, width, height, prompt}]."""
    data = json.loads((KNOW / "krea_style_base_prompts.json").read_text(encoding="utf-8"))
    return [{k: p[k] for k in ("id", "label", "width", "height", "prompt")} for p in data["prompts"] if p.get("enabled", True)]


def get(name: str | None) -> dict | None:
    if not name:
        return None
    return next((s for s in catalog() if s["name"].lower() == name.lower()), None)


def style_line(name: str | None) -> str | None:
    s = get(name)
    return f"Style: {s['name']}. {s['paragraph']}" if s and s["paragraph"] else None


def enforce_style(prompt: str, line: str | None) -> str:
    """Make the prompt's Style block exactly `line`: replace an existing 'Style:' line,
    or add it (in its own blank-line block) when the writer left it out."""
    if not line:
        return prompt
    lines = prompt.splitlines()
    idx = [i for i, l in enumerate(lines) if l.strip().lower().startswith("style:")]
    if idx:
        lines[idx[0]] = line
        for i in reversed(idx[1:]):           # only one Style block allowed
            del lines[i]
        return "\n".join(lines)
    return prompt.rstrip() + "\n\n" + line


async def image(base: str, style_id: str, size: int | None = None) -> Path:
    """Local path of a style's sample image (base 'portrait' etc.), fetching it once.
    With size, a JPEG thumbnail whose long edge is `size`."""
    if not re.fullmatch(r"[a-z0-9-]+", base) or not re.fullmatch(r"[a-z0-9-]+", style_id):
        raise ValueError("bad style or base id")
    orig = CACHE / base / f"{style_id}.webp"
    lock = _locks.setdefault(orig, asyncio.Lock())
    async with lock:
        if not orig.exists():
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                r = await c.get(f"{SITE}images/{base}/{style_id}.webp")
            if r.status_code != 200:
                raise FileNotFoundError(f"style image {base}/{style_id}: HTTP {r.status_code}")
            orig.parent.mkdir(parents=True, exist_ok=True)
            orig.write_bytes(r.content)
        if not size:
            return orig
        thumb = orig.with_name(f"{style_id}_{size}.jpg")
        if not thumb.exists():
            def make():
                with Image.open(orig) as im:
                    im = im.convert("RGB")
                    im.thumbnail((size, size))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=82)
                thumb.write_bytes(buf.getvalue())
            await asyncio.to_thread(make)
        return thumb
