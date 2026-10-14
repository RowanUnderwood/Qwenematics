"""Turning project media into message parts Qwen can see, at a predictable token cost.

Measured on Ninfer (see H:/Ninfer Qwen/CLAUDE.md): an image costs
round(w/32)*round(h/32) tokens (min 64) + 2, with no downscale, so a 1400px reference
would cost ~2k tokens. We downscale to a long edge of 768 (~300-580 tokens).
Video is sampled server-side at 2 fps; we send a no-audio H.264 proxy (long edge 768,
2 fps) with an optional burned-in clock so the critic can cite times. A 15 s clip is
then ~3k tokens. In `frames` mode (backends without video input) the clip becomes N
timestamped JPEGs instead.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
STAMP_FONTS = (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf")


def image_tokens(w: int, h: int) -> int:
    return max(64, round(w / 32) * round(h / 32)) + 2


def _fit(w: int, h: int, edge: int) -> tuple[int, int]:
    s = min(1.0, edge / max(w, h, 1))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def image_part(path: Path, edge: int = 768) -> tuple[dict, int]:
    """(image_url part, estimated tokens)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = _fit(*im.size, edge)
        if (w, h) != im.size:
            im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": uri}}, image_tokens(w, h)


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, timeout=60, creationflags=NO_WINDOW)
    data = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "duration": dur, "has_audio": a is not None}


def _font() -> str | None:
    return next((f for f in STAMP_FONTS if os.path.exists(f)), None)


def video_proxy(path: Path, cache_dir: Path, edge: int = 768, fps: float = 2, stamp: bool = True) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{edge}|{fps}|{stamp}".encode()).hexdigest()[:12]
    target = cache_dir / f"{path.stem[:40]}-{key}.mp4"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    chain = ["setpts=PTS-STARTPTS", f"fps={fps:g}",
             f"scale=w='min({edge},iw)':h='min({edge},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"]
    font = _font() if stamp else None
    if font:
        f = font.replace("\\", "/").replace(":", r"\:")
        chain.append(f"drawtext=fontfile={f}:text='%{{pts\\:hms}}':fontcolor=white:fontsize=h/18:"
                     "box=1:boxcolor=black@0.6:boxborderw=8:x=w*0.02:y=h*0.02")
    tmp = target.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-an", "-sn", "-dn", "-vf", ",".join(chain),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, timeout=900, creationflags=NO_WINDOW)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg proxy failed: " + r.stderr.decode("utf-8", "replace")[-800:])
    os.replace(tmp, target)
    return target


def video_part(path: Path, cache_dir: Path, edge: int = 768, fps: float = 2) -> tuple[dict, int]:
    """(video_url part, estimated tokens) for Ninfer's native video input."""
    proxy = video_proxy(path, cache_dir, edge, fps)
    info = probe(proxy)
    per_frame = image_tokens(info["width"], info["height"])
    # the server merges frame pairs temporally; ~half a frame's tokens per sampled frame
    est = int(per_frame * max(1, info["duration"] * fps) / 2) + 16
    uri = "data:video/mp4;base64," + base64.b64encode(proxy.read_bytes()).decode("ascii")
    return {"type": "video_url", "video_url": {"url": uri}}, est


def frame_parts(path: Path, cache_dir: Path, n: int = 8, edge: int = 768) -> tuple[list[dict], int]:
    """`frames` mode: N evenly spaced stills, each preceded by its timestamp."""
    info = probe(path)
    dur = max(info["duration"], 0.1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts, total = [], 0
    for i in range(n):
        t = dur * (i + 0.5) / n
        out = cache_dir / f"{path.stem[:40]}-f{i}-{int(t * 1000)}.jpg"
        if not out.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                            "-q:v", "3", str(out)], capture_output=True, timeout=120, creationflags=NO_WINDOW)
        if out.exists():
            part, tok = image_part(out, edge)
            parts += [{"type": "text", "text": f"[t={t:05.2f}s]"}, part]
            total += tok + 8
    return parts, total


def last_frame(path: Path, dest: Path) -> Path:
    """Grab the final frame of a clip (used to chain shots: last frame -> next first frame)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(path), "-frames:v", "1",
                    "-q:v", "2", str(dest)], capture_output=True, timeout=120, creationflags=NO_WINDOW)
    if not dest.exists():
        raise RuntimeError(f"could not extract last frame of {path}")
    return dest
