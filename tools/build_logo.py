"""Render frontend/public/logo.svg into the PNG and Windows icon, and (re)make the launcher shortcut.

    python tools/build_logo.py              # logo.png + assets/qwenematics.ico
    python tools/build_logo.py --shortcut   # also Qwenematics.lnk in the project folder

Needs ImageMagick 7 (`magick`) on PATH.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "frontend" / "public" / "logo.svg"
PNG = ROOT / "frontend" / "public" / "logo.png"
ICO = ROOT / "assets" / "qwenematics.ico"
LNK = ROOT / "Qwenematics.lnk"
BAT = ROOT / "start_qwenematics.bat"
ICO_SIZES = (16, 24, 32, 48, 64, 256)


def render(size: int, out: Path):
    # Rasterise big, then downsample: ImageMagick's SVG renderer antialiases poorly at small sizes.
    subprocess.run(["magick", "-background", "none", "-density", "1536", str(SVG),
                    "-resize", f"{size}x{size}", str(out)], check=True)


def build():
    render(256, PNG)
    ICO.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        frames = []
        for s in ICO_SIZES:
            render(s, Path(td) / f"{s}.png")
            frames.append(str(Path(td) / f"{s}.png"))
        subprocess.run(["magick", *frames, str(ICO)], check=True)
    print(f"wrote {PNG.relative_to(ROOT)} and {ICO.relative_to(ROOT)}")


def shortcut():
    ps = (f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{LNK}'); "
          f"$s.TargetPath = '{BAT}'; $s.WorkingDirectory = '{ROOT}'; $s.IconLocation = '{ICO},0'; "
          "$s.WindowStyle = 7; $s.Description = 'Qwenematics'; $s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"wrote {LNK.name}")


if __name__ == "__main__":
    build()
    if "--shortcut" in sys.argv:
        shortcut()
