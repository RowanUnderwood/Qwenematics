"""One render on a ComfyUI server, with the harness's file discipline:

inputs are uploaded under unique names, outputs are downloaded into the project, and
everything the render left in ComfyUI's shared input/ and output/ folders is deleted
afterwards (or reported, if deletion fails) so nothing accumulates there.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import ComfyClient, OutputFile, ProgressCb
from .manifest import Registry, WorkflowVersion

log = logging.getLogger(__name__)


@dataclass
class RenderResult:
    workflow: str                       # "<id>@<version>"
    graph: str
    params: dict[str, Any]
    prompt: dict                        # the exact API prompt queued
    prompt_id: str
    outputs: dict[str, list[Path]]      # manifest output name -> local files
    leftovers: list[Path] = field(default_factory=list)   # ComfyUI-side files we could not delete


class Runner:
    def __init__(self, registry: Registry, clients: dict[str, ComfyClient],
                 object_info: dict[str, dict], comfy_dir: Path):
        self.registry = registry
        self.clients = clients
        self.object_info = object_info
        self.input_dir = comfy_dir / "input"
        self.output_dir = comfy_dir / "output"
        self.temp_dir = comfy_dir / "temp"

    def client_for(self, wv: WorkflowVersion) -> ComfyClient:
        return self.clients[wv.manifest["server"]]

    async def render(self, wv: WorkflowVersion, graph: str, params: dict[str, Any], *,
                     dest_dir: Path, tag: str, mode: str | None = None,
                     pictures: list[Path] = (), handoff: dict[str, Path] | None = None,
                     on_progress: ProgressCb | None = None, cleanup: bool = True,
                     server: str | None = None) -> RenderResult:
        """server overrides the manifest's server (e.g. Krea on a borrowed video GPU)."""
        server = server or wv.manifest["server"]
        client = self.clients[server]
        uploaded: list[Path] = []
        try:
            pic_names = []
            for i, p in enumerate(pictures, 1):
                name = await client.upload(p, subfolder="harness", name=f"{tag}_pic{i}{p.suffix.lower()}")
                pic_names.append(name)
                uploaded.append(self.input_dir / name)
            hand_names = {}
            for key, p in (handoff or {}).items():
                # LoadLatent only lists the top level of input/
                name = await client.upload(p, subfolder="", name=f"{tag}_{key}{p.suffix}")
                hand_names[key] = name
                uploaded.append(self.input_dir / name)

            prompt = wv.build(graph, params, mode=mode, pictures=pic_names, handoff=hand_names,
                              object_info=self.object_info.get(server),
                              filename_prefix=f"harness/{tag}")
            prompt_id, files = await client.run(prompt, on_progress=on_progress)

            outputs: dict[str, list[Path]] = {}
            by_node = {o["node"]: name for name, o in wv.manifest.get("outputs", {}).get(graph, {}).items()}
            comfy_side: list[Path] = []
            for f in files:
                comfy_side += self._comfy_paths(f)
                name = by_node.get(f.node)
                if name is None:
                    continue
                dest = dest_dir / f"{tag}_{name}{Path(f.filename).suffix}"
                n = 1
                while dest.exists():
                    n += 1
                    dest = dest_dir / f"{tag}_{name}_{n}{Path(f.filename).suffix}"
                await client.download(f, dest)
                outputs.setdefault(name, []).append(dest)
            leftovers = self._delete(comfy_side + uploaded) if cleanup else comfy_side + uploaded
            return RenderResult(wv.key, graph, params, prompt, prompt_id, outputs, leftovers)
        except Exception:
            if cleanup:
                self._delete(uploaded)
            raise

    def _comfy_paths(self, f: OutputFile) -> list[Path]:
        """Every file ComfyUI wrote for this output. VHS_VideoCombine writes 'x.mp4' (silent),
        'x-audio.mp4' (reported) and 'x.png' (workflow metadata) side by side."""
        base = {"output": self.output_dir, "temp": self.temp_dir, "input": self.input_dir}.get(f.type, self.output_dir)
        folder = base / f.subfolder if f.subfolder else base
        paths = [folder / f.filename]
        stem, suffix = Path(f.filename).stem, Path(f.filename).suffix
        if stem.endswith("-audio"):
            paths.append(folder / f"{stem[:-len('-audio')]}{suffix}")
        if f.extra.get("workflow"):
            paths.append(folder / f.extra["workflow"])
        return paths

    def _delete(self, paths: list[Path]) -> list[Path]:
        left = []
        for p in paths:
            try:
                if p.is_file():
                    p.unlink()
            except OSError as e:
                log.warning("could not delete %s: %s", p, e)
                left.append(p)
        return left


def copy_into(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest
