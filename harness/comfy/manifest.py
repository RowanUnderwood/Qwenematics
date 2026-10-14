"""Workflow registry: every workflows/<id>/<version>/manifest.yaml plus its API graphs.

Adding a workflow version = adding a folder. Old versions stay loadable so anything
rendered with them can be re-rendered the same way; renders record "<id>@<version>".
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import graph_ops as g

SEED_MAX = 2**50


class WorkflowError(Exception):
    pass


@dataclass
class WorkflowVersion:
    id: str
    version: str
    path: Path
    manifest: dict
    graphs: dict[str, dict]
    problems: list[str] = field(default_factory=list)   # filled by check_against_server

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def usable(self) -> bool:
        return not self.problems

    @property
    def params(self) -> dict[str, dict]:
        return self.manifest.get("params", {})

    @property
    def modes(self) -> dict[str, dict]:
        return self.manifest.get("modes", {})

    # ------------------------------------------------------------ parameters
    def defaults(self) -> dict[str, Any]:
        return {k: copy.deepcopy(p["default"]) for k, p in self.params.items() if "default" in p}

    def qwen_params(self) -> dict[str, dict]:
        return {k: p for k, p in self.params.items() if p.get("qwen")}

    def resolve_params(self, given: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        """defaults < values implied by enum `sets` (e.g. ref_kind) < explicit values."""
        context = context or {}
        out = self.defaults()
        implied: dict[str, Any] = {}
        for name, spec in self.params.items():
            sets = spec.get("sets")
            if not sets:
                continue
            choice = given.get(name, out.get(name))
            for k, v in (sets.get(choice) or {}).items():
                if isinstance(v, str) and v.startswith("@"):
                    v = context.get(v[1:], out.get(k))
                implied[k] = v
        out.update(implied)
        out.update({k: v for k, v in given.items() if v is not None})
        for name, spec in self.params.items():
            if spec.get("type") == "seed" and out.get(name) in (None, -1):
                out[name] = random.randrange(SEED_MAX)
        return out

    def choices(self, name: str, object_info: dict | None) -> list | None:
        spec = self.params[name]
        if "choices" in spec:
            return spec["choices"]
        src = spec.get("choices_from")
        if src and object_info:
            for graph in self.graphs.values():
                node = graph.get(src["node"])
                if node:
                    info = object_info.get(node["class_type"], {}).get("input", {})
                    for section in ("required", "optional"):
                        s = info.get(section, {}).get(src["input"])
                        if s:
                            return s[0] if isinstance(s[0], list) else s[1].get("options")
        return None

    # ------------------------------------------------------------ graph build
    def build(self, graph: str, params: dict[str, Any], *, mode: str | None = None,
              pictures: list[str] = (), handoff: dict[str, str] | None = None,
              object_info: dict | None = None, filename_prefix: str | None = None) -> dict:
        """Return a ready-to-queue API prompt for one graph of this workflow."""
        if graph not in self.graphs:
            raise WorkflowError(f"{self.key} has no graph '{graph}'")
        api = copy.deepcopy(self.graphs[graph])

        for name, value in params.items():
            spec = self.params.get(name)
            if not spec or spec.get("virtual"):
                continue
            for t in spec.get("targets", []):
                node = api.get(t["node"])
                if node is None:
                    continue
                if spec.get("type") == "lora":
                    lname, strength = (value.get("name"), value.get("strength", 1.0)) if isinstance(value, dict) else (value, None)
                    node["inputs"][t["input"]] = lname or "None"
                    if strength is not None and "strength" in t:
                        node["inputs"][t["strength"]] = strength
                elif spec.get("type") == "pixaroma_loras" and not isinstance(value, str):
                    node["inputs"][t["input"]] = json.dumps(value)
                else:
                    node["inputs"][t["input"]] = value

        # picture slots: fill in order, drop the rest
        slots = self.manifest.get("slots", {}).get("pictures")
        if slots:
            if len(pictures) > slots["max"]:
                raise WorkflowError(f"{len(pictures)} pictures given, {self.key} takes at most {slots['max']}")
            for i, node_id in enumerate(slots["nodes"]):
                if node_id not in api:
                    continue
                if i < len(pictures):
                    api[node_id]["inputs"][slots.get("input", "image")] = pictures[i]
                else:
                    g.remove_node(api, node_id)

        if mode is not None:
            mspec = self.modes.get(mode)
            if mspec is None:
                raise WorkflowError(f"{self.key} has no mode '{mode}' (has {sorted(self.modes)})")
            lim = mspec.get("pictures", {})
            if not lim.get("min", 0) <= len(pictures) <= lim.get("max", 99):
                raise WorkflowError(f"mode {mode} needs {lim.get('min', 0)}-{lim.get('max')} pictures, got {len(pictures)}")
            for op in mspec.get("ops", []):
                if "bypass" in op and op["bypass"] in api:
                    if object_info is None:
                        raise WorkflowError("bypass ops need the server's object_info")
                    g.bypass_node(api, op["bypass"], object_info)
                elif "link" in op:
                    l = op["link"]
                    if l["node"] in api and l["to"][0] in api:
                        api[l["node"]]["inputs"][l["input"]] = list(l["to"])

        for key, filename in (handoff or {}).items():
            h = self.manifest.get("handoff", {}).get(key)
            if h and h["node"] in api:
                api[h["node"]]["inputs"][h["input"]] = filename

        outputs = self.manifest.get("outputs", {}).get(graph, {})
        if filename_prefix:
            for name, o in outputs.items():
                node = api.get(o["node"])
                if node and "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = f"{filename_prefix}_{name}"
        return g.prune_to(api, [o["node"] for o in outputs.values()])

    def prompt_header(self, mode: str, duration: float, last_shot: int = 1) -> str | None:
        h = self.modes.get(mode, {}).get("header")
        return h.format(duration=f"{duration:.2f}", last_shot=last_shot) if h else None

    # ------------------------------------------------------------ checks
    def check_against_server(self, object_info: dict) -> list[str]:
        """Disable this version if the server lacks a node, model, or LoRA it needs."""
        self.problems = self.problems_on(object_info)
        return self.problems

    def problems_on(self, object_info: dict) -> list[str]:
        """What would stop this version running on a server with this object_info (no side effects)."""
        problems = []
        for gname, graph in self.graphs.items():
            for e in g.validate(graph, object_info):
                if "PLACEHOLDER" not in e:
                    problems.append(f"[{gname}] {e}")
        for name, spec in self.params.items():
            for t in spec.get("targets", []):
                if not any(t["node"] in gr for gr in self.graphs.values()):
                    problems.append(f"param {name}: node {t['node']} not in any graph")
            ch = self.choices(name, object_info) if "choices_from" in spec else None
            if ch and "default" in spec and spec["default"] not in ch:
                problems.append(f"param {name}: default {spec['default']!r} not on server")
        return problems


def load_version(path: Path) -> WorkflowVersion:
    m = yaml.safe_load((path / "manifest.yaml").read_text(encoding="utf-8"))
    graphs = {name: json.loads((path / f).read_text(encoding="utf-8"))
              for name, f in m.get("graphs", {}).items()}
    return WorkflowVersion(m["id"], str(m["version"]), path, m, graphs)


class Registry:
    def __init__(self, root: Path, preferred: dict[str, str] | None = None):
        self.root = root
        self.preferred = preferred or {}
        self.versions: dict[str, dict[str, WorkflowVersion]] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self):
        self.versions.clear()
        self.errors.clear()
        for mf in sorted(self.root.glob("*/*/manifest.yaml")):
            try:
                wv = load_version(mf.parent)
                self.versions.setdefault(wv.id, {})[wv.version] = wv
            except Exception as e:      # a broken drop-in must not take the others down
                self.errors[str(mf.parent)] = f"{type(e).__name__}: {e}"

    def get(self, wf_id: str, version: str | None = None) -> WorkflowVersion:
        vs = self.versions.get(wf_id)
        if not vs:
            raise WorkflowError(f"unknown workflow {wf_id}")
        if version:
            if version not in vs:
                raise WorkflowError(f"{wf_id} has no version {version} (has {sorted(vs)})")
            return vs[version]
        pref = self.preferred.get(wf_id)
        if pref and pref in vs and vs[pref].usable:
            return vs[pref]
        usable = [v for k, v in sorted(vs.items()) if v.usable]
        return (usable or list(vs.values()))[-1]

    def get_key(self, key: str) -> WorkflowVersion:
        wf_id, _, version = key.partition("@")
        return self.get(wf_id, version or None)

    def check_all(self, object_info_by_server: dict[str, dict]):
        for vs in self.versions.values():
            for wv in vs.values():
                oi = object_info_by_server.get(wv.manifest.get("server"))
                if oi is not None:
                    wv.check_against_server(oi)
