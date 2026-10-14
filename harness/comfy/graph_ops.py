"""Small, generic edits on API-format graphs ({id: {class_type, inputs, _meta}}).

Used both by the one-off build scripts that derive harness graphs from a source
workflow and by the adapter at render time (dropping unused reference slots,
switching guides off). Everything here works on plain dicts and never touches disk.
"""
from __future__ import annotations

import copy
from typing import Any, Iterator


def is_link(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def consumers(api: dict, node_id: str) -> Iterator[tuple[str, str, int]]:
    """(consumer_id, input_name, output_slot) for every link that reads node_id."""
    for cid, node in api.items():
        for name, v in node["inputs"].items():
            if is_link(v) and v[0] == node_id:
                yield cid, name, v[1]


def add_node(api: dict, node_id: str, class_type: str, inputs: dict, title: str | None = None) -> str:
    if node_id in api:
        raise ValueError(f"node id {node_id} already exists")
    api[node_id] = {"inputs": dict(inputs), "class_type": class_type,
                    "_meta": {"title": title or class_type}}
    return node_id


def remove_node(api: dict, node_id: str) -> None:
    """Delete a node and detach every input that read from it."""
    api.pop(node_id, None)
    for cid, name, _ in list(consumers(api, node_id)):
        del api[cid]["inputs"][name]


def relink(api: dict, old: list, new: Any) -> None:
    """Point every input that reads `old` ([id, slot]) at `new` (a link or a value)."""
    for node in api.values():
        for name, v in list(node["inputs"].items()):
            if is_link(v) and v == old:
                node["inputs"][name] = copy.deepcopy(new)


def bypass_node(api: dict, node_id: str, object_info: dict) -> None:
    """Remove node_id, wiring each consumer to the node's first input of matching type
    (the same rule ComfyUI's bypass uses). Consumers with no match are detached."""
    node = api[node_id]
    info = object_info[node["class_type"]]
    out_types = info["output"]
    in_types = {}
    for section in ("required", "optional"):
        for name, spec in info["input"].get(section, {}).items():
            in_types[name] = spec[0] if isinstance(spec[0], str) else "COMBO"
    for cid, name, slot in list(consumers(api, node_id)):
        want = out_types[slot]
        src = None
        for iname, v in node["inputs"].items():
            if is_link(v) and want in str(in_types.get(iname, "")).split(","):
                src = v
                break
        if src is None:
            del api[cid]["inputs"][name]
        else:
            api[cid]["inputs"][name] = list(src)
    del api[node_id]


def upstream(api: dict, roots: list[str]) -> set[str]:
    keep, stack = set(), [r for r in roots if r in api]
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        for v in api[k]["inputs"].values():
            if is_link(v) and v[0] in api:
                stack.append(v[0])
    return keep


def prune_to(api: dict, roots: list[str]) -> dict:
    keep = upstream(api, roots)
    out = {k: v for k, v in api.items() if k in keep}
    for v in out.values():
        for name, val in list(v["inputs"].items()):
            if is_link(val) and val[0] not in out:
                del v["inputs"][name]
    return out


def find_by_title(api: dict, title: str) -> list[str]:
    return [k for k, v in api.items() if v.get("_meta", {}).get("title") == title]


# Loader inputs whose choices are whatever files sit in ComfyUI's input folder; the
# harness uploads those right before queueing, so offline checks skip them.
FILE_INPUTS = {("LoadImage", "image"), ("LoadLatent", "latent")}


def validate(api: dict, object_info: dict, check_files: bool = False) -> list[str]:
    """Offline checks: known classes, required inputs present, links resolve, combo values valid."""
    errors = []
    for k, v in api.items():
        cls = v["class_type"]
        info = object_info.get(cls)
        if info is None:
            errors.append(f"{k}: unknown node class {cls}")
            continue
        req = info["input"].get("required", {})
        for name, spec in req.items():
            if spec[0] in ("COMFY_AUTOGROW_V3",):
                continue
            if name not in v["inputs"]:
                errors.append(f"{k} ({cls}): missing required input '{name}'")
        allspecs = {**req, **info["input"].get("optional", {})}
        for name, val in v["inputs"].items():
            if is_link(val):
                if val[0] not in api:
                    errors.append(f"{k} ({cls}): input '{name}' links to missing node {val[0]}")
                continue
            spec = allspecs.get(name)
            if spec is None or (not check_files and (cls, name) in FILE_INPUTS):
                continue
            options = spec[0] if isinstance(spec[0], list) else (
                spec[1].get("options") if spec[0] == "COMBO" and len(spec) > 1 else None)
            if options and val not in options:
                errors.append(f"{k} ({cls}): '{name}'={val!r} not available on server")
    return errors
