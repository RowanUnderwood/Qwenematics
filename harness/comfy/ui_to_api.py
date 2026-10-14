"""Convert a ComfyUI *editor* workflow (ui.json) into the API prompt format.

The browser export does this with JavaScript that runs extension hooks; when an
extension breaks it (the MiniMax SeedHunter workflow), the export silently fails.
This module reproduces the parts of graphToPrompt our workflows rely on, driven by
the server's /object_info so widget order comes from the real node definitions:

- subgraphs are flattened, inner ids become "outer:inner" like the frontend does
- muted nodes (mode 2) drop out; bypassed nodes (mode 4) pass inputs through by type
- Reroute / PrimitiveNode / Note-style frontend-only nodes are resolved or dropped
- Anything Everywhere links (extra.ue_links) are materialised as real links
- widgets_values are mapped to input names, skipping control_after_generate values
  and expanding dynamic combos ("mode" -> "mode.scale")
- PixaromaLoraLoader's browser-held state is written into its hidden input

Group/toggle switches (PixaromaGroupSwitch etc.) need no special handling: they work
by changing node modes, and the saved modes already reflect their current state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MUTED, BYPASS = 2, 4
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
FRONTEND_PASSTHROUGH = {"Reroute"}
FRONTEND_PRIMITIVE = {"PrimitiveNode"}


class ConversionError(Exception):
    pass


@dataclass
class Node:
    gid: str                      # global id ("5", "106:13", ...)
    ui: dict
    mode: int
    is_subgraph: bool = False
    # input name -> source; a source is ("node", gid, slot) or ("value", v)
    links: dict[str, tuple] = field(default_factory=dict)
    # outputs of a subgraph instance: slot -> source tuple (filled from inner links)
    sg_outputs: dict[int, tuple] = field(default_factory=dict)


def _link_tuple(link) -> tuple:
    """(id, origin_id, origin_slot, target_id, target_slot, type) for list or dict links."""
    if isinstance(link, dict):
        return (link["id"], link["origin_id"], link["origin_slot"],
                link["target_id"], link["target_slot"], link.get("type"))
    return tuple(link[:6])


def _is_widget_spec(spec: list) -> bool:
    typ = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if opts.get("forceInput"):
        return False
    return isinstance(typ, list) or typ in WIDGET_TYPES or typ == "COMFY_DYNAMICCOMBO_V3"


def _ordered_inputs(info: dict) -> list[tuple[str, list]]:
    inp = info.get("input", {})
    order = info.get("input_order") or {k: list(v) for k, v in inp.items()}
    out = []
    for section in ("required", "optional"):
        for name in order.get(section, []):
            spec = inp.get(section, {}).get(name)
            if spec is not None:
                out.append((name, spec))
    return out


def map_widgets(info: dict, values: Any) -> dict[str, Any]:
    """Map a node's widgets_values onto input names using its /object_info entry."""
    if values is None:
        return {}
    widgets = [(n, s) for n, s in _ordered_inputs(info) if _is_widget_spec(s)]
    if isinstance(values, dict):          # VHS-style named widgets
        return {n: values[n] for n, _ in widgets if n in values}
    result: dict[str, Any] = {}
    vals = list(values)
    i = 0

    def take():
        nonlocal i
        if i >= len(vals):
            raise IndexError
        v = vals[i]
        i += 1
        return v

    def skip_control(spec):
        nonlocal i
        typ = spec[0]
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if typ == "INT" and i < len(vals) and (
                opts.get("control_after_generate") or vals[i] in CONTROL_VALUES):
            if vals[i] in CONTROL_VALUES:
                i += 1

    try:
        for name, spec in widgets:
            if spec[0] == "COMFY_DYNAMICCOMBO_V3":
                key = take()
                result[name] = key
                for opt in spec[1].get("options", []):
                    if opt["key"] != key:
                        continue
                    sub = opt.get("inputs", {})
                    for section in ("required", "optional"):
                        for sname, sspec in sub.get(section, {}).items():
                            if _is_widget_spec(sspec):
                                result[f"{name}.{sname}"] = take()
                                skip_control(sspec)
                continue
            result[name] = take()
            skip_control(spec)
    except IndexError:
        pass                               # fewer values than widgets: keep what we have
    return result


class Converter:
    def __init__(self, ui: dict, object_info: dict):
        self.ui = ui
        self.oi = object_info
        self.subgraphs = {s["id"]: s for s in ui.get("definitions", {}).get("subgraphs", [])}
        self.nodes: dict[str, Node] = {}

    # ---------------------------------------------------------------- flatten
    def _add_graph(self, nodes: list[dict], links: list, prefix: str,
                   parent: Node | None, sg_def: dict | None, extra_links: list = ()):
        by_local: dict[Any, Node] = {}
        for n in nodes:
            gid = f"{prefix}{n['id']}"
            mode = n.get("mode", 0)
            if parent is not None and parent.mode in (MUTED, BYPASS):
                mode = MUTED                # inner nodes of a muted/bypassed subgraph never run
            node = Node(gid, n, mode, is_subgraph=n["type"] in self.subgraphs)
            self.nodes[gid] = node
            by_local[n["id"]] = node

        def src_of(origin_id, origin_slot):
            if origin_id == -10:            # subgraph input boundary
                return parent.links.get(f"__sg_in_{origin_slot}") if parent else None
            o = by_local.get(origin_id)
            return ("node", o.gid, origin_slot) if o else None

        for raw in list(links) + list(extra_links):
            _, oid, oslot, tid, tslot, _typ = _link_tuple(raw)
            src = src_of(oid, oslot)
            if tid == -20:                  # subgraph output boundary
                if parent is not None and src is not None:
                    parent.sg_outputs[tslot] = src
                continue
            tgt = by_local.get(tid)
            if tgt is None or src is None:
                continue
            ins = tgt.ui.get("inputs", [])
            if tslot < len(ins):
                tgt.links.setdefault(ins[tslot]["name"], src)

        # recurse into subgraph instances after their outer links are known
        for n in nodes:
            node = by_local[n["id"]]
            if node.is_subgraph:
                self._expand_subgraph(node)

    def _expand_subgraph(self, node: Node):
        sg = self.subgraphs[node.ui["type"]]
        sg_inputs = sg.get("inputs", [])
        # Promoted widget values on the instance, in subgraph-input order, for inputs
        # that feed an inner widget.
        inner_by_id = {n["id"]: n for n in sg["nodes"]}
        inner_links = {_link_tuple(l)[0]: _link_tuple(l) for l in sg.get("links", [])}
        widget_slots = []
        for k, si in enumerate(sg_inputs):
            is_widget = False
            for lid in si.get("linkIds", []):
                lt = inner_links.get(lid)
                if not lt:
                    continue
                tgt = inner_by_id.get(lt[3])
                if tgt and lt[4] < len(tgt.get("inputs", [])) and "widget" in tgt["inputs"][lt[4]]:
                    is_widget = True
            if is_widget:
                widget_slots.append(k)
        wv = node.ui.get("widgets_values") or []
        for idx, k in enumerate(widget_slots):
            if idx < len(wv) and wv[idx] is not None:
                node.links[f"__sg_in_{k}"] = ("value", wv[idx])
        # outer links override promoted widget values
        for k, si in enumerate(sg_inputs):
            if si["name"] in node.links:
                node.links[f"__sg_in_{k}"] = node.links[si["name"]]
        self._add_graph(sg["nodes"], sg.get("links", []), f"{node.gid}:", node, sg)

    # ---------------------------------------------------------------- resolve
    def _out_type(self, node: Node, slot: int):
        outs = node.ui.get("outputs", [])
        return outs[slot].get("type") if slot < len(outs) else None

    def resolve(self, src, depth=0):
        """Follow a source through subgraph boundaries, reroutes and bypasses."""
        if src is None or src[0] == "value" or depth > 200:
            return src
        _, gid, slot = src
        node = self.nodes.get(gid)
        if node is None:
            return None
        if node.mode == MUTED:
            return None
        if node.mode == BYPASS:
            return self.resolve(self._bypass_source(node, slot), depth + 1)
        if node.is_subgraph:
            return self.resolve(node.sg_outputs.get(slot), depth + 1)
        cls = node.ui["type"]
        if cls in FRONTEND_PASSTHROUGH:
            ins = node.ui.get("inputs", [])
            return self.resolve(node.links.get(ins[0]["name"]) if ins else None, depth + 1)
        if cls in FRONTEND_PRIMITIVE:
            wv = node.ui.get("widgets_values") or [None]
            return ("value", wv[0])
        if cls not in self.oi:
            return None
        return src

    def _bypass_source(self, node: Node, slot: int):
        want = self._out_type(node, slot)
        ins = node.ui.get("inputs", [])

        def matches(inp, wildcard=False):
            types = str(inp.get("type", "")).split(",")
            if wildcard:
                return "*" in types or want == "*"
            return want in types

        def linked(inp):
            return inp["name"] in node.links

        if slot < len(ins) and matches(ins[slot]) and linked(ins[slot]):
            return node.links[ins[slot]["name"]]
        for wildcard in (False, True):
            for inp in ins:
                if matches(inp, wildcard) and linked(inp):
                    return node.links[inp["name"]]
        return None

    # ---------------------------------------------------------------- build
    def convert(self, outputs: list[str] | None = None) -> dict:
        extra = []
        for ue in self.ui.get("extra", {}).get("ue_links", []) or []:
            extra.append([None, int(ue["upstream"]), ue["upstream_slot"],
                          int(ue["downstream"]), ue["downstream_slot"], ue.get("type")])
        self._add_graph(self.ui["nodes"], self.ui.get("links", []), "", None, None, extra)

        api: dict[str, dict] = {}
        for gid, node in self.nodes.items():
            cls = node.ui["type"]
            if node.mode in (MUTED, BYPASS) or node.is_subgraph or cls not in self.oi:
                continue
            info = self.oi[cls]
            inputs = map_widgets(info, node.ui.get("widgets_values"))
            if "LoraLoaderState" in info.get("input", {}).get("hidden", {}):
                inputs["LoraLoaderState"] = _pixaroma_lora_state(node.ui)
                inputs = {k: v for k, v in inputs.items() if k == "LoraLoaderState"}
            for name, src in node.links.items():
                if name.startswith("__sg_in_"):
                    continue
                r = self.resolve(src)
                if r is None:
                    # a dead link on a converted widget falls back to the widget value
                    if not _is_widget_input(node.ui, name):
                        inputs.pop(name, None)
                    continue
                inputs[name] = r[1] if r[0] == "value" else [r[1], r[2]]
            entry = {"inputs": inputs, "class_type": cls}
            title = node.ui.get("title")
            entry["_meta"] = {"title": title or info.get("display_name", cls)}
            api[gid] = entry
        return prune(api, self.oi, outputs)


def _is_widget_input(ui_node: dict, name: str) -> bool:
    return any(i["name"] == name and "widget" in i for i in ui_node.get("inputs", []))


def _pixaroma_lora_state(ui_node: dict) -> str:
    state = None
    wv = ui_node.get("widgets_values")
    if wv and isinstance(wv[0], dict):
        state = wv[0]
    if state is None:
        for v in (ui_node.get("properties") or {}).values():
            if isinstance(v, str) and '"loras"' in v:
                state = json.loads(v)
    state = state or {"loras": []}
    return json.dumps({
        "version": 1, "sep": state.get("sep", ", "), "cacheMode": state.get("cacheMode", "last"),
        "loras": [{"name": e.get("name", ""), "on": bool(e.get("on")), "sm": e.get("sm", 1),
                   "sc": e.get("sc", 1), "triggers": e.get("triggers", [])}
                  for e in state.get("loras", []) if e.get("name")],
    })


def prune(api: dict, object_info: dict, outputs: list[str] | None = None) -> dict:
    """Keep only nodes upstream of the chosen (or all) output nodes."""
    if outputs is None:
        outputs = [k for k, v in api.items() if object_info.get(v["class_type"], {}).get("output_node")]
    keep, stack = set(), [o for o in outputs if o in api]
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        for v in api[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api:
                stack.append(v[0])
    pruned = {k: v for k, v in api.items() if k in keep}
    # drop links that point at nodes which were removed
    for v in pruned.values():
        for name, val in list(v["inputs"].items()):
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0] not in pruned:
                del v["inputs"][name]
    return pruned


def convert(ui: dict, object_info: dict, outputs: list[str] | None = None) -> dict:
    return Converter(ui, object_info).convert(outputs)
