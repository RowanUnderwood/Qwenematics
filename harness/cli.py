"""Command-line checks and smoke renders (no LLM involved).

    python -m harness.cli status                 servers up? on the expected GPUs?
    python -m harness.cli snapshot [server]      save /object_info for offline tests
    python -m harness.cli check                  validate every workflow version against live servers
    python -m harness.cli convert ui.json out.json [server]   editor workflow -> API (export fallback)
    python -m harness.cli smoke-image "prompt" [--kind character]
    python -m harness.cli smoke-video "prompt" [--mode t2va] [--pics a.png b.png] [--seeds 2] [--seconds 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from .comfy.client import ComfyClient
from .comfy.manifest import Registry
from .comfy.runner import Runner
from .comfy.ui_to_api import convert
from .config import ROOT, get_settings


def _clients(s) -> dict[str, ComfyClient]:
    return {name: ComfyClient(cfg["url"], name) for name, cfg in s.servers.items()}


async def cmd_status(args):
    s = get_settings()
    clients = _clients(s)
    for name, c in clients.items():
        cfg = s.servers[name]
        if not await c.alive():
            print(f"  {name:6} {cfg['url']:26} DOWN   (launcher: {s.server_launcher(name)})")
            continue
        st = await c.system_stats()
        gpus = ", ".join(d["name"].split(":")[1].strip() if ":" in d["name"] else d["name"] for d in st["devices"])
        ok = cfg.get("expect_gpu", "") in gpus
        print(f"  {name:6} {cfg['url']:26} UP     {gpus}{'' if ok else '   <-- expected ' + cfg.get('expect_gpu', '?')}")
        await c.close()
    llm = s.llm["base_url"].rstrip("/")
    try:
        r = httpx.get(f"{llm}/v1/models", timeout=3)
        models = [m.get("id") for m in r.json().get("data", [])]
        print(f"  llm    {llm:26} UP     {models}")
    except httpx.HTTPError:
        print(f"  llm    {llm:26} DOWN   (launcher: {s.llm.get('launcher')})")


async def cmd_snapshot(args):
    s = get_settings()
    c = ComfyClient(s.servers[args.server]["url"])
    oi = await c.object_info()
    out = ROOT / "tests" / "object_info_snapshot.json"
    out.write_text(json.dumps(oi), encoding="utf-8")
    print(f"{len(oi)} node classes -> {out}")
    await c.close()


async def _live(s):
    clients = _clients(s)
    ois = {}
    for name, c in clients.items():
        if await c.alive():
            ois[name] = await c.object_info()
    reg = Registry(s.workflows_root, s.preferred_workflows)
    reg.check_all(ois)
    return clients, ois, reg


async def cmd_check(args):
    s = get_settings()
    clients, ois, reg = await _live(s)
    for path, err in reg.errors.items():
        print(f"BROKEN  {path}: {err}")
    for wf_id, vs in reg.versions.items():
        for v, wv in vs.items():
            server = wv.manifest.get("server")
            if server not in ois:
                print(f"?       {wv.key}: server '{server}' is down, not checked")
            elif wv.problems:
                print(f"UNUSABLE {wv.key}:")
                for p in wv.problems:
                    print(f"    {p}")
            else:
                print(f"OK      {wv.key}")
    for c in clients.values():
        await c.close()


async def cmd_convert(args):
    s = get_settings()
    c = ComfyClient(s.servers[args.server]["url"])
    oi = await c.object_info()
    api = convert(json.loads(Path(args.ui).read_text(encoding="utf-8")), oi)
    Path(args.out).write_text(json.dumps(api, indent=1), encoding="utf-8")
    print(f"{len(api)} nodes -> {args.out}")
    await c.close()


def _progress(prefix):
    last = [0.0]

    def cb(e):
        now = time.time()
        if e["type"] == "progress" and now - last[0] < 2 and e["value"] != e["max"]:
            return
        last[0] = now
        if e["type"] == "progress":
            print(f"  {prefix} node {e['node']}: {e['value']}/{e['max']}")
        elif e["type"] == "queued":
            print(f"  {prefix} queued {e['prompt_id']}")
    return cb


async def cmd_smoke_image(args):
    s = get_settings()
    clients, ois, reg = await _live(s)
    wv = reg.get("krea2")
    if wv.problems:
        sys.exit(f"{wv.key} unusable: {wv.problems}")
    runner = Runner(reg, clients, ois, s.comfy_dir)
    dest = s.projects_root / "_smoke" / "media"
    params = wv.resolve_params({"prompt": args.prompt, "ref_kind": args.kind})
    t = time.time()
    res = await runner.render(wv, "t2i", params, dest_dir=dest, tag=f"smoke_{int(t)}",
                              on_progress=_progress("krea"))
    print(f"done in {time.time() - t:.0f}s: {res.outputs}  leftovers={res.leftovers}")


async def cmd_smoke_video(args):
    s = get_settings()
    clients, ois, reg = await _live(s)
    wv = reg.get("minimax_h3")
    if wv.problems:
        sys.exit(f"{wv.key} unusable: {wv.problems}")
    runner = Runner(reg, clients, ois, s.comfy_dir)
    dest = s.projects_root / "_smoke" / "media"
    pics = [Path(p) for p in args.pics]
    base = {"prompt": args.prompt, "duration_s": args.seconds}
    header = wv.prompt_header(args.mode, args.seconds)
    if header:
        base["prompt"] = f"{header}\n\n{args.prompt}"
    previews = []
    for i in range(args.seeds):
        params = wv.resolve_params(base)
        t = time.time()
        res = await runner.render(wv, "stage1", params, mode=args.mode, pictures=pics, dest_dir=dest,
                                  tag=f"smoke_s1_{int(t)}", on_progress=_progress(f"seed{i + 1}"))
        print(f"seed {params['seed']}: {time.time() - t:.0f}s {res.outputs}")
        previews.append((params, res))
    params, res = previews[args.pick - 1]
    t = time.time()
    res2 = await runner.render(wv, "stage2", params, mode=args.mode, pictures=pics, dest_dir=dest,
                               tag=f"smoke_s2_{int(t)}", on_progress=_progress("upscale"),
                               handoff={"video_latent": res.outputs["video_latent"][0],
                                        "audio_latent": res.outputs["audio_latent"][0]})
    print(f"final {time.time() - t:.0f}s: {res2.outputs} leftovers={res2.leftovers}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harness.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("snapshot"); p.add_argument("server", nargs="?", default="video")
    sub.add_parser("check")
    p = sub.add_parser("convert"); p.add_argument("ui"); p.add_argument("out"); p.add_argument("server", nargs="?", default="video")
    p = sub.add_parser("smoke-image"); p.add_argument("prompt"); p.add_argument("--kind", default="character")
    p = sub.add_parser("smoke-video"); p.add_argument("prompt")
    p.add_argument("--mode", default="t2va"); p.add_argument("--pics", nargs="*", default=[])
    p.add_argument("--seeds", type=int, default=1); p.add_argument("--pick", type=int, default=1)
    p.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    fn = {"status": cmd_status, "snapshot": cmd_snapshot, "check": cmd_check, "convert": cmd_convert,
          "smoke-image": cmd_smoke_image, "smoke-video": cmd_smoke_video}[args.cmd]
    asyncio.run(fn(args))


if __name__ == "__main__":
    main()
