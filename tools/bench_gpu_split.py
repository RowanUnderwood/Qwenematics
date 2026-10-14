"""A/B benchmark: Krea on the warm 3090 alone vs borrowing the 5090 (video instance).

    .venv\\Scripts\\python tools/bench_gpu_split.py [--n 4]

Measures (wall-clock, harness-side, so upload/queue/download are included):
  a. 3090 warm seconds per Krea image
  b. 5090 cold first image (after /free) and warm per image, plus VRAM in use
  c. both cards rendering at the same time (does either slow down?)
  d. swap back: /free the 5090, then a 5 s MiniMax t2va stage 1, cold vs warm
  e. Qwen per-phase seconds: Krea prompt write, critic, caption
Writes projects/_bench/results.md (+ results.json). Renders go to projects/_bench/media.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.cli import _live                       # noqa: E402
from harness.comfy.runner import Runner             # noqa: E402
from harness.config import get_settings             # noqa: E402
from harness.llm import media, roles                # noqa: E402
from harness.llm.client import LLM                  # noqa: E402

PROMPTS = [
    "an old lighthouse keeper with a white beard and a yellow oilskin coat, standing on wet rocks",
    "a young courier in a red cycling jacket leaning on a bicycle in a rainy city street",
    "a tall woman in a green wool coat holding an umbrella on a foggy bridge",
    "a boy in a blue raincoat with rubber boots standing in a puddle",
    "an elderly baker in a flour-dusted apron holding a tray of bread in a warm kitchen",
    "a mechanic in grey overalls wiping her hands next to a vintage motorcycle",
    "a street musician with a battered guitar sitting on a stone step at dusk",
    "a fisherman in a knitted sweater coiling rope on a wooden pier",
]


def gb(stats: dict) -> float:
    d = stats["devices"][0]
    return round((d["vram_total"] - d["vram_free"]) / 2**30, 1)


async def main(n: int):
    s = get_settings()
    clients, ois, reg = await _live(s)
    if set(ois) != {"image", "video"}:
        sys.exit(f"both ComfyUI servers must be up (have {list(ois)})")
    krea = reg.get("krea2")
    krea5090 = copy.copy(krea)
    krea5090.manifest = {**krea.manifest, "server": "video"}
    probs = krea5090.problems_on(ois["video"])
    if probs:
        sys.exit(f"Krea is not runnable on the video server: {probs[:3]}")
    mm = reg.get("minimax_h3")
    runner = Runner(reg, clients, ois, s.comfy_dir)
    out = s.projects_root / "_bench"
    dest = out / "media"
    R: dict = {}
    k = [0]

    async def img(wv, label: str) -> tuple[float, Path]:
        p = PROMPTS[k[0] % len(PROMPTS)]
        k[0] += 1
        params = wv.resolve_params({"prompt": p, "ref_kind": "character"})
        t = time.perf_counter()
        res = await runner.render(wv, "t2i", params, dest_dir=dest, tag=f"bench_{label}_{k[0]}")
        dt = time.perf_counter() - t
        print(f"  {label:10} {dt:6.1f}s", flush=True)
        return dt, res.outputs["image"][0]

    async def mm_stage1(label: str) -> float:
        params = mm.resolve_params({"prompt": f"{mm.prompt_header('t2va', 5) or ''}\n\nintegrated_multimodal_description: "
                                              "[Shot 1] Live-action, a static medium shot of waves breaking on a rocky shore "
                                              "under grey sky.\noverall_soundscape: Waves and wind.\nnon_diegetic_music: N/A",
                                    "duration_s": 5})
        t = time.perf_counter()
        await runner.render(mm, "stage1", params, mode="t2va", dest_dir=dest, tag=f"bench_{label}")
        dt = time.perf_counter() - t
        print(f"  {label:10} {dt:6.1f}s", flush=True)
        return dt

    print("a. 3090 warm-up + warm images", flush=True)
    await img(krea, "3090-warmup")
    a = [ (await img(krea, "3090"))[0] for _ in range(n) ]
    R["3090_warm"] = a

    print("b. 5090 cold then warm", flush=True)
    t = time.perf_counter(); await clients["video"].free(); R["free_s"] = time.perf_counter() - t
    await asyncio.sleep(3)
    R["5090_vram_after_free_gb"] = gb(await clients["video"].system_stats())
    R["5090_cold"], _ = await img(krea5090, "5090-cold")
    R["5090_warm"] = [(await img(krea5090, "5090"))[0] for _ in range(n)]
    R["5090_vram_krea_gb"] = gb(await clients["video"].system_stats())

    print("c. both at once", flush=True)
    async def run_many(wv, label, m):
        return [(await img(wv, label))[0] for _ in range(m)]
    t = time.perf_counter()
    c3090, c5090 = await asyncio.gather(run_many(krea, "3090-par", 2), run_many(krea5090, "5090-par", 4))
    R["parallel_wall_s"] = time.perf_counter() - t
    R["3090_parallel"], R["5090_parallel"] = c3090, c5090

    print("e. Qwen phases", flush=True)
    llm = LLM(s.llm["base_url"], s.llm["model"], s.llm.get("reasoning_effort", "low"))
    sample = sorted(dest.glob("bench_3090_*image*.png"))[-1]
    w, c_, cap = [], [], []
    for i in range(2):
        t = time.perf_counter()
        pr = await roles.write_krea_prompt(llm, bible="", name="Lighthouse keeper", kind="character",
                                           description=PROMPTS[i])
        w.append(time.perf_counter() - t)
        part, _ = media.image_part(sample)
        t = time.perf_counter()
        await roles.critique(llm, intent=f"character reference: {PROMPTS[0]}", prompt=pr, media=[part])
        c_.append(time.perf_counter() - t)
        part, _ = media.image_part(sample, 512)
        t = time.perf_counter()
        await roles.caption(llm, image_part=part, name="Lighthouse keeper", kind="character")
        cap.append(time.perf_counter() - t)
        print(f"  qwen write {w[-1]:.1f}s critic {c_[-1]:.1f}s caption {cap[-1]:.1f}s", flush=True)
    R["qwen_write"], R["qwen_critic"], R["qwen_caption"] = w, c_, cap
    await llm.close()

    print("d. swap back to MiniMax", flush=True)
    t = time.perf_counter(); await clients["video"].free(); R["free2_s"] = time.perf_counter() - t
    R["mm_stage1_after_krea"] = await mm_stage1("mm-cold")
    R["mm_stage1_warm"] = await mm_stage1("mm-warm")
    R["5090_vram_minimax_gb"] = gb(await clients["video"].system_stats())

    for c in clients.values():
        await c.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    report(R, out / "results.md")


def report(R: dict, path: Path):
    med = statistics.median
    t3, t5c, t5 = med(R["3090_warm"]), R["5090_cold"], med(R["5090_warm"])
    swap = R["mm_stage1_after_krea"] - R["mm_stage1_warm"] + R["free2_s"]
    lines = ["# GPU split benchmark", "",
             f"- 3090 warm: **{t3:.1f} s**/image (runs: {', '.join(f'{x:.1f}' for x in R['3090_warm'])})",
             f"- 5090 cold first image: **{t5c:.1f} s**; warm **{t5:.1f} s**/image "
             f"(runs: {', '.join(f'{x:.1f}' for x in R['5090_warm'])})",
             f"- Both at once: 3090 {', '.join(f'{x:.1f}' for x in R['3090_parallel'])} s; "
             f"5090 {', '.join(f'{x:.1f}' for x in R['5090_parallel'])} s",
             f"- 5090 VRAM in use: after free {R['5090_vram_after_free_gb']} GB, with Krea {R['5090_vram_krea_gb']} GB, "
             f"with MiniMax {R['5090_vram_minimax_gb']} GB",
             f"- Swap back: free {R['free2_s']:.1f} s; MiniMax 5 s stage 1 after Krea {R['mm_stage1_after_krea']:.1f} s vs warm "
             f"{R['mm_stage1_warm']:.1f} s -> **switch cost ~{swap:.0f} s**",
             f"- Qwen per image: write {med(R['qwen_write']):.1f} s, critic {med(R['qwen_critic']):.1f} s, "
             f"caption {med(R['qwen_caption']):.1f} s", "",
             "## Render-only makespan (seconds)", "",
             "| Images | 3090 only | 5090 only (+switch) | Best split 5090/3090 | Both (+switch) | Saved vs 3090 |",
             "|---:|---:|---:|---:|---:|---:|"]
    p3, p5 = med(R["3090_parallel"]), med(R["5090_parallel"])
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 20, 50):
        only3 = n * t3
        only5 = t5c + (n - 1) * t5 + swap
        best = min(((max(t5c + (a - 1) * p5 if a else 0, (n - a) * p3), a) for a in range(0, n + 1)))
        both = best[0] + (swap if best[1] else 0)
        lines.append(f"| {n} | {only3:.0f} | {only5:.0f} | {best[1]} / {n - best[1]} | {both:.0f} | {only3 - both:.0f} |")
    q = med(R["qwen_write"]) + med(R["qwen_critic"]) + med(R["qwen_caption"])
    lines += ["", f"Qwen time per image (write + critic + caption, one round): **{q:.0f} s**, serialized on the 4090. "
              f"With pipelining the batch can't finish faster than N x {q:.0f} s, whatever the GPUs do."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    asyncio.run(main(ap.parse_args().n))
