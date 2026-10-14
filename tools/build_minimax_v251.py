"""Derive the harness graphs for MiniMax H3 SeedHunter v2.5.1 from the editor workflow.

    .venv\\Scripts\\python tools\\build_minimax_v251.py [object_info.json]

Reads  source_workflows/minimax_h3_v251_ui_original.json
Writes workflows/minimax_h3/v251/stage1.api.json   (one seed -> preview mp4 + saved latents)
       workflows/minimax_h3/v251/stage2.api.json   (saved latents -> upscale pass -> final mp4)

What changes relative to the editor workflow, and why:
- MediaLoader + ReferenceSplitter (browser upload widget) -> nine LoadImage nodes
  "HARNESS pic 1..9" wired to the same consumers. Ref video/audio slots are dropped for v1.
- H3 Prompt IDE is a pass-through in Python; the prompt goes straight into node 5.
- First-frame guide resizes to the *video* size (the original resized the image to its
  own size, which only works when the ref already has the video's aspect ratio).
- A last-frame guide (frame_idx -1, Picture 2) is added after the first-frame guide;
  the adapter bypasses either guide per mode.
- Stage 1 ends in LTXVSeparateAVLatent -> two SaveLatent nodes, because the joint AV
  latent is a NestedTensor that SaveLatent cannot store. Stage 2 starts from two
  LoadLatent nodes in place of the separator that fed the latent upscaler.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.comfy import graph_ops as g  # noqa: E402
from harness.comfy.ui_to_api import convert  # noqa: E402

SRC = ROOT / "source_workflows" / "minimax_h3_v251_ui_original.json"
OUT = ROOT / "workflows" / "minimax_h3" / "v251"

RES = "22:9"            # Resolution Selector inside the Video Settings subgraph
REF_TO_VIDEO = "5"
SPLITTER = "417"
SAMPLER_1 = "125:12"    # stage-1 SamplerCustomAdvanced (slot 1 = denoised latent)
FIRST_GUIDE, FIRST_RESIZE, FIRST_SIZE = "272", "274", "276"
STAGE1_GUIDER = "125:14"
AV_SEPARATE = "242"     # fed the latent upscaler from the stage-1 switch
PREVIEW_VIDEO, FINAL_VIDEO = "18", "34"


def build(object_info: dict) -> tuple[dict, dict]:
    ui = json.loads(SRC.read_text(encoding="utf-8"))
    api = convert(ui, object_info)

    # --- prompt: drop the IDE pass-through
    api[REF_TO_VIDEO]["inputs"]["prompt"] = ""
    api[REF_TO_VIDEO]["_meta"]["title"] = "HARNESS ref to video"
    for nid in ("365", "418"):
        g.remove_node(api, nid)
    api[REF_TO_VIDEO]["inputs"]["prompt"] = ""

    # --- pictures: replace the splitter's picture outputs with LoadImage slots
    for k in range(1, 10):
        pid = f"h_pic_{k}"
        g.add_node(api, pid, "LoadImage", {"image": "PLACEHOLDER"}, f"HARNESS pic {k}")
        g.relink(api, [SPLITTER, k - 1], [pid, 0])
    for nid in ("510", "511", SPLITTER, "416"):
        g.remove_node(api, nid)

    # --- first-frame guide: resize to the video size
    api[FIRST_RESIZE]["inputs"]["width"] = [RES, 0]
    api[FIRST_RESIZE]["inputs"]["height"] = [RES, 1]
    g.remove_node(api, FIRST_SIZE)
    api[FIRST_GUIDE]["_meta"]["title"] = "HARNESS first frame guide"

    # --- last-frame guide on Picture 2, chained after the first-frame guide
    fr = api[FIRST_RESIZE]["inputs"]
    g.add_node(api, "h_lf_resize", "ImageResizeKJv2",
               {**{k: v for k, v in fr.items() if k != "image"}, "image": ["h_pic_2", 0]},
               "HARNESS last frame resize")
    fg = api[FIRST_GUIDE]["inputs"]
    g.add_node(api, "h_lf_guide", "MiniMaxH3AddGuide",
               {**{k: v for k, v in fg.items() if k not in ("positive", "image", "frame_idx")},
                "positive": [FIRST_GUIDE, 0], "image": ["h_lf_resize", 0], "frame_idx": -1},
               "HARNESS last frame guide")
    api[STAGE1_GUIDER]["inputs"]["conditioning"] = ["h_lf_guide", 0]

    # --- stage 1: one seed -> preview + latents
    s1 = json.loads(json.dumps(api))
    g.add_node(s1, "h_sep", "LTXVSeparateAVLatent", {"av_latent": [SAMPLER_1, 1]}, "HARNESS separate AV")
    g.add_node(s1, "h_save_video_latent", "SaveLatent",
               {"samples": ["h_sep", 0], "filename_prefix": "harness/latent_v"}, "HARNESS save video latent")
    g.add_node(s1, "h_save_audio_latent", "SaveLatent",
               {"samples": ["h_sep", 1], "filename_prefix": "harness/latent_a"}, "HARNESS save audio latent")
    s1[PREVIEW_VIDEO]["inputs"].update(save_output=True, filename_prefix="harness/preview")
    s1[PREVIEW_VIDEO]["_meta"]["title"] = "HARNESS preview video"
    s1 = g.prune_to(s1, [PREVIEW_VIDEO, "h_save_video_latent", "h_save_audio_latent"])

    # --- stage 2: latents -> upscale pass -> final video
    s2 = json.loads(json.dumps(api))
    g.add_node(s2, "h_load_video_latent", "LoadLatent", {"latent": "PLACEHOLDER"}, "HARNESS load video latent")
    g.add_node(s2, "h_load_audio_latent", "LoadLatent", {"latent": "PLACEHOLDER"}, "HARNESS load audio latent")
    g.relink(s2, [AV_SEPARATE, 0], ["h_load_video_latent", 0])
    g.relink(s2, [AV_SEPARATE, 1], ["h_load_audio_latent", 0])
    g.remove_node(s2, AV_SEPARATE)
    s2[FINAL_VIDEO]["inputs"]["filename_prefix"] = "harness/final"
    s2[FINAL_VIDEO]["_meta"]["title"] = "HARNESS final video"
    s2 = g.prune_to(s2, [FINAL_VIDEO])
    return s1, s2


def main():
    oi_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "object_info_snapshot.json"
    oi = json.loads(oi_path.read_text(encoding="utf-8"))
    s1, s2 = build(oi)
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, graph in (("stage1", s1), ("stage2", s2)):
        errs = [e for e in g.validate(graph, oi) if "PLACEHOLDER" not in e]
        for e in errs:
            print(f"[{name}] {e}")
        ok &= not errs
        (OUT / f"{name}.api.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
        print(f"{name}: {len(graph)} nodes")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
