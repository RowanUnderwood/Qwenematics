"""Derive the harness graph for Krea 2 from the editor workflow.

    .venv\\Scripts\\python tools\\build_krea2_v1.py [object_info.json]

Reads  source_workflows/krea2_ui_original.json
Writes workflows/krea2/v1/t2i.api.json

Changes: CLIPTextEncode (PROMPT) and the inert negative prompt (CFG 1) are replaced by
KJNodes' Krea2PromptWeight, so the guide's (word:1.5) / (word:-2) weighting works.
Its patched MODEL and its CONDITIONING both feed the sampler; positive and negative
take the same tensor, as in the guide's reference workflow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.comfy import graph_ops as g  # noqa: E402
from harness.comfy.ui_to_api import convert  # noqa: E402

SRC = ROOT / "source_workflows" / "krea2_ui_original.json"
OUT = ROOT / "workflows" / "krea2" / "v1"
LORAS, PROMPT, NEGATIVE, SAMPLER, SAVE = "172", "104", "17", "138", "152"


def build(object_info: dict) -> dict:
    api = convert(json.loads(SRC.read_text(encoding="utf-8")), object_info)
    text = api[PROMPT]["inputs"]["text"]
    g.remove_node(api, PROMPT)
    g.remove_node(api, NEGATIVE)
    g.add_node(api, "h_prompt", "Krea2PromptWeight",
               {"clip": [LORAS, 1], "model": [LORAS, 0], "text": text, "strength": 1.0}, "PROMPT")
    api[SAMPLER]["inputs"].update(model=["h_prompt", 0], positive=["h_prompt", 1], negative=["h_prompt", 1])
    api[SAVE]["inputs"]["filename_prefix"] = "harness/krea"
    return g.prune_to(api, [SAVE])


def main():
    oi_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "object_info_snapshot.json"
    oi = json.loads(oi_path.read_text(encoding="utf-8"))
    api = build(oi)
    errs = g.validate(api, oi)
    for e in errs:
        print(e)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t2i.api.json").write_text(json.dumps(api, indent=1), encoding="utf-8")
    print(f"t2i: {len(api)} nodes")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
