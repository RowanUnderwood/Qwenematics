"""Offline tests for the UI->API converter and the workflow adapter.

They run against tests/object_info_snapshot.json (taken from the live ComfyUI), so no
server is needed. Refresh the snapshot with `python -m harness.cli snapshot`.
"""
import json
from pathlib import Path

import pytest

from harness.comfy import graph_ops as g
from harness.comfy.manifest import Registry, WorkflowError
from harness.comfy.ui_to_api import convert

ROOT = Path(__file__).resolve().parents[1]
OI = json.loads((ROOT / "tests" / "object_info_snapshot.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reg():
    return Registry(ROOT / "workflows", {"krea2": "v1", "minimax_h3": "v251"})


def test_converter_matches_browser_export():
    ui = json.loads((ROOT / "source_workflows" / "krea2_ui_original.json").read_text(encoding="utf-8"))
    ref = json.loads((ROOT / "source_workflows" / "krea2_api_export_original.json").read_text(encoding="utf-8"))
    api = convert(ui, OI)
    assert set(api) == set(ref)
    for k in ref:
        a, b = dict(api[k]["inputs"]), dict(ref[k]["inputs"])
        a.pop("filename_prefix", None), b.pop("filename_prefix", None)   # browser expands %date%
        assert a == b, k


def test_minimax_conversion_is_complete():
    ui = json.loads((ROOT / "source_workflows" / "minimax_h3_v251_ui_original.json").read_text(encoding="utf-8"))
    api = convert(ui, OI)
    assert g.validate(api, OI) == []
    # stage-2 guider gets conditioning through the bypassed RefModApply node
    assert api["135:29"]["inputs"]["conditioning"] == ["471", 0]


def test_registry_versions_usable(reg):
    for wf in ("krea2", "minimax_h3"):
        wv = reg.get(wf)
        assert wv.check_against_server(OI) == [], wv.problems


def test_krea_ref_kind_sets_aspect(reg):
    wv = reg.get("krea2")
    p = wv.resolve_params({"prompt": "a knight", "ref_kind": "character"})
    assert p["aspect_ratio"] == "9:16 (Portrait Widescreen)"
    p = wv.resolve_params({"ref_kind": "first_frame"}, context={"video_aspect": "21:9 (Ultrawide)"})
    assert p["aspect_ratio"] == "21:9 (Ultrawide)"
    p = wv.resolve_params({"ref_kind": "prop", "aspect_ratio": "4:3 (Standard)"})
    assert p["aspect_ratio"] == "4:3 (Standard)"           # explicit beats implied
    api = wv.build("t2i", p, object_info=OI, filename_prefix="harness/x")
    assert api["166"]["inputs"]["aspect_ratio"] == "4:3 (Standard)"
    assert api["152"]["inputs"]["filename_prefix"] == "harness/x_image"
    assert isinstance(api["138"]["inputs"]["seed"], int)
    assert g.validate(api, OI) == []


def test_krea_lora(reg):
    wv = reg.get("krea2")
    p = wv.resolve_params({"lora_1": {"name": "KNPV3_1.safetensors", "strength": 0.7}})
    api = wv.build("t2i", p, object_info=OI)
    assert api["172"]["inputs"]["lora_01"] == "KNPV3_1.safetensors"
    assert api["172"]["inputs"]["strength_01"] == 0.7


def _pics(n):
    return [f"harness/p{i}.png" for i in range(1, n + 1)]


def test_minimax_modes(reg):
    wv = reg.get("minimax_h3")
    p = wv.resolve_params({"prompt": "x", "duration_s": 15})

    t2v = wv.build("stage1", p, mode="t2va", object_info=OI)
    assert not any(k.startswith("h_pic") for k in t2v)
    assert "272" not in t2v and "h_lf_guide" not in t2v
    assert t2v["125:14"]["inputs"]["conditioning"] == ["471", 0]
    assert t2v["22:23"]["inputs"]["value"] == 15

    r2v = wv.build("stage1", p, mode="ref2va", pictures=_pics(3), object_info=OI)
    assert [k for k in r2v if k.startswith("h_pic")] == ["h_pic_1", "h_pic_2", "h_pic_3"]
    assert "ref_images.ref_image_3" not in r2v["5"]["inputs"]

    i2v = wv.build("stage1", p, mode="i2va", pictures=_pics(2), object_info=OI)
    assert "272" in i2v and "h_lf_guide" not in i2v
    assert i2v["125:14"]["inputs"]["conditioning"] == ["272", 0]

    fl = wv.build("stage1", p, mode="fl2va", pictures=_pics(2), object_info=OI)
    assert fl["125:14"]["inputs"]["conditioning"] == ["h_lf_guide", 0]
    assert fl["h_lf_resize"]["inputs"]["image"] == ["h_pic_2", 0]

    l2v = wv.build("stage1", p, mode="l2va", pictures=_pics(1), object_info=OI)
    assert "272" not in l2v
    assert l2v["h_lf_resize"]["inputs"]["image"] == ["h_pic_1", 0]
    assert l2v["h_lf_guide"]["inputs"]["positive"] == ["471", 0]

    for api in (t2v, r2v, i2v, fl, l2v):
        assert [e for e in g.validate(api, OI) if "PLACEHOLDER" not in e] == []

    with pytest.raises(WorkflowError):
        wv.build("stage1", p, mode="fl2va", pictures=_pics(1), object_info=OI)


def test_minimax_stage2_handoff(reg):
    wv = reg.get("minimax_h3")
    p = wv.resolve_params({"prompt": "x"})
    s2 = wv.build("stage2", p, mode="ref2va", pictures=_pics(1), object_info=OI,
                  handoff={"video_latent": "a_v.latent", "audio_latent": "a_a.latent"})
    assert s2["h_load_video_latent"]["inputs"]["latent"] == "a_v.latent"
    assert s2["243"]["inputs"]["latent"] == ["h_load_video_latent", 0]
    assert s2["244"]["inputs"]["audio_latent"] == ["h_load_audio_latent", 0]
    assert "34" in s2


def test_prompt_header(reg):
    wv = reg.get("minimax_h3")
    assert wv.prompt_header("fl2va", 15.083, 2).endswith("aligns with the 15.08-second mark of the target video.")
    assert wv.prompt_header("ref2va", 15) is None


def test_vhs_cleanup_paths(tmp_path):
    from harness.comfy.client import OutputFile
    from harness.comfy.runner import Runner
    r = Runner(None, {}, {}, tmp_path)
    f = OutputFile("x_00001-audio.mp4", "harness", "output", "34", "gifs", {"workflow": "x_00001.png"})
    out = {p.name for p in r._comfy_paths(f)}
    assert out == {"x_00001-audio.mp4", "x_00001.mp4", "x_00001.png"}
    assert all(p.parent == tmp_path / "output" / "harness" for p in r._comfy_paths(f))
