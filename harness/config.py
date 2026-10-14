"""settings.yaml loader. One Settings object per process; paths resolved against the repo."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    raw: dict[str, Any]
    path: Path

    def _p(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def comfy_dir(self) -> Path:
        return Path(self.raw["comfy"]["comfy_dir"])

    @property
    def comfy_input_dir(self) -> Path:
        return self.comfy_dir / "input"

    @property
    def comfy_output_dir(self) -> Path:
        return self.comfy_dir / "output"

    @property
    def servers(self) -> dict[str, dict]:
        return self.raw["comfy"]["servers"]

    def server_launcher(self, name: str) -> Path | None:
        l = self.servers[name].get("launcher")
        return self._p(l) if l else None

    @property
    def llm(self) -> dict:
        return self.raw["llm"]

    @property
    def workflows_root(self) -> Path:
        return self._p(self.raw["workflows"]["root"])

    @property
    def preferred_workflows(self) -> dict[str, str]:
        return self.raw["workflows"].get("preferred", {})

    @property
    def iterate(self) -> dict:
        return self.raw.get("iterate", {})

    @property
    def gpu_sharing(self) -> dict:
        """settings.yaml defaults with GUI overrides from the state file applied."""
        base = {"image_on_video_gpu": "auto", "min_backlog": 2, "image_jobs_in_flight": 3,
                "est_seconds": {"image": 33, "video": 11}}
        base.update(self.raw.get("gpu_sharing") or {})
        base.update(load_state(self.projects_root).get("gpu_sharing") or {})
        return base

    @property
    def projects_root(self) -> Path:
        return self._p(self.raw.get("projects_root", "projects"))

    def save(self):
        self.path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True), encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings(path: str | None = None) -> Settings:
    p = Path(path) if path else ROOT / "settings.yaml"
    return Settings(yaml.safe_load(p.read_text(encoding="utf-8")), p)


def load_state(projects_root: Path) -> dict:
    """Small harness state (last project, GUI overrides) kept out of settings.yaml so its comments survive."""
    try:
        return json.loads((projects_root / ".harness_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(projects_root: Path, **changes) -> dict:
    st = load_state(projects_root)
    for k, v in changes.items():
        if v is None:
            st.pop(k, None)
        else:
            st[k] = v
    projects_root.mkdir(parents=True, exist_ok=True)
    (projects_root / ".harness_state.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st
