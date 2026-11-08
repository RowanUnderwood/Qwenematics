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
        """The active LLM profile: settings.yaml `llm`, or its `remote` block overlaid when the
        GUI (state file `llm`) picked the remote backend. A remote has no launcher/stopper."""
        base = {k: v for k, v in self.raw["llm"].items() if k != "remote"}
        st = load_state(self.projects_root).get("llm") or {}
        remote = self.llm_remote
        backend = st.get("backend") or base.get("backend") or "local"
        if backend == "remote":
            base = {k: v for k, v in base.items() if k not in ("launcher", "stopper")}
            base.update({"media_mode": "frames", **remote})
        base["backend"] = backend
        return base

    @property
    def llm_remote(self) -> dict:
        """settings.yaml `llm.remote` with the GUI's url/model edits applied."""
        r = dict(self.raw["llm"].get("remote") or {})
        st = load_state(self.projects_root).get("llm") or {}
        if st.get("remote_url"):
            r["base_url"] = st["remote_url"]
        if st.get("remote_model"):
            r["model"] = st["remote_model"]
        if r.get("base_url"):
            r["base_url"] = normalize_llm_url(r["base_url"])
        return r

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


def normalize_llm_url(url: str) -> str:
    """'192.168.2.74:8080/v1' -> 'http://192.168.2.74:8080' (the client adds /v1/...)."""
    u = url.strip().rstrip("/")
    if "://" not in u:
        u = "http://" + u
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


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
