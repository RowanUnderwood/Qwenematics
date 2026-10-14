"""Per-project SQLite schema.

Reference versions are immutable: editing a reference adds a version, and shots point
at a specific version through ShotSlot. A render stores the input hash of its shot at
render time, so a shot is stale when its current hash differs from its final render's.
Every file the harness writes, in the project or in ComfyUI's folders, has a FileRec,
which is what makes cascade deletes and project deletes complete.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def now() -> datetime:
    return datetime.now(timezone.utc)


def _json(default=None):
    return Field(default_factory=(lambda: default() if callable(default) else default),
                 sa_column=Column(JSON))


class Meta(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str = ""


class BibleEntry(SQLModel, table=True):
    """Project-level design notes: plot, characters, locations, style, audio, dialogue."""
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = Field(index=True)          # plot|character|location|style|audio|dialogue|note
    title: str = ""
    body: str = ""
    updated: datetime = Field(default_factory=now)


class FileRec(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str                              # relative to the project for location=project, else absolute
    location: str = "project"              # project | comfy_input | comfy_output | external
    kind: str = ""                         # image | video | preview | latent | proxy | upload | export | audio
    owner_type: str = ""                   # ref_version | render | shot | chat | export | upload
    owner_id: Optional[int] = Field(default=None, index=True)
    bytes: int = 0
    created: datetime = Field(default_factory=now)


class Reference(SQLModel, table=True):
    """A logical reference (e.g. 'Aria - full body'); its pixels live in RefVersions."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    kind: str = "character"                # character | character_closeup | location | establishing | prop | first_frame | concept
    tags: list[str] = _json(list)
    description: str = ""                  # what it should depict (the design intent)
    current_version_id: Optional[int] = None
    created: datetime = Field(default_factory=now)


class RefVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ref_id: int = Field(index=True, foreign_key="reference.id")
    n: int = 1
    file_id: Optional[int] = Field(default=None, foreign_key="filerec.id")
    source: str = "generated"              # generated | uploaded | frame_grab
    prompt: str = ""
    params: dict[str, Any] = _json(dict)
    workflow: str = ""                     # "<id>@<version>"
    seed: Optional[int] = None
    caption: str = ""                      # one line written by Qwen at creation, used for reuse search
    parent_id: Optional[int] = None
    critic: dict[str, Any] = _json(dict)
    created: datetime = Field(default_factory=now)


class Shot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    order: float = Field(default=0, index=True)    # float so inserts between shots need no renumbering
    title: str = ""
    description: str = ""                  # what happens (design intent, user language)
    dialogue: str = ""
    audio_notes: str = ""
    mode: str = "ref2va"                   # workflow mode: t2va | ref2va | i2va | fl2va | l2va
    duration_s: float = 15.0
    prompt: str = ""                       # the final H3 prompt as last written
    params: dict[str, Any] = _json(dict)   # per-shot overrides of workflow params
    workflow: str = ""                     # pinned "<id>@<version>", empty = preferred
    final_render_id: Optional[int] = None
    status: str = "draft"                  # draft | ready | rendering | done
    created: datetime = Field(default_factory=now)


class ShotSlot(SQLModel, table=True):
    """Picture slot k of a shot -> a reference, pinned to one immutable version.
    ref_version_id is None while the reference is still rendering; the shot's render
    waits for it and pins the version then. <Picture k> = slot k."""
    id: Optional[int] = Field(default=None, primary_key=True)
    shot_id: int = Field(index=True, foreign_key="shot.id")
    ref_id: int = Field(index=True, foreign_key="reference.id")
    ref_version_id: Optional[int] = Field(default=None, index=True, foreign_key="refversion.id")
    slot: int = 1
    role: str = "subject"                  # subject | first_frame | last_frame | style


class Render(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    shot_id: Optional[int] = Field(default=None, index=True, foreign_key="shot.id")
    stage: str = "preview"                 # preview (stage 1) | final (stage 2)
    parent_id: Optional[int] = None        # final -> the preview it upscaled
    seed: Optional[int] = None
    workflow: str = ""
    mode: str = ""
    prompt: str = ""
    params: dict[str, Any] = _json(dict)
    slot_versions: list[int] = _json(list) # ref_version ids in slot order at render time
    input_hash: str = ""
    file_id: Optional[int] = None          # preview mp4 or final mp4
    latent_video_file_id: Optional[int] = None
    latent_audio_file_id: Optional[int] = None
    status: str = "done"                   # queued | running | done | failed
    picked: bool = False                   # chosen in seed hunt
    critic: dict[str, Any] = _json(dict)
    error: str = ""
    created: datetime = Field(default_factory=now)


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    role: str                              # user | assistant | tool | system_note
    content: str = ""
    attachments: list[dict] = _json(list)  # [{"type": "ref_version"|"render"|"file", "id": n}]
    tool_calls: list[dict] = _json(list)
    tool_call_id: str = ""
    summarized: bool = False               # folded into a summary message, kept for the UI
    summary_of: list[int] = _json(list)    # for summary messages: the ids they replace
    created: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str                              # ref_image | shot_preview | shot_final | final_cut | ...
    server: str = ""                       # image | video | local
    payload: dict[str, Any] = _json(dict)
    state: str = "queued"                  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = _json(dict)
    created: datetime = Field(default_factory=now)
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
