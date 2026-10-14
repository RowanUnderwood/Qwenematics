"""A project = one folder: project.db + media/ + exports/.

The folder is the save: the DB commits as you go, so "Save" is implicit. "Save As"
snapshots the folder; "Archive" zips it; "Delete" removes the folder *and* every file
the project ever caused ComfyUI to write (FileRec rows outside the project).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, func, select

from . import models as m

MEDIA_DIRS = ("refs", "renders", "previews", "latents", "uploads", "proxies", "frames")
SCHEMA_VERSION = "1"


def next_id(s: Session, model) -> int:
    """A primary key never handed out before in this project. SQLite reuses the top id
    after a delete, and ids end up in media URLs and chat attachments, so a reused id
    would show (or point at) the wrong file."""
    key = f"seq_{model.__tablename__}"
    row = s.get(m.Meta, key) or m.Meta(key=key, value="0")
    nid = max(int(row.value or 0), s.exec(select(func.max(model.id))).one() or 0) + 1
    row.value = str(nid)
    s.add(row)
    return nid


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return s[:60] or "project"


class Project:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.db_path = self.root / "project.db"
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}",
                                    connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(conn, _):
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, projects_root: Path, name: str) -> "Project":
        root = projects_root / slugify(name)
        n = 1
        while root.exists():
            n += 1
            root = projects_root / f"{slugify(name)}_{n}"
        for d in MEDIA_DIRS:
            (root / "media" / d).mkdir(parents=True, exist_ok=True)
        (root / "exports").mkdir(parents=True, exist_ok=True)
        p = cls(root)
        SQLModel.metadata.create_all(p.engine)
        with p.session() as s:
            s.add(m.Meta(key="name", value=name))
            s.add(m.Meta(key="schema", value=SCHEMA_VERSION))
            s.add(m.Meta(key="created", value=datetime.now().isoformat(timespec="seconds")))
        return p

    @classmethod
    def open(cls, root: Path) -> "Project":
        if not (Path(root) / "project.db").is_file():
            raise FileNotFoundError(f"no project.db in {root}")
        p = cls(Path(root))
        SQLModel.metadata.create_all(p.engine)      # adds tables introduced since creation
        return p

    @staticmethod
    def list(projects_root: Path) -> list[dict]:
        out = []
        for db in sorted(projects_root.glob("*/project.db")):
            out.append({"path": str(db.parent), "folder": db.parent.name,
                        "modified": datetime.fromtimestamp(db.stat().st_mtime).isoformat(timespec="seconds")})
        return out

    @property
    def name(self) -> str:
        with self.session() as s:
            row = s.get(m.Meta, "name")
            return row.value if row else self.root.name

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as s:
            yield s
            s.commit()

    def close(self):
        self.engine.dispose()

    # ------------------------------------------------------------ files
    def media(self, sub: str) -> Path:
        d = self.root / "media" / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def abspath(self, f: m.FileRec) -> Path:
        return self.root / f.path if f.location == "project" else Path(f.path)

    def register_file(self, s: Session, path: Path, kind: str, owner_type: str = "",
                      owner_id: int | None = None, location: str | None = None) -> m.FileRec:
        path = Path(path)
        if location is None:
            try:
                rel = path.resolve().relative_to(self.root.resolve())
                location, stored = "project", rel.as_posix()
            except ValueError:
                location, stored = "external", str(path)
        else:
            stored = str(path) if location != "project" else path.as_posix()
        size = path.stat().st_size if path.is_file() else 0
        rec = m.FileRec(id=next_id(s, m.FileRec), path=stored, location=location, kind=kind, owner_type=owner_type,
                        owner_id=owner_id, bytes=size)
        s.add(rec)
        s.flush()
        return rec

    def delete_files(self, s: Session, recs: list[m.FileRec]) -> list[str]:
        """Delete files on disk and their rows. Returns paths that could not be removed."""
        failed = []
        for f in recs:
            p = self.abspath(f)
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                failed.append(str(p))
                continue
            s.delete(f)
        return failed

    # ------------------------------------------------------------ save as / archive / delete
    def snapshot(self, dest_root: Path, name: str | None = None) -> Path:
        """Copy the whole project (consistent DB copy via SQLite backup) to a new folder."""
        dest = dest_root / slugify(name or f"{self.root.name}_{datetime.now():%Y%m%d_%H%M%S}")
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copytree(self.root, dest, ignore=shutil.ignore_patterns("project.db*"))
        self._backup_db(dest / "project.db")
        if name:
            snap = Project.open(dest)
            with snap.session() as s:
                row = s.get(m.Meta, "name")
                row.value = name
                s.add(row)
            snap.close()
        return dest

    def archive(self, zip_path: Path) -> Path:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_db = self.root / "exports" / "_archive_project.db"
        self._backup_db(tmp_db)
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:   # media is already compressed
                z.write(tmp_db, "project.db")
                for p in self.root.rglob("*"):
                    if p.is_file() and not p.name.startswith("project.db") and p != tmp_db:
                        z.write(p, p.relative_to(self.root).as_posix())
        finally:
            tmp_db.unlink(missing_ok=True)
        return zip_path

    def _backup_db(self, dest: Path):
        import sqlite3
        src = sqlite3.connect(self.db_path)
        dst = sqlite3.connect(dest)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()

    def delete_plan(self) -> dict:
        """Everything Delete Project would remove, for the confirmation dialog."""
        with self.session() as s:
            outside = [f for f in s.exec(select(m.FileRec)).all() if f.location != "project"]
        total = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        return {"folder": str(self.root), "folder_bytes": total,
                "outside_files": [str(self.abspath(f)) for f in outside if self.abspath(f).exists()]}

    def delete_everything(self) -> list[str]:
        plan = self.delete_plan()
        failed = []
        for p in plan["outside_files"]:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                failed.append(p)
        self.close()
        shutil.rmtree(self.root, ignore_errors=True)
        if self.root.exists():
            failed.append(str(self.root))
        return failed


def reveal_in_explorer(path: Path):
    """Open Explorer with the file selected (Windows)."""
    path = Path(path)
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", str(path)] if path.is_file() else ["explorer", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])


def dumps(obj) -> str:
    return json.dumps(obj, default=str)
