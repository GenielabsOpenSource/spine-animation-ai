"""Detect folders that contain MORE than one Spine project triplet.

Some Spine exports drop several `<base>.{json,atlas|atlas.txt,png}` triplets
into one folder (e.g. a "characters" folder with multiple rigs side by side).
The single-project model in `projects.py` would silently pick one and hide
the rest. This module enumerates every base, materializes a per-base
sub-folder under `.genie/projects/<base>/` (renaming `.atlas.txt` → `.atlas`),
and lets the caller surface the choice.
"""
from __future__ import annotations

import json as _json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .spine import parser as spine_parser


@dataclass
class ProjectCandidate:
    base: str
    display_name: str
    path: Path  # ready-to-open materialized folder


class MultipleProjectsError(Exception):
    """Raised by open_project when a folder has >1 project triplet.

    The server endpoint catches this and returns a 409 with the candidate
    list so the frontend can show a picker.
    """

    def __init__(self, folder: Path, candidates: list[ProjectCandidate]):
        super().__init__(f"{folder} has {len(candidates)} Spine projects")
        self.folder = folder
        self.candidates = candidates


def _is_skeleton_json(p: Path) -> bool:
    """Mirror find_spine_json's signature check."""
    try:
        d = _json.loads(p.read_text())
    except Exception:
        return False
    return isinstance(d, dict) and "skeleton" in d and "bones" in d


def _atlas_for_base(folder: Path, base: str) -> Path | None:
    for ext in (".atlas", ".atlas.txt"):
        p = folder / f"{base}{ext}"
        if p.exists():
            return p
    return None


def _sheet_for_atlas(atlas_path: Path) -> Path | None:
    """First non-blank line of the atlas is the sheet filename, resolved
    relative to the atlas's folder. Returns None if missing."""
    try:
        text = atlas_path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line:
            sheet = atlas_path.parent / line
            return sheet if sheet.exists() else None
    return None


def _materialize_candidate(
    folder: Path,
    base: str,
    json_path: Path,
    atlas_path: Path,
    sheet_path: Path,
) -> Path:
    """Copy the triplet into `{folder}/.genie/projects/{base}/` and rename
    the atlas to `.atlas` (no `.txt` suffix). Idempotent."""
    out = folder / ".genie" / "projects" / base
    out.mkdir(parents=True, exist_ok=True)
    norm_json = out / f"{base}.json"
    norm_atlas = out / f"{base}.atlas"
    norm_sheet = out / sheet_path.name
    if not norm_json.exists():
        shutil.copy2(json_path, norm_json)
    if not norm_atlas.exists():
        shutil.copy2(atlas_path, norm_atlas)
    if not norm_sheet.exists():
        shutil.copy2(sheet_path, norm_sheet)
    return out


def find_project_candidates(folder: Path) -> list[ProjectCandidate]:
    """Enumerate Spine project triplets in `folder`.

    - 0 candidates → caller raises FileNotFoundError as today.
    - 1 candidate → return `[ProjectCandidate(path=folder, ...)]` (no
      materialization needed; the existing single-project flow handles the
      `.atlas.txt` → auto-explode rename via _maybe_auto_explode).
    - >1 candidates → materialize each into `.genie/projects/<base>/` and
      return a list whose paths point at those sub-folders.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []

    triplets: list[tuple[str, Path, Path, Path]] = []
    for json_path in sorted(folder.glob("*.json")):
        if not _is_skeleton_json(json_path):
            continue
        base = json_path.stem
        atlas_path = _atlas_for_base(folder, base)
        if atlas_path is None:
            continue
        sheet_path = _sheet_for_atlas(atlas_path)
        if sheet_path is None:
            continue
        triplets.append((base, json_path, atlas_path, sheet_path))

    if not triplets:
        return []

    if len(triplets) == 1:
        base, *_ = triplets[0]
        return [ProjectCandidate(base=base, display_name=base, path=folder)]

    out: list[ProjectCandidate] = []
    for base, json_path, atlas_path, sheet_path in triplets:
        sub = _materialize_candidate(folder, base, json_path, atlas_path, sheet_path)
        out.append(ProjectCandidate(base=base, display_name=base, path=sub))
    return out
