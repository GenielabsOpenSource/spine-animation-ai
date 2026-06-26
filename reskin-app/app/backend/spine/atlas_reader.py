"""Read region rects from a Spine atlas.

Supports both the standard Spine 4.x format (`xy: x, y` + `size: w, h` +
optional `rotate:`) AND the legacy compact format (`bounds: x, y, w, h`).
The new atlas-based reskin pipeline reads region rects from the project's
ORIGINAL atlas — without ever needing to compute slot bboxes from the live
spine instance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AtlasRegion:
    name: str
    x: int
    y: int
    w: int      # packed width on the sheet (after any rotation)
    h: int      # packed height on the sheet (after any rotation)
    rotate: int  # 0 / 90 / 180 / 270
    orig_w: int  # original (un-rotated) width
    orig_h: int


@dataclass
class AtlasPage:
    sheet_filename: str
    sheet_w: int
    sheet_h: int
    regions: list[AtlasRegion]


def parse_atlas(atlas_path: Path) -> AtlasPage:
    """Parse a Spine atlas file. Returns the first page (Genie projects use a
    single-page atlas). Region order matches the file order."""
    text = Path(atlas_path).read_text()
    lines = text.splitlines()

    # First non-blank line is the sheet filename.
    sheet_filename = ""
    cursor = 0
    while cursor < len(lines):
        s = lines[cursor].strip()
        if s:
            sheet_filename = s
            break
        cursor += 1
    cursor += 1

    # Page-level fields are indented properties immediately after the sheet
    # filename. The standard format puts `size:` here. The legacy format puts
    # `size:` at the very top of the file (before the sheet filename).
    sheet_w = sheet_h = 0

    # Try to find sheet size anywhere — both layouts have a `size: W, H`.
    for ln in lines:
        m = re.match(r"\s*size:\s*(\d+)\s*[,xX]\s*(\d+)\s*$", ln)
        if m:
            sheet_w = int(m.group(1))
            sheet_h = int(m.group(2))
            break

    regions: list[AtlasRegion] = []
    cur_name: str | None = None
    cur: dict | None = None

    def _flush():
        nonlocal cur_name, cur
        if cur_name is None or cur is None:
            return
        rotate = cur.get("rotate", 0)
        if "bounds" in cur:
            x, y, w, h = cur["bounds"]
        else:
            x, y = cur.get("xy", (0, 0))
            w, h = cur.get("size", (0, 0))
        if w <= 0 or h <= 0:
            cur_name = None
            cur = None
            return
        orig_w, orig_h = cur.get("orig", (w, h))
        if rotate in (90, 270):
            # In rotated regions, the packed (w, h) is the rotated footprint.
            # Original (un-rotated) dims are typically `orig:` if present, else
            # we infer them by swapping w and h.
            if "orig" not in cur:
                orig_w, orig_h = h, w
        regions.append(AtlasRegion(
            name=cur_name,
            x=int(x), y=int(y),
            w=int(w), h=int(h),
            rotate=int(rotate),
            orig_w=int(orig_w), orig_h=int(orig_h),
        ))
        cur_name = None
        cur = None

    # Walk regions: a region starts when we hit a non-indented, non-empty
    # line that ISN'T a key:value line and ISN'T the sheet filename we already
    # consumed.
    seen_first_blank = False
    for raw in lines[cursor:]:
        line = raw.rstrip("\n")
        if not raw.strip():
            seen_first_blank = True
            continue
        # Region property lines are indented OR contain a `:`.
        is_indented = raw.startswith((" ", "\t"))
        if not is_indented and ":" not in line.strip():
            # New region name
            _flush()
            cur_name = line.strip()
            cur = {}
            continue
        if cur is None:
            continue
        s = line.strip()
        m = re.match(r"bounds:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", s)
        if m:
            cur["bounds"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            continue
        m = re.match(r"xy:\s*(\d+)\s*,\s*(\d+)", s)
        if m:
            cur["xy"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.match(r"size:\s*(\d+)\s*,\s*(\d+)", s)
        if m and "size" not in cur:
            cur["size"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.match(r"orig:\s*(\d+)\s*,\s*(\d+)", s)
        if m:
            cur["orig"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.match(r"rotate:\s*(\d+|true|false)", s, re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            if v == "true":
                cur["rotate"] = 90
            elif v == "false":
                cur["rotate"] = 0
            else:
                cur["rotate"] = int(v)
            continue
    _flush()

    return AtlasPage(
        sheet_filename=sheet_filename,
        sheet_w=sheet_w,
        sheet_h=sheet_h,
        regions=regions,
    )


def _atlas_files(folder: Path) -> list[Path]:
    """All files in `folder` that look like a Spine atlas — `.atlas` plus
    the variant some exporters use (`.atlas.txt`). Same content format
    either way; the parser doesn't care."""
    out: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".atlas") or n.endswith(".atlas.txt"):
            out.append(p)
    return out


def _atlas_stem(p: Path) -> str:
    """Strip the atlas extension (handles both `.atlas` and `.atlas.txt`)."""
    if p.name.lower().endswith(".atlas.txt"):
        return p.name[: -len(".atlas.txt")]
    return p.stem


def find_atlas_pair(project_path: Path, spine_json_stem: str) -> tuple[Path, Path] | None:
    """Locate the project's original (atlas, sheet) pair.

    Mirrors the selection rules in projects.Project.to_payload(): prefer the
    atlas matching the folder/spine-json name, deprioritize per-skin variants
    (`Spine-*` stem). Returns None if no atlas+sheet pair exists.
    """
    project_path = Path(project_path)
    folder = project_path.name.lower()
    base_names = {"spine", folder, spine_json_stem.lower()}

    def rank(p: Path):
        stem = _atlas_stem(p).lower()
        is_per_skin = stem.startswith("spine-")
        is_base = stem in base_names
        is_txt = p.name.lower().endswith(".atlas.txt")
        return (is_per_skin, is_txt, not is_base, p.name.lower())

    candidates = sorted(_atlas_files(project_path), key=rank)
    for atlas_path in candidates:
        page = parse_atlas(atlas_path)
        sheet = project_path / page.sheet_filename if page.sheet_filename else None
        if sheet and sheet.exists():
            return atlas_path, sheet
    return None
