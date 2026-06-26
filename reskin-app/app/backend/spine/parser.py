"""Read-only helpers for Spine 4.x JSON skeletons.

The schema we care about (subset):
{
  "skeleton": {...},
  "bones": [...],
  "slots":   [{"name": str, "bone": str, "attachment": str?}],
  "skins":   [{"name": str, "attachments": {slotName: {attName: {x,y,width,height,rotation,scaleX,scaleY,...}}}}]
              # Spine 4.0+ array form. Some older files use object form {skinName: {...}}.
  "animations": {...}
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SlotInfo:
    name: str
    bone: str
    attachment: str | None  # default attachment name (slot name if missing)


def load_spine_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def find_spine_json(project_dir: Path) -> Path | None:
    """Return the most likely Spine .json in a project directory.

    Preference order:
      1. *.json files whose top-level has both "skeleton" and "bones"
      2. Files literally called Spine.json or skeleton.json
    """
    candidates = sorted(project_dir.glob("*.json"))
    typed = []
    for p in candidates:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "skeleton" in d and "bones" in d:
            typed.append(p)
    if not typed:
        return None
    # Prefer (in order): file matching folder name, "Spine.json"/"skeleton.json",
    # then files without a hyphen suffix (which we use for per-skin variants).
    folder = project_dir.name.lower()
    base_names = {"spine", "skeleton", folder}

    def rank(p):
        stem = p.stem.lower()
        is_base = stem in base_names
        has_dash = "-" in stem
        return (not is_base, has_dash, p.name.lower())

    typed.sort(key=rank)
    return typed[0]


def slots(spine_json: dict) -> list[SlotInfo]:
    out = []
    for s in spine_json.get("slots", []):
        out.append(
            SlotInfo(
                name=s["name"],
                bone=s.get("bone", "root"),
                attachment=s.get("attachment") or s["name"],
            )
        )
    return out


def skin_names(spine_json: dict) -> list[str]:
    skins = spine_json.get("skins")
    if isinstance(skins, list):
        return [s.get("name", "") for s in skins if s.get("name")]
    if isinstance(skins, dict):
        return list(skins.keys())
    return []


def animation_names(spine_json: dict) -> list[str]:
    """Return the list of animation names defined in the skeleton."""
    anims = spine_json.get("animations")
    if isinstance(anims, dict):
        return list(anims.keys())
    return []


def default_skin_attachments(spine_json: dict) -> dict[str, dict[str, dict]]:
    """Return {slotName: {attachmentName: attachmentMeta}} for the default skin.

    Handles both list-of-skins (4.0+) and dict-of-skins (legacy) shapes.
    """
    skins = spine_json.get("skins")
    if isinstance(skins, list):
        for s in skins:
            if s.get("name") == "default":
                return s.get("attachments", {}) or {}
        return {}
    if isinstance(skins, dict):
        return skins.get("default", {}) or {}
    return {}


def attachment_meta(spine_json: dict, slot: str, skin: str = "default") -> dict | None:
    """Return the meta dict for the default attachment in `slot` under `skin`."""
    skins = spine_json.get("skins")
    if isinstance(skins, list):
        skin_obj = next((s for s in skins if s.get("name") == skin), None)
    elif isinstance(skins, dict):
        skin_obj = skins.get(skin)
    else:
        skin_obj = None
    if not skin_obj:
        return None
    atts = skin_obj.get("attachments", skin_obj)
    slot_atts = atts.get(slot)
    if not slot_atts:
        return None
    # First (or only) attachment in the slot
    return next(iter(slot_atts.values()), None)


def list_part_pngs(project_dir: Path) -> dict[str, Path]:
    """Return {slotName: pngPath} for per-region PNGs at the project root.

    Skips atlas spritesheets (PNGs that have a sibling .atlas) and per-skin
    atlas sheets named `Spine-<skin>.png`.
    """
    out = {}
    for p in sorted(project_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        if p.with_suffix(".atlas").exists():
            continue
        if p.stem.startswith("Spine-"):
            continue
        out[p.stem] = p
    return out
