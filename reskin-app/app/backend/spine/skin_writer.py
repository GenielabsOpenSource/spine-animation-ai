"""Add a new skin to an existing Spine 4.x JSON skeleton.

Spine 4.0+ stores skins as an array:
    "skins": [
      { "name": "default",  "attachments": { slotName: { attName: { x, y, w, h, ... } } } },
      { "name": "dark-elf", "attachments": { ... } }
    ]

Older 3.8 files use a dict keyed by skin name. We support both, but always
write back in the same shape we read.

CRITICAL: bones, animations, IK constraints, and existing skins are NEVER
mutated. The function returns a deep-copied skeleton with ONLY the skins
section augmented.
"""
from __future__ import annotations

import copy
from typing import Any


def _attachment_meta(
    skins_root: list | dict, slot: str, *, source_skin: str = "default"
) -> dict | None:
    """Find an attachment dict in the source skin (used to preserve fields like
    `path`, `rotation`, `scale*` that we want to inherit)."""
    if isinstance(skins_root, list):
        skin_obj = next((s for s in skins_root if s.get("name") == source_skin), None)
        atts = (skin_obj or {}).get("attachments", {})
    else:
        atts = (skins_root or {}).get(source_skin, {})
    slot_atts = (atts or {}).get(slot)
    if not slot_atts:
        return None
    return next(iter(slot_atts.values()), None)


def add_skin(
    spine_json: dict,
    skin_name: str,
    *,
    placements: dict[str, dict[str, Any]],
    source_skin: str = "default",
    overwrite: bool = True,
    attachment_names: dict[str, str] | None = None,
) -> dict:
    """Return a copy of `spine_json` with `skins[skin_name]` filled in.

    `placements` is `{slotName: {x, y, w, h}}` from atlas_repack — the new
    skin's attachment region offsets/sizes.

    For each slot in placements:
      - inherit any non-region fields (rotation, scaleX, scaleY, color, ...) from
        the corresponding default-skin attachment if present
      - overwrite x/y/width/height with the new placement
      - the attachment name is taken from the slot's existing attachment-name
        in the default skin (or defaults to the slot name)

    The result is a deep copy — the input is never mutated.
    """
    out = copy.deepcopy(spine_json)
    skins = out.get("skins")

    new_skin_atts: dict[str, dict[str, dict]] = {}
    for slot_name, placement in placements.items():
        att_key = slot_name
        existing = _attachment_meta(skins, slot_name, source_skin=source_skin) or {}
        meta = dict(existing)
        # IMPORTANT: keep the default skin's width/height. The per-region PNG
        # in the new atlas might be a trimmed version of the original
        # un-trimmed image (e.g. legacy `bounds:` atlases lose offset info on
        # explode). Overriding width/height to the trimmed pixel size makes
        # the renderer produce a smaller, off-center part. We only fill them
        # in if the default skin's attachment didn't have them.
        if "width" not in meta:
            meta["width"] = placement["w"]
        if "height" not in meta:
            meta["height"] = placement["h"]
        # Atlas region name (may differ from slot name when slot has whitespace
        # and the atlas sanitized it).
        region = (attachment_names or {}).get(slot_name, att_key)
        meta["name"] = region
        new_skin_atts[slot_name] = {att_key: meta}

    new_entry = {"name": skin_name, "attachments": new_skin_atts}

    if isinstance(skins, list):
        if overwrite:
            skins = [s for s in skins if s.get("name") != skin_name]
        skins.append(new_entry)
        out["skins"] = skins
    elif isinstance(skins, dict):
        if overwrite or skin_name not in skins:
            skins[skin_name] = new_skin_atts
        out["skins"] = skins
    else:
        # No skins block at all — create the array form (Spine 4.x default)
        out["skins"] = [{"name": "default", "attachments": {}}, new_entry]

    return out
