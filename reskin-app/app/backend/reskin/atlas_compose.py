"""Build the [snapshot | atlas-sheet] composite for the atlas-based reskin
pipeline.

Gemini receives a single image with both halves so it can use the rendered
character (LEFT) as the colour/style reference while reskinning the packed
atlas regions (RIGHT) in-place. After the round-trip we split the result
back into the two halves; the atlas half is the source of truth for parts.

Gemini's `_pad_to_aspect_ratio` (in app/backend/ai/gemini.py) handles aspect
ratio matching — we just paste both halves at native size with a shared
height.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ATLAS_RESKIN_PROMPT = """\
This image is a 2-half side-by-side composition of ONE character.

LEFT half: a fully-rendered single character in a pose. Use this as the
single source of truth for the new colour palette, shading, line weight,
and overall art style.

RIGHT half: that SAME character's atlas spritesheet — every body part
packed into a rectangular grid at native size, with empty space between
parts. Each packed region corresponds to one body part on the character.

TASK: reskin the entire character to: "{user_prompt}".

CRITICAL CONSTRAINTS:
- Every region on the RIGHT must belong to the SAME reskinned character
  shown on the LEFT. Pull colours from the LEFT and apply them
  consistently to every region on the RIGHT.
- Keep every region on the RIGHT in EXACTLY the same position, size, and
  rotation as the input. Do NOT move, scale, rotate, merge, split, add,
  or remove any region. Each region stays inside its current rectangle on
  the sheet.
- Preserve the LEFT-half pose, silhouette, and proportions exactly.

OUTPUT:
- Same dimensions as the input.
- Background between regions on the RIGHT must remain the same neutral
  colour as the input (transparent / black / white as the input shows).
  Do NOT introduce shadows, gradients, or coloured fills in the empty
  atlas areas.
- Do NOT render any text anywhere in the output.
"""


ATLAS_RESKIN_NEGATIVE = (
    "do not move regions on the right; do not change region sizes; do not "
    "rotate regions; do not merge or overlap regions; do not add shadows or "
    "gradients to the atlas background; do not change the silhouette on the "
    "left; do not crop or reframe; do not render any text."
)


def build_atlas_composite(
    snapshot_path: Path,
    atlas_sheet_path: Path,
    workdir: Path,
    *,
    atlas_meta_name: str | None = None,
) -> dict:
    """Compose `[snapshot | atlas_sheet]` side-by-side at native size.

    The shorter half is V-centered with white padding above and below.
    Writes:
      workdir/composite.png
      workdir/layout_map.json
    Returns the layout-map dict.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    snap = Image.open(snapshot_path).convert("RGBA")
    atlas = Image.open(atlas_sheet_path).convert("RGBA")
    sw, sh = snap.size
    aw, ah = atlas.size

    canvas_h = max(sh, ah)
    canvas_w = sw + aw
    composite = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    snap_off_y = (canvas_h - sh) // 2
    atlas_off_y = (canvas_h - ah) // 2
    composite.paste(snap, (0, snap_off_y), snap)
    composite.paste(atlas, (sw, atlas_off_y), atlas)

    composite.save(workdir / "composite.png")

    layout = {
        "version": 2,
        "mode": "atlas",
        "composite_w": canvas_w,
        "composite_h": canvas_h,
        "snapshot_rect": {"x": 0,  "y": snap_off_y,  "w": sw, "h": sh},
        "atlas_rect":    {"x": sw, "y": atlas_off_y, "w": aw, "h": ah},
        "snapshot_path": str(Path(snapshot_path)),
        "atlas_sheet_name": Path(atlas_sheet_path).name,
        "atlas_meta_name": atlas_meta_name or "",
        "atlas_scale": 1.0,
    }
    (workdir / "layout_map.json").write_text(json.dumps(layout, indent=2))
    return layout
