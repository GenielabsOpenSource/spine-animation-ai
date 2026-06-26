"""Build the "exploded character" composite for the alternative reskin pipeline.

Lays each part roughly where it appears on the assembled character — using
the default-skin attachment x/y from the Spine JSON — then iteratively
pushes overlapping rects apart until every pair has at least `padding` px
of breathing room between them. Result: a recognisable layout of the
character with all parts visible, like an exploded mechanical drawing.

Compared to a packed atlas: Gemini gets stronger anatomical context
(head→body→limbs), which usually improves cross-part style coherence.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..spine import parser as spine_parser


EXPLODED_RESKIN_PROMPT = """\
This image shows EVERY body part of ONE single character, laid out roughly
in their assembled positions but separated with small gaps so each part is
fully visible (an "exploded view" of the character). Each rectangular
region in the image is a separate body part of the SAME character — head,
eyes, hair, hands, body, limbs, clothing pieces, accessories, etc.

TASK: reskin every part to: "{user_prompt}".

CRITICAL CONSTRAINTS:
- All parts belong to ONE coherent character. Apply the SAME colour palette,
  shading, line weight, and art style across every part.
- Keep every part in EXACTLY the same position, size, and rotation as the
  input. Do NOT move, scale, rotate, merge, split, add, or remove any part.
  Each part stays inside its current rectangle.
- Preserve each part's silhouette and overall shape — only the surface
  appearance changes.

OUTPUT:
- Same dimensions as the input.
- Background between parts must remain clean white. Do NOT add shadows,
  gradients, or coloured fills in the gaps.
- Do NOT render any text in the output.
"""


EXPLODED_RESKIN_NEGATIVE = (
    "do not move parts; do not change part sizes; do not rotate parts; do "
    "not merge or overlap parts; do not add shadows or gradients to the "
    "background; do not change part silhouettes; do not crop or reframe; "
    "do not render any text."
)


def _list_part_pngs(project_dir: Path) -> dict[str, Path]:
    """Loose region PNGs at the project root (post auto-explode), keyed by stem."""
    out: dict[str, Path] = {}
    for p in sorted(Path(project_dir).iterdir()):
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        if p.with_suffix(".atlas").exists():
            continue
        if p.stem.startswith("Spine-"):
            continue
        out[p.stem] = p
    return out


def _default_skin_attachments(spine_json: dict) -> dict:
    skins = spine_json.get("skins")
    if isinstance(skins, list):
        for s in skins:
            if s.get("name") == "default":
                return s.get("attachments", {}) or {}
        return {}
    if isinstance(skins, dict):
        return skins.get("default", {}) or {}
    return {}


def _world_centers_by_region(spine_json: dict) -> dict[str, dict]:
    """Map region_name → first-slot's attachment placement.

    Multiple slots can share a region (mirrored eyes etc.); we keep the
    first encountered to avoid duplicating regions in the composite.
    Result fields: cx, cy (Y-flipped to image space), scaleX, scaleY.
    """
    atts = _default_skin_attachments(spine_json)
    out: dict[str, dict] = {}
    for slot_atts in atts.values():
        for att_key, meta in slot_atts.items():
            if not isinstance(meta, dict):
                continue
            region = meta.get("name") or att_key
            if region in out:
                continue
            out[region] = {
                "cx": float(meta.get("x", 0.0)),
                "cy": -float(meta.get("y", 0.0)),  # Spine Y up → image Y down
                "scaleX": float(meta.get("scaleX", 1.0)),
                "scaleY": float(meta.get("scaleY", 1.0)),
            }
    return out


def _resolve_overlaps(
    placed: list[dict],
    padding: int,
    *,
    max_iter: int = 250,
    damping: float = 0.5,
    min_push_px: float = 0.5,
) -> None:
    """Iterative AABB push: separate every pair until gap ≥ padding.

    Mutates `placed` in place. Each entry must have cx, cy, w, h.
    """
    n = len(placed)
    for _ in range(max_iter):
        any_push = False
        for i in range(n):
            a = placed[i]
            for j in range(i + 1, n):
                b = placed[j]
                dx = b["cx"] - a["cx"]
                dy = b["cy"] - a["cy"]
                req_x = (a["w"] + b["w"]) / 2 + padding
                req_y = (a["h"] + b["h"]) / 2 + padding
                overlap_x = req_x - abs(dx)
                overlap_y = req_y - abs(dy)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                    a["cx"] -= 1.0
                    b["cx"] += 1.0
                    any_push = True
                    continue
                if overlap_x < overlap_y:
                    push = overlap_x * damping
                    if push < min_push_px:
                        continue
                    sign = 1.0 if dx > 0 else -1.0
                    a["cx"] -= sign * push / 2
                    b["cx"] += sign * push / 2
                else:
                    push = overlap_y * damping
                    if push < min_push_px:
                        continue
                    sign = 1.0 if dy > 0 else -1.0
                    a["cy"] -= sign * push / 2
                    b["cy"] += sign * push / 2
                any_push = True
        if not any_push:
            break


def build_exploded_composite(
    project_dir: Path,
    workdir: Path,
    *,
    padding: int = 10,
) -> dict:
    """Place each region at its default-skin attachment x/y, separate
    overlapping rects with `padding` px gaps, paste onto a single canvas.
    Writes:
        workdir/composite.png
        workdir/layout_map.json
    Returns the layout-map dict.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(project_dir)

    parts = _list_part_pngs(project_dir)
    if not parts:
        raise FileNotFoundError(
            f"no per-region PNGs in {project_dir} — run scripts/explode_spine_atlas.py first"
        )

    spine_json_path = spine_parser.find_spine_json(project_dir)
    if not spine_json_path:
        raise FileNotFoundError(f"no Spine JSON in {project_dir}")
    spine_json = spine_parser.load_spine_json(spine_json_path)
    centers = _world_centers_by_region(spine_json)

    placed: list[dict] = []
    for region, png_path in parts.items():
        meta = centers.get(region) or {"cx": 0.0, "cy": 0.0, "scaleX": 1.0, "scaleY": 1.0}
        img = Image.open(png_path).convert("RGBA")
        sx, sy = meta["scaleX"], meta["scaleY"]
        if abs(sx) != 1.0 or abs(sy) != 1.0:
            new_w = max(1, int(round(img.width * abs(sx))))
            new_h = max(1, int(round(img.height * abs(sy))))
            img = img.resize((new_w, new_h), Image.LANCZOS)
        if sx < 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if sy < 0:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        placed.append({
            "region": region,
            "img": img,
            "cx": meta["cx"],
            "cy": meta["cy"],
            "w": img.width,
            "h": img.height,
        })

    _resolve_overlaps(placed, padding)

    min_x = min(p["cx"] - p["w"] / 2 for p in placed)
    min_y = min(p["cy"] - p["h"] / 2 for p in placed)
    max_x = max(p["cx"] + p["w"] / 2 for p in placed)
    max_y = max(p["cy"] + p["h"] / 2 for p in placed)
    margin = padding
    canvas_w = int(round(max_x - min_x)) + 2 * margin
    canvas_h = int(round(max_y - min_y)) + 2 * margin

    composite = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    placements: dict[str, dict[str, int]] = {}
    for p in placed:
        x = int(round(p["cx"] - p["w"] / 2 - min_x + margin))
        y = int(round(p["cy"] - p["h"] / 2 - min_y + margin))
        composite.paste(p["img"], (x, y), p["img"])
        placements[p["region"]] = {
            "x": x, "y": y, "w": int(p["w"]), "h": int(p["h"]),
        }

    composite.save(workdir / "composite.png")
    layout = {
        "version": 2,
        "mode": "exploded",
        "composite_w": canvas_w,
        "composite_h": canvas_h,
        "padding": int(padding),
        "placements": placements,
    }
    (workdir / "layout_map.json").write_text(json.dumps(layout, indent=2))
    return layout
