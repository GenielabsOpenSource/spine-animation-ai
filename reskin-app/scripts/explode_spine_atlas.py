#!/usr/bin/env python3
"""Explode a packed Spine atlas (single big PNG + .atlas + .json) into the
per-region-PNG layout that Genie Reskin expects.

Inputs (in one folder):
  - {name}.atlas
  - {name}.png   (the packed spritesheet)
  - {name}.json  (the Spine skeleton)

Outputs (written to --output-dir):
  - {region}.png  for every region in the atlas (UTF-8 names supported)
  - Spine.json    a copy of the input skeleton (Genie's parser preference)
  - {base}.atlas  re-packed in standard Spine 4.x format
  - {base}.png    re-packed atlas spritesheet

Usage:
    python3 explode_spine_atlas.py \
      --char-dir /path/to/Furina \
      --base-name B_funingn \
      --output-dir /tmp/genie-furina
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_atlas(atlas_path: Path) -> tuple[tuple[int, int], list[dict]]:
    """Return ((sheet_w, sheet_h), [{name, x, y, w, h, rotate}]).

    Spine atlas format here uses `bounds: x, y, w, h` and optional
    `rotate: 90|180|270`. Header has `size: W, H` and may have other lines.
    """
    text = atlas_path.read_text()
    sheet_size = (0, 0)
    for line in text.splitlines():
        m = re.match(r"\s*size:\s*(\d+)\s*,\s*(\d+)", line)
        if m:
            sheet_size = (int(m.group(1)), int(m.group(2)))
            break

    regions: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        # Region name lines are at column 0 (no leading whitespace) and don't
        # contain ':' as a key. The first such line in the file is the page
        # name (the .png filename) — we skip whatever the first one is by
        # tracking the page header.
        if not raw.startswith((" ", "\t")) and ":" not in line:
            if cur is not None:
                regions.append(cur)
            cur = {"name": line, "x": 0, "y": 0, "w": 0, "h": 0, "rotate": 0}
            continue
        if cur is None:
            continue
        ms = re.match(r"\s*bounds:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if ms:
            cur["x"] = int(ms.group(1))
            cur["y"] = int(ms.group(2))
            cur["w"] = int(ms.group(3))
            cur["h"] = int(ms.group(4))
            continue
        mr = re.match(r"\s*rotate:\s*(\d+|true|false)", line)
        if mr:
            v = mr.group(1)
            if v == "true":
                cur["rotate"] = 90
            elif v == "false":
                cur["rotate"] = 0
            else:
                cur["rotate"] = int(v)
            continue
        # The first line we encountered as a "name" was actually the page (.png),
        # but we'll filter it later by checking it has bounds. If a region has
        # no bounds it gets discarded.
    if cur is not None:
        regions.append(cur)

    # Drop entries without bounds (the page header sneaks in as the first one)
    regions = [r for r in regions if r["w"] > 0 and r["h"] > 0]
    return sheet_size, regions


def crop_region(sheet: Image.Image, region: dict) -> tuple[str, Image.Image]:
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    rot = region.get("rotate", 0)
    box_w, box_h = (h, w) if rot in (90, 270) else (w, h)
    crop = sheet.crop((x, y, x + box_w, y + box_h))
    if rot == 90:
        crop = crop.rotate(-90, expand=True)
    elif rot == 180:
        crop = crop.rotate(180, expand=True)
    elif rot == 270:
        crop = crop.rotate(90, expand=True)
    return region["name"], crop


def render_static(
    spine: dict,
    cropped: dict[str, Image.Image],
    pad: int = 100,
) -> Image.Image:
    """Best-effort composite of every default-skin attachment at its bone-local
    (x, y, rotation, scale) projected against an identity bone transform.

    This won't be a true rest pose because we ignore the bone hierarchy, but
    it gives the user something visually meaningful to use as a reference.
    """
    skins = spine.get("skins")
    if isinstance(skins, list):
        default = next((s for s in skins if s.get("name") == "default"), None)
        atts = (default or {}).get("attachments", {}) if default else {}
    elif isinstance(skins, dict):
        atts = skins.get("default", {}) or {}
    else:
        atts = {}

    # Collect placement instructions in atlas-region coords
    instructions = []
    bbox_x = bbox_y = math.inf
    bbox_X = bbox_Y = -math.inf
    for slot_name, slot_atts in atts.items():
        for att_name, meta in slot_atts.items():
            region_name = meta.get("name", att_name)
            img = cropped.get(region_name)
            if img is None:
                continue
            ax = float(meta.get("x", 0))
            ay = float(meta.get("y", 0))
            arot = float(meta.get("rotation", 0))
            asx = float(meta.get("scaleX", 1))
            asy = float(meta.get("scaleY", 1))
            w = float(meta.get("width", img.width))
            h = float(meta.get("height", img.height))
            # The attachment's (x, y) is its CENTER in slot-local space (Spine
            # convention). We treat slot-local as world-space here (no bone
            # transforms applied) — this is the inaccurate part, but it lays
            # parts out roughly correctly.
            cx, cy = ax, -ay  # flip Y for image-space
            # Update overall bbox
            half_w = w * abs(asx) / 2
            half_h = h * abs(asy) / 2
            bbox_x = min(bbox_x, cx - half_w)
            bbox_y = min(bbox_y, cy - half_h)
            bbox_X = max(bbox_X, cx + half_w)
            bbox_Y = max(bbox_Y, cy + half_h)
            instructions.append({
                "img": img,
                "cx": cx,
                "cy": cy,
                "rot": arot,
                "scale_x": asx,
                "scale_y": asy,
                "target_w": int(w * abs(asx)),
                "target_h": int(h * abs(asy)),
            })

    if not instructions:
        # Fallback: just lay everything out in a packed grid
        from make_atlas import pack  # type: ignore
        pw, ph, placements = pack(cropped, padding=4)
        out = Image.new("RGBA", (pw, ph), (255, 255, 255, 255))
        for name, (x, y, _w, _h) in placements.items():
            out.paste(cropped[name], (x, y), cropped[name])
        return out

    canvas_w = int(bbox_X - bbox_x) + pad * 2
    canvas_h = int(bbox_Y - bbox_y) + pad * 2
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))

    ox = -bbox_x + pad
    oy = -bbox_y + pad

    for ins in instructions:
        img = ins["img"]
        if ins["target_w"] != img.width or ins["target_h"] != img.height:
            img = img.resize((max(1, ins["target_w"]), max(1, ins["target_h"])), Image.LANCZOS)
        if ins["rot"] != 0:
            img = img.rotate(ins["rot"], resample=Image.BICUBIC, expand=True)
        # Place by center
        px = int(ins["cx"] + ox - img.width / 2)
        py = int(ins["cy"] + oy - img.height / 2)
        canvas.paste(img, (px, py), img)
    return canvas


def explode(char_dir: Path | str, base_name: str, output_dir: Path | str) -> Path:
    """Module-callable variant of the CLI. Returns the output dir."""
    char_dir = Path(char_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = char_dir / f"{base_name}.atlas"
    sheet_path = char_dir / f"{base_name}.png"
    json_path = char_dir / f"{base_name}.json"
    for p in (atlas_path, sheet_path, json_path):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    sheet_size, regions = parse_atlas(atlas_path)
    print(f"[atlas] sheet {sheet_size}, {len(regions)} regions")
    sheet = Image.open(sheet_path).convert("RGBA")

    def _safe_local(name: str) -> str:
        return re.sub(r"\s+", "_", name).strip("_") or name

    cropped: dict[str, Image.Image] = {}
    for r in regions:
        name, img = crop_region(sheet, r)
        cropped[name] = img
        # Save as sanitized-name PNG so atlas region & filename stay aligned.
        out_name = _safe_local(name).replace("/", "_") + ".png"
        img.save(out_dir / out_name)
    print(f"[parts] wrote {len(cropped)} part PNGs")

    # Copy skeleton JSON in (Genie expects Spine.json), then RE-PACK the
    # exploded regions into a standard-format Spine atlas. The original GI
    # atlas uses an old compact "bounds:" format that spine-pixi-v8 can't
    # parse — re-packing yields the modern xy/size/orig/offset format.
    #
    # Some regions have spaces in their names (e.g. "马尾 2") which confuse
    # spine-pixi's parser. We sanitize by replacing whitespace with `_` in
    # both the atlas region names AND the Spine.json attachment `name:`
    # fields so the skeleton still resolves them.
    def _safe(name: str) -> str:
        return re.sub(r"\s+", "_", name).strip("_") or name

    name_map = {n: _safe(n) for n in cropped.keys()}

    spine = json.loads(json_path.read_text())
    skins = spine.get("skins")

    def _patch_skin(skin_obj):
        for slot_atts in (skin_obj.get("attachments") or {}).values():
            for att_key, att in slot_atts.items():
                if not isinstance(att, dict):
                    continue
                # The atlas region a Spine attachment resolves to is `att.name`
                # if set, else the attachment KEY. We need it to point to the
                # sanitized region name when the original had whitespace.
                effective = att.get("name", att_key)
                if effective in name_map and name_map[effective] != effective:
                    att["name"] = name_map[effective]

    if isinstance(skins, list):
        for skin in skins:
            _patch_skin(skin)
    elif isinstance(skins, dict):
        for skin in skins.values():
            if isinstance(skin, dict):
                _patch_skin(skin)
    (out_dir / "Spine.json").write_text(json.dumps(spine, indent=2))

    # Repack the per-region PNGs into a standard-format atlas
    sys.path.insert(0, str(SCRIPT_DIR))
    from make_atlas import pack as _pack_fn  # type: ignore  # noqa: E402

    region_imgs = {name_map[name]: img for name, img in cropped.items()}
    sheet_w, sheet_h, placements = _pack_fn(region_imgs, padding=2)
    new_sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for stem, (x, y, w, h) in placements.items():
        new_sheet.paste(region_imgs[stem], (x, y))
    new_sheet_path = out_dir / sheet_path.name  # same name as original
    new_sheet.save(new_sheet_path)

    # Spine 4.x atlas format: page header lines are indented, blank line
    # separates page from regions, region names are at column 0 with their
    # properties indented.
    new_atlas_lines = [
        "",
        sheet_path.name,
        f"  size: {sheet_w},{sheet_h}",
        "  format: RGBA8888",
        "  filter: Linear,Linear",
        "  repeat: none",
    ]
    for stem, (x, y, w, h) in placements.items():
        new_atlas_lines.extend([
            stem,
            "  rotate: false",
            f"  xy: {x}, {y}",
            f"  size: {w}, {h}",
            f"  orig: {w}, {h}",
            "  offset: 0, 0",
            "  index: -1",
        ])
    (out_dir / atlas_path.name).write_text("\n".join(new_atlas_lines) + "\n")
    print(f"[repack] {atlas_path.name} ({sheet_w}x{sheet_h}, {len(placements)} regions)")
    # No static.png: spine-pixi renders the project from atlas + Spine.json,
    # and the per-skin canvas snapshot is what feeds the reskin pipeline.
    print(f"[done] {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-dir", required=True)
    ap.add_argument("--base-name", required=True,
                    help="Stem shared by .atlas/.png/.json (e.g. B_funingn)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    explode(args.char_dir, args.base_name, args.output_dir)


if __name__ == "__main__":
    main()
