"""Re-pack a directory of part PNGs into a Spine atlas.

Wraps `scripts/make_atlas.py`'s `pack()` so we don't reimplement the bin-packing
logic. Returns placement metadata that `skin_writer.add_skin()` consumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from make_atlas import pack  # noqa: E402  pylint: disable=wrong-import-position


def repack_atlas(
    parts_dir: Path,
    output_dir: Path,
    name: str,
    *,
    padding: int = 2,
) -> dict:
    """Pack every PNG in `parts_dir` into `output_dir/{name}.png` + `{name}.atlas`.

    Returns:
        {
          "image": Path to spritesheet,
          "atlas": Path to .atlas metadata,
          "placements": {part_filename: {x, y, w, h, sheet_w, sheet_h}}
        }
    """
    parts_dir = Path(parts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import re as _re
    def _safe(name: str) -> str:
        return _re.sub(r"\s+", "_", name).strip("_") or name

    images: dict[str, Image.Image] = {}
    # Map stem (sanitized) → original-stem so callers can look up by original
    # slot name (which may contain whitespace).
    stem_to_orig: dict[str, str] = {}
    for p in sorted(parts_dir.glob("*.png")):
        safe = _safe(p.stem)
        # If two files collapse to the same safe name, the later one wins —
        # that's intended (SAM-extracted version overwrites the original).
        images[safe] = Image.open(p).convert("RGBA")
        stem_to_orig[p.stem] = safe
    if not images:
        raise FileNotFoundError(f"No PNGs in {parts_dir}")

    sheet_w, sheet_h, placements = pack(images, padding)

    atlas = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for stem, (x, y, w, h) in placements.items():
        atlas.paste(images[stem], (x, y))

    img_path = output_dir / f"{name}.png"
    atlas.save(img_path)

    # Spine 4.x atlas format: blank line, then page header (indented props),
    # then regions (name at column 0, props indented).
    lines = [
        "",
        f"{name}.png",
        f"  size: {sheet_w},{sheet_h}",
        "  format: RGBA8888",
        "  filter: Linear,Linear",
        "  repeat: none",
    ]
    for stem, (x, y, w, h) in placements.items():
        lines.extend([
            stem,
            "  rotate: false",
            f"  xy: {x}, {y}",
            f"  size: {w}, {h}",
            f"  orig: {w}, {h}",
            "  offset: 0, 0",
            "  index: -1",
        ])
    atlas_path = output_dir / f"{name}.atlas"
    atlas_path.write_text("\n".join(lines) + "\n")

    return {
        "image": img_path,
        "atlas": atlas_path,
        "sheet_w": sheet_w,
        "sheet_h": sheet_h,
        "placements": {
            stem: {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for stem, (x, y, w, h) in placements.items()
        },
        # Original PNG stem → atlas region name (whitespace sanitized). Use
        # this to find a slot's atlas region when the slot name has spaces.
        "name_map": {orig: safe for orig, safe in stem_to_orig.items()},
    }
