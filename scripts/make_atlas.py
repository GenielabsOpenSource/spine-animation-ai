#!/usr/bin/env python3
"""
make_atlas.py — Pack individual body part PNGs into a Spine-compatible texture atlas.

Usage:
    python3 make_atlas.py --parts parts/ --output atlas/ --name skeleton

Input: Directory of individual .png files (head.png, torso.png, etc.)
Output: skeleton.png (spritesheet) + skeleton.atlas (Spine atlas metadata)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required. Install: pip install Pillow --break-system-packages")
    sys.exit(1)


def next_pow2(v):
    v -= 1
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16
    return max(v + 1, 1)


def pack(images, padding=2):
    """Row-based bin packing. Returns (width, height, placements dict)."""
    sorted_imgs = sorted(images.items(), key=lambda x: -x[1].height)

    total_area = sum(img.width * img.height for img in images.values())
    est = int(math.sqrt(total_area) * 1.3)
    atlas_w = next_pow2(est)

    placements = {}
    rx, ry, rh, max_w = padding, padding, 0, 0

    for name, img in sorted_imgs:
        if rx + img.width + padding > atlas_w:
            rx = padding
            ry += rh + padding
            rh = 0
        placements[name] = (rx, ry, img.width, img.height)
        max_w = max(max_w, rx + img.width + padding)
        rh = max(rh, img.height)
        rx += img.width + padding

    return next_pow2(max_w), next_pow2(ry + rh + padding), placements


def main():
    parser = argparse.ArgumentParser(description="Pack PNGs into a Spine texture atlas")
    parser.add_argument("--parts", required=True, help="Directory with part PNG files")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--name", default="skeleton", help="Base filename for atlas")
    parser.add_argument("--padding", type=int, default=2, help="Pixel padding between regions")
    args = parser.parse_args()

    images = {}
    for f in sorted(os.listdir(args.parts)):
        if f.lower().endswith(".png"):
            name = Path(f).stem
            images[name] = Image.open(os.path.join(args.parts, f)).convert("RGBA")
            print(f"  {name}: {images[name].width}x{images[name].height}")

    if not images:
        print("ERROR: No PNGs found in", args.parts)
        sys.exit(1)

    aw, ah, placements = pack(images, args.padding)
    print(f"Atlas: {aw}x{ah} ({len(images)} regions)")

    # Compose atlas image
    atlas = Image.new("RGBA", (aw, ah), (0, 0, 0, 0))
    for name, (x, y, w, h) in placements.items():
        atlas.paste(images[name], (x, y))

    os.makedirs(args.output, exist_ok=True)
    img_path = os.path.join(args.output, f"{args.name}.png")
    atlas.save(img_path)

    # Write .atlas file
    lines = [f"{args.name}.png", f"size: {aw},{ah}",
             "format: RGBA8888", "filter: Linear,Linear", "repeat: none"]
    for name, (x, y, w, h) in placements.items():
        lines += [name, "  rotate: false", f"  xy: {x}, {y}",
                  f"  size: {w}, {h}", f"  orig: {w}, {h}",
                  "  offset: 0, 0", "  index: -1"]

    atlas_path = os.path.join(args.output, f"{args.name}.atlas")
    with open(atlas_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {img_path}, {atlas_path}")

if __name__ == "__main__":
    main()
