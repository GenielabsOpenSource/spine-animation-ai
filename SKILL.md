---
name: spine-animation
description: >
  Create Spine 2D skeletal animations from pre-existing character assets (separated body-part PNGs,
  atlas spritesheet, or a full character image). Use this skill whenever the user wants to animate
  a 2D character, create Spine JSON from existing art assets, rig a character with bones, build
  walk/idle/run/attack animations, produce an interactive Spine Web Player preview, or generate
  Spine-compatible export files (.json + .atlas + .png). Also trigger when the user mentions
  "Spine animation", "2D rigging", "skeletal animation", "bone animation", "cutout animation",
  "animate this character", "make this walk", "create walk cycle", or uploads separated character
  body parts and wants them animated. This skill handles the full pipeline: asset analysis,
  skeleton rigging, animation keyframing, Spine JSON export, and interactive HTML5 preview.
---
<!-- ⚠️  AUTO-GENERATED FILE — DO NOT EDIT DIRECTLY
     Edit SKILL.template.md instead, then push to trigger rebuild.
     Scripts are embedded automatically by build_skill.py via GitHub Actions. -->



# Spine Animation Skill

Turn pre-existing 2D character assets into fully animated, interactive Spine animations.

## Step 0: Set Up Scripts

This skill includes Python scripts that do the heavy lifting. Claude MUST write them to disk
before use. Each script is embedded below — Claude should save them to `/home/claude/spine-scripts/`
at the start of every session.

```bash
mkdir -p /home/claude/spine-scripts
pip install opencv-python Pillow numpy --break-system-packages -q
```

### Embedded Scripts

The following scripts are auto-injected from the repository's `scripts/` directory.
**Claude: read these carefully, then write each one to `/home/claude/spine-scripts/`**
before running the pipeline.

<!-- EMBED:scripts/position_parts.py -->
<details>
<summary>📄 <code>scripts/position_parts.py</code> (492 lines)</summary>

```python
#!/usr/bin/env python3
"""
position_parts.py — Part positioning via SIFT + RANSAC homography, z-order via occlusion.

Given a fully assembled character image and individual body-part PNGs,
determines where each part goes (x, y, scale, rotation) and the draw order.

Algorithm:
  Phase 1 — SIFT keypoint matching + RANSAC homography
    - Extract SIFT features from each part (alpha-masked) and the reference
    - Match descriptors via FLANN (knnMatch + Lowe's ratio test)
    - Estimate homography via RANSAC → extract position, scale, rotation
    - For small/low-texture parts that fail SIFT: fall back to template matching

  Phase 2 — Pairwise occlusion voting for z-order
    - Sample overlap pixels, compare to reference → occlusion graph → topo sort

Usage:
  python3 position_parts.py \
    --reference character.png \
    --parts parts_folder/ \
    --output layout.json \
    [--min-matches 4] \
    [--ratio 0.80] \
    [--debug debug_folder/]
"""

import argparse, json, os, sys, math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def load_rgba(path):
    return np.array(Image.open(path).convert("RGBA"))

def create_foreground_mask(rgba, bg_color=(255,255,255), bg_threshold=30):
    alpha = rgba[:, :, 3]
    is_opaque = alpha > 128
    rgb = rgba[:, :, :3].astype(float)
    dist = np.sqrt(np.sum((rgb - np.array(bg_color, dtype=float)) ** 2, axis=2))
    mask = (is_opaque & (dist > bg_threshold)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)


# ─────────────────────────────────────────────────────────────────
# Phase 1: SIFT + RANSAC
# ─────────────────────────────────────────────────────────────────

def sift_match_part(ref_gray, ref_kp, ref_des, part_rgba,
                    sift, ratio_thresh=0.80, min_matches=4):
    """
    Match a part to the reference using SIFT + FLANN + RANSAC affine transform.
    Uses estimateAffinePartial2D (4 DOF: translate + scale + rotation) instead
    of full homography — much more robust with sparse matches on game art.
    Returns dict with position/scale/rotation/score, or None.
    """
    part_h, part_w = part_rgba.shape[:2]
    part_gray = cv2.cvtColor(part_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    part_mask = (part_rgba[:, :, 3] > 128).astype(np.uint8) * 255

    part_kp, part_des = sift.detectAndCompute(part_gray, part_mask)
    if part_des is None or len(part_kp) < 2:
        return None

    # FLANN matching
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=150))
    try:
        matches = flann.knnMatch(part_des, ref_des, k=2)
    except cv2.error:
        return None

    # Lowe's ratio test
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good.append(pair[0])

    if len(good) < min_matches:
        return None

    src_pts = np.float32([part_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC similarity transform (4 DOF: translate + uniform scale + rotation)
    # This is much more constrained than homography (8 DOF) and needs only 2 points
    M, inliers_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M is None or inliers_mask is None:
        return None

    inliers = int(inliers_mask.sum())
    if inliers < min_matches:
        return None

    # Extract scale and rotation from 2x3 affine matrix
    # M = [[s*cos(θ), -s*sin(θ), tx], [s*sin(θ), s*cos(θ), ty]]
    scale = np.sqrt(M[0,0]**2 + M[1,0]**2)
    rotation = math.degrees(math.atan2(M[1,0], M[0,0]))

    # Sanity: game parts should be ~0.5–2.0x scale, ~0° rotation
    if scale < 0.3 or scale > 3.0:
        return None
    if abs(rotation) > 20:
        return None

    # Transform corners via the affine matrix
    corners = np.float32([[0,0],[part_w,0],[part_w,part_h],[0,part_h]]).reshape(-1,1,2)
    transformed = cv2.transform(corners, M).reshape(-1, 2)

    x_min, y_min = transformed[:, 0].min(), transformed[:, 1].min()
    x_max, y_max = transformed[:, 0].max(), transformed[:, 1].max()
    out_w, out_h = x_max - x_min, y_max - y_min

    if out_w < 5 or out_h < 5:
        return None

    inlier_ratio = inliers / len(good) if good else 0

    return {
        "x": int(round(x_min)), "y": int(round(y_min)),
        "width": int(round(out_w)), "height": int(round(out_h)),
        "original_width": part_w, "original_height": part_h,
        "scale": round(scale, 4), "rotation": round(rotation, 2),
        "score": round(inlier_ratio, 4),
        "n_matches": inliers, "n_good": len(good),
        "n_keypoints": len(part_kp), "method": "sift",
    }


def template_match_fallback(ref_bgr, ref_fg_mask, part_bgra,
                            scales=None):
    """Fallback for parts too small/featureless for SIFT."""
    if scales is None:
        scales = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    ref_h, ref_w = ref_bgr.shape[:2]
    best = None

    for scale in scales:
        sw = max(1, int(part_bgra.shape[1] * scale))
        sh = max(1, int(part_bgra.shape[0] * scale))
        if sw >= ref_w - 2 or sh >= ref_h - 2:
            continue

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(part_bgra, (sw, sh), interpolation=interp)
        tmpl_bgr = cv2.cvtColor(scaled, cv2.COLOR_BGRA2BGR)
        mask = (scaled[:, :, 3] > 128).astype(np.uint8) * 255
        opaque = np.count_nonzero(mask)
        if opaque < 20:
            continue

        try:
            result = cv2.matchTemplate(ref_bgr, tmpl_bgr, cv2.TM_CCORR_NORMED, mask=mask)
        except cv2.error:
            continue

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        fg_region = ref_fg_mask[max_loc[1]:max_loc[1]+sh, max_loc[0]:max_loc[0]+sw]
        fg_ratio = 0.0
        if fg_region.shape == (sh, sw):
            fg_ratio = np.count_nonzero(fg_region[mask > 128] > 128) / max(1, opaque)

        combined = max_val * (0.3 + 0.7 * fg_ratio)

        if best is None or combined > best["score"]:
            best = {
                "x": int(max_loc[0]), "y": int(max_loc[1]),
                "width": sw, "height": sh,
                "original_width": part_bgra.shape[1], "original_height": part_bgra.shape[0],
                "scale": round(scale, 4), "rotation": 0.0,
                "score": round(combined, 4),
                "n_matches": 0, "n_good": 0, "n_keypoints": 0,
                "method": "template",
            }
    return best


def find_all_positions(reference_path, parts_folder, ratio_thresh, min_matches):
    ref_rgba = load_rgba(reference_path)
    ref_gray = cv2.cvtColor(ref_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    ref_bgra = cv2.cvtColor(ref_rgba, cv2.COLOR_RGBA2BGRA)
    ref_bgr = cv2.cvtColor(ref_bgra, cv2.COLOR_BGRA2BGR)

    fg_mask = create_foreground_mask(ref_rgba)

    # Tuned SIFT: lower contrast threshold to find more features on game art
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=20)

    print("Computing SIFT on reference...")
    ref_kp, ref_des = sift.detectAndCompute(ref_gray, None)
    print(f"  Reference: {ref_gray.shape[1]}x{ref_gray.shape[0]}, {len(ref_kp)} keypoints\n")

    part_files = sorted([f for f in os.listdir(parts_folder) if f.lower().endswith(('.png','.webp'))])

    # First pass: try SIFT on all parts
    sift_results = {}
    failed_parts = []
    for fname in part_files:
        name = Path(fname).stem
        part_rgba = load_rgba(os.path.join(parts_folder, fname))
        if np.count_nonzero(part_rgba[:,:,3] > 128) / part_rgba[:,:,3].size < 0.01:
            print(f"  SKIP {name}: <1% opaque")
            continue

        result = sift_match_part(ref_gray, ref_kp, ref_des, part_rgba,
                                 sift, ratio_thresh, min_matches)
        if result:
            sift_results[name] = result
            print(f"  SIFT {name:>20}: pos=({result['x']},{result['y']}) "
                  f"scale={result['scale']:.3f} rot={result['rotation']:.1f}° "
                  f"inliers={result['n_matches']}/{result['n_good']} "
                  f"score={result['score']:.3f}")
        else:
            failed_parts.append((name, part_rgba))

    # Derive template matching scales from SIFT results
    tmpl_scales = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
    if sift_results:
        sift_scales = [r["scale"] for r in sift_results.values()]
        median_scale = float(np.median(sift_scales))
        # Generate scale range around the SIFT median: ±20%
        tmpl_scales = tuple(round(median_scale * f, 4)
                            for f in (0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20))
        print(f"\n  SIFT median scale: {median_scale:.3f} → template range: "
              f"{tmpl_scales[0]:.3f}–{tmpl_scales[-1]:.3f}")

    # Second pass: template matching for failed parts using SIFT-derived scales
    positions = dict(sift_results)
    for name, part_rgba in failed_parts:
        part_bgra = cv2.cvtColor(part_rgba, cv2.COLOR_RGBA2BGRA)
        result = template_match_fallback(ref_bgr, fg_mask, part_bgra, scales=tmpl_scales)
        if result:
            positions[name] = result
            print(f"  TMPL {name:>20}: pos=({result['x']},{result['y']}) "
                  f"scale={result['scale']:.3f} score={result['score']:.3f}")
        else:
            print(f"  FAIL {name:>20}: no match")

    return positions, fg_mask


# ─────────────────────────────────────────────────────────────────
# Phase 2: Z-Order via Occlusion
# ─────────────────────────────────────────────────────────────────

def compute_z_order(reference_path, parts_folder, positions):
    reference = load_rgba(reference_path)
    ref_h, ref_w = reference.shape[:2]

    part_images = {}
    for name, pos in positions.items():
        fp = None
        for ext in ['.png','.webp']:
            c = os.path.join(parts_folder, name+ext)
            if os.path.exists(c): fp = c; break
        if not fp: continue
        img = load_rgba(fp)
        tw, th = pos["width"], pos["height"]
        if (tw, th) != (img.shape[1], img.shape[0]):
            img = np.array(Image.fromarray(img).resize((tw, th), Image.LANCZOS))
        part_images[name] = img

    names = list(part_images.keys())
    n = len(names)
    wins = defaultdict(lambda: defaultdict(int))

    print(f"\nZ-order analysis ({n} parts):")
    for i in range(n):
        for j in range(i+1, n):
            a, b = names[i], names[j]
            ap, bp = positions[a], positions[b]
            ai, bi = part_images[a], part_images[b]

            ox1 = max(ap["x"], bp["x"])
            oy1 = max(ap["y"], bp["y"])
            ox2 = min(ap["x"]+ap["width"], bp["x"]+bp["width"])
            oy2 = min(ap["y"]+ap["height"], bp["y"]+bp["height"])
            if ox1 >= ox2 or oy1 >= oy2: continue

            step = max(1, int(math.sqrt((ox2-ox1)*(oy2-oy1)/500)))
            aw, bw, tot = 0, 0, 0

            for sy in range(oy1, oy2, step):
                for sx in range(ox1, ox2, step):
                    if sy >= ref_h or sx >= ref_w: continue
                    rp = reference[sy, sx]
                    if rp[3] < 128: continue
                    aly, alx = sy-ap["y"], sx-ap["x"]
                    bly, blx = sy-bp["y"], sx-bp["x"]
                    if not (0<=alx<ai.shape[1] and 0<=aly<ai.shape[0]): continue
                    if not (0<=blx<bi.shape[1] and 0<=bly<bi.shape[0]): continue
                    apx, bpx = ai[aly, alx], bi[bly, blx]
                    if apx[3] < 128 or bpx[3] < 128: continue
                    ad = np.sqrt(np.sum((rp[:3].astype(float)-apx[:3].astype(float))**2))
                    bd = np.sqrt(np.sum((rp[:3].astype(float)-bpx[:3].astype(float))**2))
                    tot += 1
                    if ad < bd - 5: aw += 1
                    elif bd < ad - 5: bw += 1

            if tot > 5:
                if aw > bw * 1.2:
                    wins[a][b] += aw
                    print(f"  {a} OVER {b} ({aw}/{tot})")
                elif bw > aw * 1.2:
                    wins[b][a] += bw
                    print(f"  {b} OVER {a} ({bw}/{tot})")

    depth = {nm: 0.0 for nm in names}
    for a in names:
        for b in names:
            if a != b and wins[a][b] > 0:
                depth[b] -= wins[a][b]
                depth[a] += wins[a][b]

    result = sorted(names, key=lambda nm: depth[nm])
    print(f"\nDraw order (back -> front):")
    for i, nm in enumerate(result):
        print(f"  z={i:>2}: {nm} (depth={depth[nm]:.0f}, {positions[nm]['method']})")
    return result, depth


# ─────────────────────────────────────────────────────────────────
# Debug Visualization
# ─────────────────────────────────────────────────────────────────

def generate_debug(ref_path, parts_folder, positions, z_order, fg_mask, debug_dir):
    os.makedirs(debug_dir, exist_ok=True)
    ref = load_rgba(ref_path)
    rh, rw = ref.shape[:2]

    # Composite
    comp = np.zeros((rh, rw, 4), dtype=np.uint8)
    comp[:,:,:3] = 255; comp[:,:,3] = 255

    for name in z_order:
        if name not in positions: continue
        pos = positions[name]
        fp = None
        for ext in ['.png','.webp']:
            c = os.path.join(parts_folder, name+ext)
            if os.path.exists(c): fp = c; break
        if not fp: continue
        img = load_rgba(fp)
        tw, th = pos["width"], pos["height"]
        if (tw, th) != (img.shape[1], img.shape[0]):
            img = np.array(Image.fromarray(img).resize((tw, th), Image.LANCZOS))

        x, y = pos["x"], pos["y"]
        ph, pw = img.shape[:2]
        sx1, sy1 = max(0,-x), max(0,-y)
        dx1, dy1 = max(0,x), max(0,y)
        sx2, sy2 = min(pw, rw-x), min(ph, rh-y)
        dx2, dy2 = dx1+(sx2-sx1), dy1+(sy2-sy1)
        if sx2<=sx1 or sy2<=sy1: continue

        pr = img[sy1:sy2, sx1:sx2]
        a = pr[:,:,3:4].astype(float)/255.0
        cr = comp[dy1:dy2, dx1:dx2, :3].astype(float)
        comp[dy1:dy2, dx1:dx2, :3] = (pr[:,:,:3].astype(float)*a + cr*(1-a)).astype(np.uint8)

    Image.fromarray(comp).save(os.path.join(debug_dir, "composite.png"))

    # Side-by-side
    gap = 10
    sb = np.zeros((rh, rw*2+gap, 4), dtype=np.uint8)
    sb[:,:,:3]=40; sb[:,:,3]=255
    sb[:rh,:rw] = ref; sb[:rh,rw+gap:rw*2+gap] = comp
    Image.fromarray(sb).save(os.path.join(debug_dir, "comparison.png"))

    # Bboxes
    rv = ref.copy()
    colors = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),
              (80,255,255),(200,140,80),(140,80,200),(80,200,140),
              (255,160,120),(120,255,160),(160,120,255),(200,200,100)]
    for i, (name, pos) in enumerate(positions.items()):
        c = colors[i % len(colors)]
        m = pos["method"][0].upper()
        x1, y1 = pos["x"], pos["y"]
        x2, y2 = x1+pos["width"], y1+pos["height"]
        cv2.rectangle(rv, (x1,y1), (x2,y2), c+(255,), 2)
        label = f"{name} [{m}] s={pos['scale']:.2f} m={pos.get('n_matches',0)}"
        cv2.putText(rv, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c+(255,), 1)
    Image.fromarray(rv).save(os.path.join(debug_dir, "bboxes.png"))

    # FG mask
    Image.fromarray(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGBA)).save(
        os.path.join(debug_dir, "fg_mask.png"))

    # Per-part SIFT match visualizations
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=20)
    ref_gray = cv2.cvtColor(ref[:,:,:3], cv2.COLOR_RGB2GRAY)
    ref_kp, ref_des = sift.detectAndCompute(ref_gray, None)

    for name, pos in positions.items():
        if pos["method"] != "sift": continue
        fp = None
        for ext in ['.png','.webp']:
            c = os.path.join(parts_folder, name+ext)
            if os.path.exists(c): fp = c; break
        if not fp: continue

        prgba = load_rgba(fp)
        pgray = cv2.cvtColor(prgba[:,:,:3], cv2.COLOR_RGB2GRAY)
        pmask = (prgba[:,:,3] > 128).astype(np.uint8) * 255
        pkp, pdes = sift.detectAndCompute(pgray, pmask)
        if pdes is None: continue

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=150))
        matches = flann.knnMatch(pdes, ref_des, k=2)
        good = [m for m, n in matches if len([m,n])==2 and m.distance < 0.8*n.distance]

        if len(good) >= 4:
            src = np.float32([pkp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, hmask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if hmask is not None:
                draw_p = dict(matchColor=(0,255,0), singlePointColor=(255,0,0),
                              matchesMask=hmask.ravel().tolist(),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                pbgr = cv2.cvtColor(prgba[:,:,:3], cv2.COLOR_RGB2BGR)
                rbgr = cv2.cvtColor(ref[:,:,:3], cv2.COLOR_RGB2BGR)
                vis = cv2.drawMatches(pbgr, pkp, rbgr, ref_kp, good, None, **draw_p)
                cv2.imwrite(os.path.join(debug_dir, f"sift_{name}.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 70])

    print(f"\nDebug saved to {debug_dir}/")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Position parts via SIFT+RANSAC homography + occlusion z-order")
    p.add_argument("--reference", required=True)
    p.add_argument("--parts", required=True)
    p.add_argument("--output", default="layout.json")
    p.add_argument("--min-matches", type=int, default=4,
                   help="Min RANSAC inliers (default: 4)")
    p.add_argument("--ratio", type=float, default=0.80,
                   help="Lowe's ratio threshold (default: 0.80)")
    p.add_argument("--debug", default=None)
    args = p.parse_args()

    print("=" * 60)
    print("PHASE 1: SIFT + RANSAC Homography")
    print("=" * 60)
    positions, fg_mask = find_all_positions(
        args.reference, args.parts, args.ratio, args.min_matches)

    if not positions:
        print("ERROR: No parts matched!"); sys.exit(1)

    sift_n = sum(1 for p in positions.values() if p["method"] == "sift")
    tmpl_n = sum(1 for p in positions.values() if p["method"] == "template")
    print(f"\nResult: {sift_n} SIFT, {tmpl_n} template fallback")

    print(f"\n{'='*60}")
    print("PHASE 2: Z-Order (Occlusion Analysis)")
    print("="*60)
    z_order, depth = compute_z_order(args.reference, args.parts, positions)

    for i, name in enumerate(z_order):
        if name in positions:
            positions[name]["z_index"] = i
            positions[name]["depth_score"] = depth[name]

    ref_img = Image.open(args.reference)
    output = {
        "reference_image": os.path.basename(args.reference),
        "canvas_width": ref_img.width, "canvas_height": ref_img.height,
        "parts": positions, "z_order": z_order,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nLayout saved: {args.output}")

    if args.debug:
        generate_debug(args.reference, args.parts, positions, z_order, fg_mask, args.debug)


if __name__ == "__main__":
    main()
```

</details>

<!-- EMBED:scripts/build_spine_json.py -->
<details>
<summary>📄 <code>scripts/build_spine_json.py</code> (463 lines)</summary>

```python
#!/usr/bin/env python3
"""
build_spine_json.py — Generate a complete Spine-compatible JSON skeleton with animations.

Accepts a config describing bones, slots, attachments, and desired animations,
then outputs a valid Spine 4.2 JSON file.

Usage:
    python3 build_spine_json.py --config config.json --output skeleton.json

Config JSON format:
{
  "skeleton": {
    "name": "my-character",
    "width": 400,
    "height": 600
  },
  "bones": [
    { "name": "root" },
    { "name": "hip", "parent": "root", "x": 0, "y": 200, "length": 30 },
    { "name": "torso", "parent": "hip", "length": 120 },
    ...
  ],
  "slots": [
    { "name": "torso", "bone": "torso", "attachment": "torso" },
    ...
  ],
  "attachments": {
    "torso": { "width": 120, "height": 200, "x": 0, "y": 60 },
    ...
  },
  "animations": ["idle", "walk", "wave", "jump", "run", "attack"],
  "custom_animations": {
    "my-custom": { "bones": { "head": { "rotate": [...] } } }
  }
}
"""

import argparse
import json
import hashlib
import sys

# ─── Bezier curve presets ────────────────────────────────────────────────────
EASE = [0.25, 0, 0.75, 1]        # Standard ease in-out (most common)
EASE_IN = [0.42, 0, 1, 1]        # Accelerate from rest
EASE_OUT = [0, 0, 0.58, 1]       # Decelerate to rest
EASE_BOUNCE = [0.34, 1.56, 0.64, 1]  # Slight overshoot
EASE_FAST = [0.4, 0, 0.2, 1]     # Quick but smooth

def _has(bone_names, *names):
    """Check if any of the given bone names exist."""
    return any(n in bone_names for n in names)

def _kf(time, angle=None, x=None, y=None, curve=EASE):
    """Build a keyframe dict, omitting None values."""
    kf = {"time": round(time, 4)}
    if angle is not None:
        kf["angle"] = round(angle, 2)
    if x is not None:
        kf["x"] = round(x, 2)
    if y is not None:
        kf["y"] = round(y, 2)
    if curve:
        kf["curve"] = curve
    return kf


# ─── Animation Generators ───────────────────────────────────────────────────

def gen_idle(B):
    """Idle breathing/sway. Subtle, loopable. ~1.6s"""
    bones = {}
    D = 1.6

    for name, angle_amp, phase in [
        ("torso", 1.5, 0.5), ("neck", 1.0, 0.55), ("head", -2.0, 0.6)
    ]:
        if name in B:
            bones[name] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(D * phase, angle_amp),
                _kf(D, 0),
            ]}

    # Subtle torso lift
    if "torso" in B:
        bones.setdefault("torso", {})["translate"] = [
            _kf(0, x=0, y=0, curve=None),
            _kf(D * 0.5, x=0, y=1.5),
            _kf(D, x=0, y=0),
        ]

    # Gentle arm sway
    for side in ["left", "right"]:
        s = 1 if side == "left" else -1
        for part, amp, ph in [
            (f"{side}-upper-arm", s * 1.5, 0.5),
            (f"{side}-lower-arm", s * 1.0, 0.55),
        ]:
            if part in B:
                bones[part] = {"rotate": [
                    _kf(0, 0, curve=None), _kf(D * ph, amp), _kf(D, 0),
                ]}

    return {"bones": bones} if bones else {}


def gen_walk(B):
    """Walk cycle. Opposing arm-leg motion, hip bob. ~0.8s"""
    bones = {}
    D = 0.8
    Q = D / 4  # quarter

    if "hip" in B:
        bones["hip"] = {
            "translate": [
                _kf(0, x=0, y=0, curve=None),
                _kf(Q, x=0, y=3), _kf(Q*2, x=0, y=0),
                _kf(Q*3, x=0, y=3), _kf(D, x=0, y=0),
            ],
            "rotate": [
                _kf(0, 0, curve=None),
                _kf(Q, -2), _kf(Q*2, 0), _kf(Q*3, 2), _kf(D, 0),
            ],
        }

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(Q, 3), _kf(Q*2, 0), _kf(Q*3, -3), _kf(D, 0),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(Q, -1.5), _kf(Q*2, 0), _kf(Q*3, 1.5), _kf(D, 0),
        ]}

    # Legs: left forward at t=0, right forward at t=D/2
    for side, phase_shift in [("left", 0), ("right", 0.5)]:
        p = phase_shift * D
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        foot = f"{side}-foot"

        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -25 if phase_shift == 0 else 25, curve=None),
                _kf(Q, 0), _kf(Q*2, 25 if phase_shift == 0 else -25),
                _kf(Q*3, 0), _kf(D, -25 if phase_shift == 0 else 25),
            ]}
        if lower in B:
            # Back leg straight, front leg bent
            bones[lower] = {"rotate": [
                _kf(0, 5 if phase_shift == 0 else 35, curve=None),
                _kf(Q, 35), _kf(Q*2, 35 if phase_shift == 0 else 5),
                _kf(Q*3, 5), _kf(D, 5 if phase_shift == 0 else 35),
            ]}

    # Arms: oppose legs
    for side, phase_shift in [("left", 0.5), ("right", 0)]:
        upper = f"{side}-upper-arm"
        lower = f"{side}-lower-arm"

        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -20 if phase_shift == 0 else 20, curve=None),
                _kf(Q, 0), _kf(Q*2, 20 if phase_shift == 0 else -20),
                _kf(Q*3, 0), _kf(D, -20 if phase_shift == 0 else 20),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, -10 if phase_shift == 0 else -30, curve=None),
                _kf(Q, -20), _kf(Q*2, -30 if phase_shift == 0 else -10),
                _kf(Q*3, -20), _kf(D, -10 if phase_shift == 0 else -30),
            ]}

    return {"bones": bones} if bones else {}


def gen_run(B):
    """Run cycle. Exaggerated walk, forward lean, bigger bounce. ~0.5s"""
    bones = {}
    D = 0.5
    Q = D / 4

    if "hip" in B:
        bones["hip"] = {
            "translate": [
                _kf(0, x=0, y=0, curve=None),
                _kf(Q, x=0, y=6), _kf(Q*2, x=0, y=-2),
                _kf(Q*3, x=0, y=6), _kf(D, x=0, y=0),
            ],
            "rotate": [
                _kf(0, 0, curve=None),
                _kf(Q, -3), _kf(Q*2, 0), _kf(Q*3, 3), _kf(D, 0),
            ],
        }

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 8, curve=None),  # Constant forward lean
            _kf(Q, 12), _kf(Q*2, 8), _kf(Q*3, 12), _kf(D, 8),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, -6, curve=None),  # Compensate for torso lean
            _kf(Q, -8), _kf(Q*2, -6), _kf(Q*3, -8), _kf(D, -6),
        ]}

    for side, ph in [("left", 0), ("right", 0.5)]:
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -35 if ph == 0 else 40, curve=None),
                _kf(Q, 0), _kf(Q*2, 40 if ph == 0 else -35),
                _kf(Q*3, 0), _kf(D, -35 if ph == 0 else 40),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, 10 if ph == 0 else 50, curve=None),
                _kf(Q, 50), _kf(Q*2, 50 if ph == 0 else 10),
                _kf(Q*3, 10), _kf(D, 10 if ph == 0 else 50),
            ]}

    for side, ph in [("left", 0.5), ("right", 0)]:
        upper = f"{side}-upper-arm"
        lower = f"{side}-lower-arm"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -30 if ph == 0 else 30, curve=None),
                _kf(Q, 0), _kf(Q*2, 30 if ph == 0 else -30),
                _kf(Q*3, 0), _kf(D, -30 if ph == 0 else 30),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, -20 if ph == 0 else -50, curve=None),
                _kf(Q, -35), _kf(Q*2, -50 if ph == 0 else -20),
                _kf(Q*3, -35), _kf(D, -20 if ph == 0 else -50),
            ]}

    return {"bones": bones} if bones else {}


def gen_wave(B):
    """Waving greeting. Raise right arm, oscillate forearm. ~1.2s"""
    bones = {}
    D = 1.2

    if "right-upper-arm" in B:
        bones["right-upper-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -130, curve=EASE_OUT),
            _kf(D - 0.2, -130, curve=None),
            _kf(D, 0, curve=EASE_IN),
        ]}
    if "right-lower-arm" in B:
        bones["right-lower-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -30, curve=EASE_OUT),
            _kf(0.4, 20), _kf(0.6, -20), _kf(0.8, 20), _kf(1.0, -20),
            _kf(D, 0, curve=EASE_IN),
        ]}
    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -3), _kf(D - 0.2, -3, curve=None), _kf(D, 0),
        ]}
    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.3, 5), _kf(D - 0.2, 5, curve=None), _kf(D, 0),
        ]}

    return {"bones": bones} if bones else {}


def gen_jump(B):
    """Jump: anticipation squat → launch → air → land → settle. ~1.0s"""
    bones = {}
    D = 1.0

    if "hip" in B:
        bones["hip"] = {"translate": [
            _kf(0, x=0, y=0, curve=None),
            _kf(0.15, x=0, y=-20, curve=EASE_IN),     # squat
            _kf(0.35, x=0, y=70, curve=EASE_OUT),      # launch
            _kf(0.55, x=0, y=65, curve=None),           # float
            _kf(0.80, x=0, y=-10, curve=EASE_IN),       # land impact
            _kf(D, x=0, y=0, curve=EASE_OUT),            # settle
        ]}

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.15, 8, curve=EASE_IN),     # lean forward in squat
            _kf(0.35, -5, curve=EASE_OUT),    # extend in air
            _kf(0.80, 5, curve=EASE_IN),      # absorb landing
            _kf(D, 0, curve=EASE_OUT),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.15, 5), _kf(0.35, -8), _kf(0.80, 3), _kf(D, 0),
        ]}

    for side in ["left", "right"]:
        s = 1 if side == "left" else -1
        arm = f"{side}-upper-arm"
        if arm in B:
            bones[arm] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, s*10), _kf(0.35, s*-50, curve=EASE_OUT),
                _kf(0.80, s*8, curve=EASE_IN), _kf(D, 0),
            ]}
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, 20),   # squat bend
                _kf(0.35, -15),  # extend
                _kf(0.55, 10),   # tuck in air
                _kf(0.80, 15),   # absorb
                _kf(D, 0),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, -30), _kf(0.35, 10), _kf(0.55, -15),
                _kf(0.80, -20), _kf(D, 0),
            ]}

    return {"bones": bones} if bones else {}


def gen_attack(B):
    """Melee attack: wind-up → strike → follow-through. ~0.6s"""
    bones = {}
    D = 0.6

    if "right-upper-arm" in B:
        bones["right-upper-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, 40, curve=EASE_IN),      # wind up (pull back)
            _kf(0.25, -80, curve=EASE_FAST),   # strike forward
            _kf(0.4, -60, curve=None),          # follow through
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "right-lower-arm" in B:
        bones["right-lower-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, -40, curve=EASE_IN),
            _kf(0.25, 10, curve=EASE_FAST),
            _kf(0.4, -5, curve=None),
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, -8, curve=EASE_IN),   # lean back
            _kf(0.25, 12, curve=EASE_FAST), # lunge forward
            _kf(0.4, 5, curve=None),
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "hip" in B:
        bones["hip"] = {"translate": [
            _kf(0, x=0, y=0, curve=None),
            _kf(0.1, x=-5, y=-5, curve=EASE_IN),
            _kf(0.25, x=10, y=2, curve=EASE_FAST),
            _kf(D, x=0, y=0, curve=EASE_OUT),
        ]}

    return {"bones": bones} if bones else {}


PRESETS = {
    "idle": gen_idle,
    "walk": gen_walk,
    "run": gen_run,
    "wave": gen_wave,
    "jump": gen_jump,
    "attack": gen_attack,
}


# ─── Spine JSON Builder ──────────────────────────────────────────────────────

def build_spine_json(config):
    """Build a complete Spine JSON structure from config."""
    bone_names = {b["name"] for b in config["bones"]}

    skel_meta = config.get("skeleton", {})
    data_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]

    spine = {
        "skeleton": {
            "hash": data_hash,
            "spine": "4.2.0",
            "x": -(skel_meta.get("width", 400) // 2),
            "y": 0,
            "width": skel_meta.get("width", 400),
            "height": skel_meta.get("height", 600),
            "images": "./images/",
        },
        "bones": config["bones"],
        "slots": config.get("slots", []),
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {},
    }

    # Build attachments for default skin
    attachments = config.get("attachments", {})
    for slot in config.get("slots", []):
        att_name = slot.get("attachment", slot["name"])
        if att_name in attachments:
            spine["skins"][0]["attachments"][slot["name"]] = {
                att_name: attachments[att_name]
            }

    # Generate preset animations
    for anim_name in config.get("animations", ["idle"]):
        if anim_name in PRESETS:
            data = PRESETS[anim_name](bone_names)
            if data:
                spine["animations"][anim_name] = data
        else:
            print(f"  WARNING: Unknown animation preset '{anim_name}', skipping")

    # Merge custom animations
    for name, data in config.get("custom_animations", {}).items():
        spine["animations"][name] = data

    return spine


def main():
    parser = argparse.ArgumentParser(description="Build Spine JSON skeleton with animations")
    parser.add_argument("--config", required=True, help="Skeleton configuration JSON")
    parser.add_argument("--output", default="skeleton.json", help="Output Spine JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Building: {config.get('skeleton', {}).get('name', 'unnamed')}")
    spine_json = build_spine_json(config)

    with open(args.output, "w") as f:
        json.dump(spine_json, f, indent=2)

    print(f"Saved: {args.output}")
    print(f"  Bones: {len(spine_json['bones'])}")
    print(f"  Slots: {len(spine_json['slots'])}")
    print(f"  Animations: {list(spine_json['animations'].keys())}")


if __name__ == "__main__":
    main()
```

</details>

<!-- EMBED:scripts/make_atlas.py -->
<details>
<summary>📄 <code>scripts/make_atlas.py</code> (102 lines)</summary>

```python
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
```

</details>

<!-- EMBED:scripts/generate_spine_player.py -->
<details>
<summary>📄 <code>scripts/generate_spine_player.py</code> (292 lines)</summary>

```python
#!/usr/bin/env python3
"""
generate_spine_player.py — Generate a self-contained HTML preview using the official Spine Web Player.

Embeds the skeleton JSON, atlas text, and atlas PNG as base64 data URIs via the
rawDataURIs configuration, so the resulting HTML file works standalone — no server needed.

Uses the official @esotericsoftware/spine-player from UNPKG CDN.

Usage:
    python3 generate_spine_player.py \
        --skeleton skeleton.json \
        --atlas skeleton.atlas \
        --atlas-image skeleton.png \
        --output preview.html \
        [--animation idle] \
        [--background "#1a1a2eff"] \
        [--skin default]

If no --atlas and --atlas-image are given but a --parts directory is provided,
the script will pack the parts into an atlas automatically.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path


def file_to_base64(path):
    """Read a file and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def json_to_base64(path):
    """Read a JSON file and return it as base64."""
    with open(path, "r") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


def text_to_base64(path):
    """Read a text file and return it as base64."""
    with open(path, "r") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


def find_atlas_images(atlas_path):
    """Parse an atlas file to find all referenced PNG filenames."""
    atlas_dir = os.path.dirname(os.path.abspath(atlas_path))
    images = []

    with open(atlas_path, "r") as f:
        lines = f.readlines()

    # The first line (or lines before the first region entry) contain page image filenames
    # Atlas format: image filename is a line that ends with .png (or other image ext)
    # followed by size:, format:, filter:, repeat: lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # A page image is the first non-empty line, or any line ending with an image extension
        # that is followed by "size:" on the next line
        if line and not line.startswith(" ") and not ":" in line:
            # Check if next line starts with "size:" indicating this is a page name
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("size:"):
                img_path = os.path.join(atlas_dir, line)
                if os.path.exists(img_path):
                    images.append((line, img_path))
                else:
                    print(f"  WARNING: Atlas references '{line}' but file not found at {img_path}")
        i += 1

    return images


def generate_html(skeleton_path, atlas_path, atlas_images,
                  animation=None, skin=None, bg_color="#1a1a2eff",
                  show_controls=True, title="Spine Animation Preview"):
    """Generate the complete HTML file with embedded Spine Web Player."""

    # Get filenames for rawDataURIs keys
    skel_filename = os.path.basename(skeleton_path)
    atlas_filename = os.path.basename(atlas_path)

    # Determine if JSON or binary
    is_json = skel_filename.lower().endswith(".json")
    skel_mime = "application/json" if is_json else "application/octet-stream"

    # Encode all assets
    skel_b64 = file_to_base64(skeleton_path)
    atlas_b64 = file_to_base64(atlas_path)

    # Build rawDataURIs object
    raw_data_entries = []
    raw_data_entries.append(
        f'        "{skel_filename}": "data:{skel_mime};base64,{skel_b64}"'
    )
    raw_data_entries.append(
        f'        "{atlas_filename}": "data:application/octet-stream;base64,{atlas_b64}"'
    )

    for img_name, img_path in atlas_images:
        img_ext = Path(img_path).suffix.lower()
        img_mime = "image/png" if img_ext == ".png" else "image/jpeg"
        img_b64 = file_to_base64(img_path)
        raw_data_entries.append(
            f'        "{img_name}": "data:{img_mime};base64,{img_b64}"'
        )

    raw_data_uris_js = ",\n".join(raw_data_entries)

    # Build config options
    config_lines = []
    config_lines.append(f'      skeleton: "{skel_filename}"')
    config_lines.append(f'      atlas: "{atlas_filename}"')

    if animation:
        config_lines.append(f'      animation: "{animation}"')

    if skin and skin != "default":
        config_lines.append(f'      skin: "{skin}"')

    config_lines.append(f'      backgroundColor: "{bg_color}"')
    config_lines.append(f'      showControls: {"true" if show_controls else "false"}')
    config_lines.append(f'      premultipliedAlpha: false')

    config_lines.append(f'      rawDataURIs: {{\n{raw_data_uris_js}\n      }}')

    # Error/success callbacks
    config_lines.append("""      success: function(player) {
        document.getElementById('status').textContent = 'Loaded successfully';
        document.getElementById('status').style.color = '#4ade80';
        // Log available animations
        var anims = player.skeleton.data.animations.map(function(a) { return a.name; });
        console.log('Available animations:', anims);
        var skins = player.skeleton.data.skins.map(function(s) { return s.name; });
        console.log('Available skins:', skins);
      }""")
    config_lines.append("""      error: function(player, reason) {
        document.getElementById('status').textContent = 'Error: ' + reason;
        document.getElementById('status').style.color = '#ef4444';
        console.error('Spine Player error:', reason);
      }""")

    config_js = ",\n".join(config_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>

<!-- Official Spine Web Player -->
<script src="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/iife/spine-player.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/spine-player.css">

<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 20px;
  }}
  h1 {{
    font-size: 1.5em;
    margin-bottom: 10px;
    color: #a8b2d1;
    letter-spacing: 0.04em;
  }}
  #status {{
    font-size: 0.85em;
    margin-bottom: 15px;
    color: #6b7da0;
    transition: color 0.3s;
  }}
  #player-container {{
    width: 700px;
    height: 600px;
    max-width: 95vw;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }}
  .info {{
    margin-top: 15px;
    font-size: 0.8em;
    color: #4a5568;
    text-align: center;
    max-width: 600px;
    line-height: 1.5;
  }}
  .info a {{ color: #6b8aad; text-decoration: none; }}
  .info a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div id="status">Loading Spine Player...</div>
<div id="player-container"></div>

<div class="info">
  Rendered with the official
  <a href="https://en.esotericsoftware.com/spine-player" target="_blank">Spine Web Player</a>.
  Use the controls to switch animations, adjust speed, and toggle debug views.
</div>

<script>
  new spine.SpinePlayer("player-container", {{
{config_js}
  }});
</script>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML preview using the official Spine Web Player"
    )
    parser.add_argument("--skeleton", required=True, help="Spine JSON or binary (.skel) file")
    parser.add_argument("--atlas", required=True, help="Spine .atlas file")
    parser.add_argument("--atlas-image", default=None,
                        help="Atlas PNG image (auto-detected from atlas if omitted)")
    parser.add_argument("--output", default="preview.html", help="Output HTML file")
    parser.add_argument("--animation", default=None, help="Default animation to play")
    parser.add_argument("--skin", default=None, help="Default skin")
    parser.add_argument("--background", default="#1a1a2eff", help="Background color (hex RGBA)")
    parser.add_argument("--title", default="Spine Animation Preview", help="Page title")
    parser.add_argument("--no-controls", action="store_true", help="Hide player controls")
    args = parser.parse_args()

    # Verify files exist
    for path, name in [(args.skeleton, "Skeleton"), (args.atlas, "Atlas")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    # Find atlas images
    atlas_images = []
    if args.atlas_image:
        img_name = os.path.basename(args.atlas_image)
        atlas_images.append((img_name, args.atlas_image))
    else:
        atlas_images = find_atlas_images(args.atlas)

    if not atlas_images:
        print("ERROR: No atlas images found. Specify --atlas-image or ensure atlas references valid PNGs.")
        sys.exit(1)

    print(f"Skeleton: {args.skeleton}")
    print(f"Atlas: {args.atlas}")
    for img_name, img_path in atlas_images:
        print(f"Atlas image: {img_name} ({os.path.getsize(img_path) / 1024:.1f} KB)")

    # Generate HTML
    print("Generating HTML with embedded Spine Web Player...")
    html = generate_html(
        skeleton_path=args.skeleton,
        atlas_path=args.atlas,
        atlas_images=atlas_images,
        animation=args.animation,
        skin=args.skin,
        bg_color=args.background,
        show_controls=not args.no_controls,
        title=args.title,
    )

    with open(args.output, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nPreview saved: {args.output} ({size_kb:.1f} KB)")
    print("Open in a browser to see your animation (requires internet for the Spine Player CDN).")


if __name__ == "__main__":
    main()
```

</details>

After writing the scripts, verify:
```bash
ls /home/claude/spine-scripts/
# Should show: position_parts.py  build_spine_json.py  make_atlas.py  generate_spine_player.py
```

---

## What You Need From the User

At minimum, one of these asset sets:

| Asset Set | What to Expect |
|-----------|---------------|
| **Separated body part PNGs** | Individual transparent PNGs for head, torso, arms, legs, etc. |
| **Separated PNGs + reference image** | Parts + an assembled character image → enables auto-positioning |
| **Texture atlas + atlas PNG** | A `.atlas` file + spritesheet `.png` (standard Spine export) |
| **Full character image** | A single image — Claude will help define part regions |
| **Existing Spine JSON** | An existing `.json` to add/modify animations |

The user should also say what animations they want (idle, walk, run, attack, wave, jump, etc.)

## Full Pipeline

```
User Assets → Analyze Parts → [Auto-Position if reference] → Build Skeleton → Animate → Preview
```

### Step 1: Analyze the Assets

Look at the uploaded files:

1. **If separated PNGs**: Use Claude's vision to identify each part (head, torso, left-arm, etc.)
   and note their dimensions. Determine the bone hierarchy from the part names and visual layout.

2. **If atlas + spritesheet**: Parse the `.atlas` file to extract region names, positions, and sizes.

3. **If full image only**: Use Claude's vision to identify body parts, then crop into separate PNGs.

4. **If existing Spine JSON**: Parse it, understand the skeleton, and add new animations.

5. **If separated PNGs + assembled reference image**: Run `position_parts.py` for auto-layout.

### Step 1.5: Auto-Position Parts (when reference image is available)

If the user provides both separated body-part PNGs **and** an assembled reference image:

```bash
python3 /home/claude/spine-scripts/position_parts.py \
  --reference assembled_character.png \
  --parts parts_folder/ \
  --output layout.json \
  --debug debug/ \
  --min-matches 4 \
  --ratio 0.80
```

**Algorithm — SIFT + RANSAC similarity transform:**

1. Extracts SIFT keypoints from each part (alpha-masked) and the reference image
2. Matches via FLANN with Lowe's ratio test (default 0.80)
3. Estimates similarity transform (4 DOF: translate + scale + rotation) via
   `cv2.estimateAffinePartial2D` — more robust than full homography for game art
4. SIFT tuning for stylized art: `contrastThreshold=0.02, edgeThreshold=20`
5. Template matching fallback for tiny/featureless parts, using SIFT-derived median scale
6. Z-order via pairwise occlusion voting

After running, **check `debug/comparison.png`** to verify positioning accuracy.
Per-part SIFT match visualizations: `debug/sift_<partname>.jpg`.

**Limitations:** Heavily occluded parts (e.g., thighs behind a belt) may need manual correction.
Compare composite vs reference with Claude's vision and adjust layout JSON offsets.

### Step 2: Build Bone Hierarchy

Standard humanoid skeleton:

```
root
└── hip
    ├── torso
    │   ├── neck → head → hat/hair
    │   ├── left-shoulder → left-upper-arm → left-lower-arm → left-hand
    │   └── right-shoulder → right-upper-arm → right-lower-arm → right-hand
    ├── left-upper-leg → left-lower-leg → left-foot
    └── right-upper-leg → right-lower-leg → right-foot
```

**Coordinate system:** Spine uses Y-up, origin at character's feet center.
- `spine_x = pixel_x - center_x`
- `spine_y = bottom_y - pixel_y`

**Bone positions** are RELATIVE to parent:
- `relative_pos = child_world_pos - parent_world_pos`

**Attachment offsets** are relative to their bone:
- `att_offset = image_center_world_pos - bone_world_pos`

### Step 3: Build Spine JSON

Spine JSON v4.2 structure:

```json
{
  "skeleton": { "hash": "...", "spine": "4.2", "width": 500, "height": 950 },
  "bones": [
    { "name": "root", "x": 0, "y": 0, "length": 0 },
    { "name": "hip", "parent": "root", "x": 0, "y": 410, "length": 30 }
  ],
  "slots": [
    { "name": "back-arm", "bone": "left-arm-bone", "attachment": "back-arm" }
  ],
  "skins": [{
    "name": "default",
    "attachments": {
      "slot-name": {
        "attachment-name": { "x": 5, "y": -10, "width": 100, "height": 200 }
      }
    }
  }],
  "animations": {
    "idle": {
      "bones": {
        "hip": {
          "translate": [
            { "time": 0, "x": 0, "y": 0, "curve": [0.25, 0, 0.75, 1] },
            { "time": 1.0, "x": 0, "y": 3 },
            { "time": 2.0, "x": 0, "y": 0 }
          ]
        }
      }
    }
  }
}
```

**Slots** define draw order — first slot is drawn first (back), last is front.

### Step 4: Create Animations

**Keyframe format:**
```json
{ "time": 0.0, "value": 0, "curve": [0.25, 0, 0.75, 1] }
```

The `curve` is a cubic bezier `[cx1, cy1, cx2, cy2]`. Use `[0.25, 0, 0.75, 1]` for ease-in-out.

**Animation presets:**

| Preset | Duration | Key Technique |
|--------|----------|---------------|
| idle | 2.0s loop | Hip ±3px translate, torso ±1° rotate, head ±1.5° counter-sway |
| walk | 0.8s loop | Opposing arm-leg swing, hip ±5px bob, torso ±3° lean |
| run | 0.5s loop | Exaggerated walk + 5° forward lean + ±8px bounce |
| wave | 1.2s | Shoulder -45°, forearm oscillate ±15° |
| jump | 1.0s | Squat → launch → air → land (4 phases) |
| attack | 0.6s | Windup → strike → follow-through (3 phases) |

**Key principles:**
- Offset timing between related bones (head peaks 0.1s after torso = follow-through)
- Larger movements on larger bones (hip > torso > head)
- All loops must return to starting values

### Step 5: Build Atlas

```bash
python3 /home/claude/spine-scripts/make_atlas.py \
  --parts parts_folder/ \
  --output . \
  --name character_name
```

Outputs: `character_name.png` (spritesheet) + `character_name.atlas` (metadata).

### Step 6: Generate Preview

For a **self-contained HTML Canvas preview** (recommended):
Build the HTML directly in Python with base64-embedded images, bone system, bezier interpolation,
and animation loop. No external dependencies needed.

For an **official Spine Web Player preview**:
```bash
python3 /home/claude/spine-scripts/generate_spine_player.py \
  --skeleton character.json \
  --atlas character.atlas \
  --atlas-image character.png \
  --output preview.html
```

### Step 7: Interactive Editor (Optional)

Build an HTML editor that allows the user to fine-tune part positions:
- Click to select parts (purple dashed border + glow)
- Drag to reposition in real-time
- Arrow keys for 1px nudge (Shift for 10px)
- Side panel with numeric X/Y/rotation inputs
- Draggable z-order list
- Export button producing layout corrections JSON:

```json
{
  "adjustments": {
    "part-name": {
      "original_offset": { "x": 0, "y": 0 },
      "user_offset": { "dx": 5.2, "dy": -3.1, "drot": 0 },
      "final_offset": { "x": 5.2, "y": -3.1 }
    }
  },
  "draw_order": ["back-part", "...", "front-part"]
}
```

This JSON can be fed back to Claude to apply corrections to the skeleton.
