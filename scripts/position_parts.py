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
