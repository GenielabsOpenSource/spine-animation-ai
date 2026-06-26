"""Rebake step — common to both reskin pipelines.

Reads `.genie/skins/{skin}/layout_map.json` to figure out where to take
per-region bboxes from:

  - mode "atlas":    bboxes from the original `.atlas` file's region rects;
                     segment `reskinned_atlas.png`.
  - mode "exploded": bboxes from the layout's `placements`; segment
                     `reskinned_composite.png` (the composite IS the source).

The downstream flow (SAM mask → crop → save extracted/raw/masks → repack →
write `Spine-{skin}.json`) is identical for both.
"""
from __future__ import annotations

import json
import os as _os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageChops

from ..ai.cc_segment import ComponentSegmenter
from ..settings import load_settings
from ..spine import atlas_reader
from ..spine.atlas_repack import repack_atlas
from ..spine.skin_writer import add_skin

if TYPE_CHECKING:
    from .. import projects
    from ..ai.sam_provider import SAMProvider


def _crop_region_rotated(sheet: Image.Image, x: int, y: int, w: int, h: int, rotate: int) -> Image.Image:
    crop = sheet.crop((x, y, x + w, y + h))
    if rotate == 90:
        return crop.rotate(-90, expand=True)
    if rotate == 180:
        return crop.rotate(180, expand=True)
    if rotate == 270:
        return crop.rotate(90, expand=True)
    return crop


def _region_for_slot(spine_json: dict, slot_name: str) -> str | None:
    """Return the atlas region a default-skin slot maps to."""
    skins = spine_json.get("skins")
    atts = None
    if isinstance(skins, list):
        for s in skins:
            if s.get("name") == "default":
                atts = s.get("attachments", {})
                break
    elif isinstance(skins, dict):
        atts = skins.get("default", {})
    if not atts:
        return None
    slot_atts = atts.get(slot_name)
    if not slot_atts:
        return None
    att_key, att_meta = next(iter(slot_atts.items()))
    if isinstance(att_meta, dict) and att_meta.get("name"):
        return att_meta["name"]
    return att_key


def _bboxes_atlas(project, skin_dir: Path) -> tuple[Path, dict[str, dict], dict[str, int]]:
    """Atlas mode: bboxes come from the original .atlas file. Source image
    is the reskinned atlas half. Returns (source_image_path, bboxes, rotates).
    """
    pair = atlas_reader.find_atlas_pair(project.path, project.spine_json_path.stem)
    if pair is None:
        raise FileNotFoundError(
            f"no atlas+sheet pair in {project.path} — cannot rebake"
        )
    atlas_path, _ = pair
    page = atlas_reader.parse_atlas(atlas_path)
    source = skin_dir / "reskinned_atlas.png"
    if not source.exists():
        raise FileNotFoundError(
            f"reskinned_atlas.png missing in {skin_dir} — atlas mode requires it"
        )
    bboxes = {r.name: {"x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in page.regions}
    rotates = {r.name: r.rotate for r in page.regions}
    # Match source dims to the original atlas so the rects apply 1:1.
    img = Image.open(source).convert("RGBA")
    if img.size != (page.sheet_w, page.sheet_h):
        img = img.resize((page.sheet_w, page.sheet_h), Image.LANCZOS)
        img.save(source)
    return source, bboxes, rotates


def _build_original_silhouettes(
    project: "projects.Project",
    mode: str,
    bboxes: dict[str, dict],
    sheet_w: int,
    sheet_h: int,
    layout: dict,
):
    """Build full-atlas-sized boolean silhouettes for each region/slot.

    The bg-components segmenter IoU-matches against these to pick which
    connected component on the reskinned atlas belongs to which slot.

    - "atlas" mode: original atlas sheet's alpha at the region's bbox.
    - "exploded" mode: per-part PNG's alpha placed at the layout placement.
    """
    import numpy as _np

    out: dict[str, _np.ndarray] = {}

    if mode == "atlas":
        pair = atlas_reader.find_atlas_pair(project.path, project.spine_json_path.stem)
        if pair is None:
            return out
        _, sheet_path = pair
        original = _np.asarray(Image.open(sheet_path).convert("RGBA"))[..., 3] > 16
        oh, ow = original.shape
        if (ow, oh) != (sheet_w, sheet_h):
            # Resize the boolean alpha by treating as uint8 then thresholding.
            from PIL import Image as _PI
            resized = _PI.fromarray(original.astype(_np.uint8) * 255).resize(
                (sheet_w, sheet_h), _PI.NEAREST
            )
            original = _np.asarray(resized) > 128
        for name, bb in bboxes.items():
            x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["w"]), int(bb["h"])
            full = _np.zeros((sheet_h, sheet_w), dtype=bool)
            x2, y2 = min(x + w, sheet_w), min(y + h, sheet_h)
            if x2 > x and y2 > y:
                full[y:y2, x:x2] = original[y:y2, x:x2]
            out[name] = full
        return out

    # exploded mode
    placements = layout.get("placements") or {}
    for name, bb in bboxes.items():
        place = placements.get(name) or bb
        x, y = int(place["x"]), int(place["y"])
        w, h = int(place["w"]), int(place["h"])
        part_path = project.path / f"{name}.png"
        if not part_path.exists():
            out[name] = _np.zeros((sheet_h, sheet_w), dtype=bool)
            continue
        part = _np.asarray(Image.open(part_path).convert("RGBA"))[..., 3] > 16
        # Resize part alpha to fit the placement size.
        if part.shape != (h, w):
            from PIL import Image as _PI
            resized = _PI.fromarray(part.astype(_np.uint8) * 255).resize(
                (w, h), _PI.NEAREST
            )
            part = _np.asarray(resized) > 128
        full = _np.zeros((sheet_h, sheet_w), dtype=bool)
        x2, y2 = min(x + w, sheet_w), min(y + h, sheet_h)
        if x2 > x and y2 > y:
            full[y:y2, x:x2] = part[: y2 - y, : x2 - x]
        out[name] = full
    return out


def _bboxes_exploded(skin_dir: Path, layout: dict) -> tuple[Path, dict[str, dict], dict[str, int]]:
    """Exploded mode: bboxes come from the layout's placements. Source image
    is the reskinned composite (no split was done).
    """
    source = skin_dir / "reskinned_composite.png"
    if not source.exists():
        raise FileNotFoundError(
            f"reskinned_composite.png missing in {skin_dir} — exploded mode requires it"
        )
    placements = layout.get("placements") or {}
    if not placements:
        raise ValueError("exploded layout_map has no placements")
    bboxes = {
        name: {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]}
        for name, p in placements.items()
    }
    rotates = {name: 0 for name in placements}  # exploded layout has no rotation
    return source, bboxes, rotates


def rebake_skin(
    project: "projects.Project",
    skin_name: str,
    *,
    sam_provider: "SAMProvider",
    pipeline_logger=None,
) -> dict:
    """Slice the reskinned image into per-region PNGs (SAM-masked when
    available), repack into a per-skin atlas, and write the per-skin Spine
    JSON. Mode is read from `layout_map.json`.

    Response shape is identical for both modes — frontend contract unchanged.
    """
    skin_dir = project.workdir / "skins" / skin_name
    layout_path = skin_dir / "layout_map.json"
    if not layout_path.exists():
        raise FileNotFoundError(
            f"layout_map.json missing in {skin_dir} — run /api/reskin/generate first"
        )
    layout = json.loads(layout_path.read_text())
    mode = layout.get("mode", "atlas")

    if mode == "exploded":
        source_path, bboxes, rotates = _bboxes_exploded(skin_dir, layout)
    else:
        source_path, bboxes, rotates = _bboxes_atlas(project, skin_dir)

    sheet = Image.open(source_path).convert("RGBA")
    sheet_w, sheet_h = sheet.size

    extracted_dir = skin_dir / "extracted"
    extracted_raw_dir = skin_dir / "extracted_raw"
    masks_dir = skin_dir / "masks"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    seg_method = settings.segmentation.method
    sam_masks: dict[str, object] = {}
    sam_ok = False
    print(
        f"[rebake] mode={mode}, seg_method={seg_method}, regions={len(bboxes)}",
        flush=True,
    )
    import time as _time
    seg_t0 = _time.time()
    seg_input_snap = None
    if pipeline_logger is not None:
        seg_input_snap = pipeline_logger.snapshot(source_path, f"rebake_{skin_name}_source")
    if seg_method == "bg_components":
        try:
            silhouettes = _build_original_silhouettes(
                project, mode, bboxes, sheet_w, sheet_h, layout
            )
            cc = ComponentSegmenter()
            sam_masks = cc.segment_with_bboxes(
                source_path, bboxes, original_silhouettes=silhouettes
            )
            sam_ok = True
            print(f"[rebake] cc ok, masks={len(sam_masks)}", flush=True)
            if pipeline_logger is not None:
                pipeline_logger.record(
                    "segmentation_bg_components",
                    skin_name=skin_name,
                    params={
                        "regions": len(bboxes),
                        "matched": len(sam_masks),
                        "min_iou_env": _os.environ.get("CC_SEGMENT_MIN_IOU"),
                    },
                    input_paths=[seg_input_snap],
                    duration_ms=(_time.time() - seg_t0) * 1000,
                )
        except Exception as e:
            import traceback
            print(f"[rebake] cc failed: {e}", flush=True)
            traceback.print_exc()
            if pipeline_logger is not None:
                pipeline_logger.record(
                    "segmentation_bg_components",
                    skin_name=skin_name,
                    params={"regions": len(bboxes)},
                    input_paths=[seg_input_snap],
                    duration_ms=(_time.time() - seg_t0) * 1000,
                    status="error",
                    error=str(e),
                )
    elif sam_provider.available:
        try:
            sam_masks = sam_provider.segment_with_bboxes(source_path, bboxes)
            sam_ok = True
            print(f"[rebake] SAM ok, masks={len(sam_masks)}", flush=True)
            if pipeline_logger is not None:
                pipeline_logger.record(
                    "segmentation_sam",
                    skin_name=skin_name,
                    params={
                        "regions": len(bboxes),
                        "masked": len(sam_masks),
                        "sam_url": sam_provider.base_url,
                    },
                    input_paths=[seg_input_snap],
                    duration_ms=(_time.time() - seg_t0) * 1000,
                )
        except Exception as e:
            import traceback
            print(f"[rebake] SAM failed: {e}", flush=True)
            traceback.print_exc()
            if pipeline_logger is not None:
                pipeline_logger.record(
                    "segmentation_sam",
                    skin_name=skin_name,
                    params={"regions": len(bboxes)},
                    input_paths=[seg_input_snap],
                    duration_ms=(_time.time() - seg_t0) * 1000,
                    status="error",
                    error=str(e),
                )

    saved_regions: list[str] = []
    for name, bb in bboxes.items():
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        rot = rotates.get(name, 0)
        raw = _crop_region_rotated(sheet, x, y, w, h, rot)
        raw.save(extracted_raw_dir / f"{name}.png")
        masked = raw.copy()
        applied_mask: Image.Image | None = None
        erode_px_used = 0
        score = None

        if sam_ok and name in sam_masks:
            try:
                mask_full = sam_masks[name].mask
                score = getattr(sam_masks[name], "score", None)
                if mask_full.size != (sheet_w, sheet_h):
                    mask_full = mask_full.resize((sheet_w, sheet_h), Image.NEAREST)
                mask_crop = _crop_region_rotated(mask_full, x, y, w, h, rot)
                try:
                    import cv2 as _cv2
                    import numpy as _np
                    m_arr = _np.asarray(mask_crop, dtype=_np.uint8)
                    # Close fills small holes / pinholes inside the SAM mask.
                    k = max(3, min(15, min(mask_crop.size) // 30) | 1)
                    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
                    m_arr = _cv2.morphologyEx(m_arr, _cv2.MORPH_CLOSE, kernel)
                    # Erode N px to trim the soft white halo Gemini paints at
                    # part edges. Radius scales with part size so thin features
                    # (eyelashes, outlines) survive. Tunable / disable-able via
                    # SettingsModal — see app/backend/settings.py.
                    side = max(1, min(mask_crop.size))
                    erode_px_used = load_settings().erosion.px_for_side(side)
                    if erode_px_used > 0:
                        ek = 2 * erode_px_used + 1
                        ekernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (ek, ek))
                        m_arr = _cv2.erode(m_arr, ekernel, iterations=1)
                    mask_crop = Image.fromarray(m_arr, mode="L")
                except Exception:
                    pass
                mask_crop.save(masks_dir / f"{name}.png")
                applied_mask = mask_crop
                base_alpha = masked.split()[3]
                new_alpha = ImageChops.multiply(base_alpha, mask_crop)
                masked.putalpha(new_alpha)
            except Exception as e:
                print(f"[rebake] mask apply failed for {name}: {e}", flush=True)

        masked.save(extracted_dir / f"{name}.png")

        if pipeline_logger is not None:
            inputs = [pipeline_logger.snapshot(raw, f"region_{name}_raw")]
            if applied_mask is not None:
                inputs.append(pipeline_logger.snapshot(applied_mask, f"region_{name}_mask"))
            pipeline_logger.record(
                "region_mask_apply",
                skin_name=skin_name,
                slot=name,
                params={
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "rotate": rot,
                    "method": seg_method,
                    "score": score,
                    "erode_px": erode_px_used,
                    "had_mask": applied_mask is not None,
                },
                input_paths=inputs,
                output_paths=[pipeline_logger.snapshot(masked, f"region_{name}_out")],
            )
        saved_regions.append(name)

    # Defensive: copy any project-root region PNGs the rebake didn't already
    # produce (so the per-skin atlas covers every region the skeleton uses).
    for original in project.path.glob("*.png"):
        if original.with_suffix(".atlas").exists():
            continue
        if original.stem.startswith("Spine-"):
            continue
        target = extracted_dir / original.name
        if not target.exists():
            shutil.copy2(original, target)

    atlas_name = f"{project.spine_json_path.stem}-{skin_name}"
    pack_t0 = _time.time()
    pack_result = repack_atlas(extracted_dir, project.path, atlas_name)
    name_map = pack_result.get("name_map", {})
    if pipeline_logger is not None:
        pipeline_logger.record(
            "atlas_pack",
            skin_name=skin_name,
            params={
                "regions_packed": len(pack_result.get("placements", {})),
                "atlas_name": atlas_name,
                "method_used": seg_method if sam_ok else "none",
            },
            input_paths=[seg_input_snap],
            output_paths=[pipeline_logger.snapshot(pack_result["image"], f"atlas_{atlas_name}")],
            duration_ms=(_time.time() - pack_t0) * 1000,
        )

    sam_region_set = set(saved_regions)
    skin_placements: dict[str, dict] = {}
    region_names: dict[str, str] = {}
    sam_slots: list[str] = []
    for slot in project.slots:
        region = _region_for_slot(project.spine_json, slot.name)
        if not region or region not in sam_region_set:
            continue
        atlas_region = name_map.get(region, region)
        placement = pack_result["placements"].get(atlas_region)
        if placement is None:
            continue
        skin_placements[slot.name] = placement
        region_names[slot.name] = atlas_region
        sam_slots.append(slot.name)

    new_spine = add_skin(
        project.spine_json,
        skin_name,
        placements=skin_placements,
        attachment_names=region_names,
    )
    skin_json_path = project.path / f"{project.spine_json_path.stem}-{skin_name}.json"
    skin_json_path.write_text(json.dumps(new_spine, indent=2))

    (skin_dir / "sam_slots.json").write_text(json.dumps(sorted(sam_slots), indent=2))

    return {
        "saved": saved_regions,
        "atlas": pack_result["atlas"].name,
        "atlas_image": pack_result["image"].name,
        "skin_spine_json": skin_json_path.name,
        "sam_used": sam_ok,
        "masks_count": len(sam_masks),
        "mode": mode,
    }


# Back-compat alias for the previous name.
rebake_from_reskinned_atlas = rebake_skin
