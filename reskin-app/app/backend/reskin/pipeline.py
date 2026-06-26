"""Reskin pipeline orchestrator.

Two methods supported:

  - "atlas" (default): snapshot + original atlas sheet → side-by-side
    composite → Gemini → split into reskinned snapshot + reskinned atlas.
    See atlas_compose.build_atlas_composite.

  - "exploded": every per-region PNG bin-packed with a small gap →
    Gemini → no split (the composite IS the segmentation source).
    See exploded_compose.build_exploded_composite.

In both cases segmentation is deferred to atlas_rebake.rebake_skin, which
reads `layout_map.json` to figure out where the per-region bboxes come from.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .atlas_compose import (
    ATLAS_RESKIN_NEGATIVE,
    ATLAS_RESKIN_PROMPT,
    build_atlas_composite,
)
from .exploded_compose import (
    EXPLODED_RESKIN_NEGATIVE,
    EXPLODED_RESKIN_PROMPT,
    build_exploded_composite,
)


VALID_METHODS = {"atlas", "exploded"}


async def full_reskin(
    project_dir: Path,
    workdir: Path,
    skin_name: str,
    user_prompt: str,
    *,
    image_provider,
    method: str = "atlas",
    atlas_path: Path | None = None,
    atlas_sheet_path: Path | None = None,
    reference_image_path: Path | None = None,
    reference_prompt: str = "",
    pipeline_logger=None,
) -> dict:
    """Run the chosen reskin pipeline. Returns layout + paths the frontend
    uses to preview the result.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(VALID_METHODS)}")

    workdir = Path(workdir) / skin_name
    workdir.mkdir(parents=True, exist_ok=True)
    project_dir = Path(project_dir)

    import time as _time
    if method == "atlas":
        snapshot_path = project_dir / ".genie" / "snapshots" / f"{skin_name}.png"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"no snapshot at {snapshot_path} — frontend must POST a canvas snapshot first"
            )
        if atlas_sheet_path is None or atlas_path is None:
            raise ValueError("atlas method requires atlas_path and atlas_sheet_path")
        build_t0 = _time.time()
        layout = build_atlas_composite(
            snapshot_path,
            atlas_sheet_path,
            workdir,
            atlas_meta_name=Path(atlas_path).name,
        )
        prompt = ATLAS_RESKIN_PROMPT.format(user_prompt=user_prompt)
        negative = ATLAS_RESKIN_NEGATIVE
        if pipeline_logger is not None:
            pipeline_logger.record(
                "composite_build",
                skin_name=skin_name,
                params={
                    "mode": "atlas",
                    "snapshot_size": [layout["snapshot_rect"]["w"], layout["snapshot_rect"]["h"]],
                    "atlas_size": [layout["atlas_rect"]["w"], layout["atlas_rect"]["h"]],
                    "composite_size": [layout["composite_w"], layout["composite_h"]],
                },
                input_paths=[
                    pipeline_logger.snapshot(snapshot_path, f"composite_{skin_name}_snapshot"),
                    pipeline_logger.snapshot(atlas_sheet_path, f"composite_{skin_name}_atlas"),
                ],
                output_paths=[
                    pipeline_logger.snapshot(workdir / "composite.png", f"composite_{skin_name}"),
                ],
                duration_ms=(_time.time() - build_t0) * 1000,
            )
    else:  # "exploded"
        build_t0 = _time.time()
        layout = build_exploded_composite(project_dir, workdir, padding=10)
        prompt = EXPLODED_RESKIN_PROMPT.format(user_prompt=user_prompt)
        negative = EXPLODED_RESKIN_NEGATIVE
        if pipeline_logger is not None:
            pipeline_logger.record(
                "composite_build",
                skin_name=skin_name,
                params={
                    "mode": "exploded",
                    "composite_size": [layout["composite_w"], layout["composite_h"]],
                    "padding_px": 10,
                    "part_count": len(layout.get("placements") or {}),
                },
                output_paths=[
                    pipeline_logger.snapshot(workdir / "composite.png", f"composite_{skin_name}"),
                ],
                duration_ms=(_time.time() - build_t0) * 1000,
            )

    composite_path = workdir / "composite.png"
    reskinned_composite_path = workdir / "reskinned_composite.png"

    refs: list[Path] = []
    if reference_image_path is not None and Path(reference_image_path).is_file():
        refs.append(Path(reference_image_path))
        if reference_prompt:
            prompt = (
                f"{prompt}\n\n"
                "Additional style guidance — use the FIRST input image (a "
                "reference style sheet) for color, rendering style, line "
                "weight, and overall mood. Do NOT redraw or include any of "
                "its characters/elements in the output. Specifically:\n"
                f"{reference_prompt}"
            )
        else:
            prompt = (
                f"{prompt}\n\n"
                "The FIRST input image is a style reference. Match its "
                "color palette, rendering style, line weight, and mood, but "
                "do not include any of its characters/elements in the output."
            )

    # Snapshot only what the MODEL actually sees: optional reference + padded
    # composite. The unpadded composite is logged separately as a build step
    # above, so it shouldn't appear here too (otherwise the atlas thumbnail
    # looks like it's being passed twice).
    pre_ref = None
    if pipeline_logger is not None and refs:
        pre_ref = pipeline_logger.snapshot(refs[0], f"reskin_{skin_name}_reference")

    t0 = _time.time()
    gemini_meta: dict = {}
    try:
        await image_provider.edit_image(
            composite_path,
            prompt,
            negative_prompt=negative,
            out_path=reskinned_composite_path,
            reference_images=refs,
            metadata=gemini_meta,
        )
    except Exception as e:
        if pipeline_logger is not None:
            padded_pil = gemini_meta.pop("padded_image_pil", None)
            padded_snap = (
                pipeline_logger.snapshot(padded_pil, f"reskin_{skin_name}_padded")
                if padded_pil is not None else None
            )
            pipeline_logger.record(
                "gemini_full_reskin",
                skin_name=skin_name,
                params={
                    "method": method,
                    "user_prompt": user_prompt,
                    "reference_used": bool(refs),
                    "reference_prompt": reference_prompt,
                    "negative_prompt": negative,
                    **gemini_meta,
                },
                input_paths=[pre_ref, padded_snap],
                duration_ms=(_time.time() - t0) * 1000,
                status="error",
                error=str(e),
            )
        raise

    if pipeline_logger is not None:
        padded_pil = gemini_meta.pop("padded_image_pil", None)
        padded_snap = (
            pipeline_logger.snapshot(padded_pil, f"reskin_{skin_name}_padded")
            if padded_pil is not None else None
        )
        post = pipeline_logger.snapshot(reskinned_composite_path, f"reskin_{skin_name}_output")
        pipeline_logger.record(
            "gemini_full_reskin",
            skin_name=skin_name,
            params={
                "method": method,
                "user_prompt": user_prompt,
                "reference_used": bool(refs),
                "reference_prompt": reference_prompt,
                **gemini_meta,
            },
            input_paths=[pre_ref, padded_snap],
            output_paths=[post],
            duration_ms=(_time.time() - t0) * 1000,
        )

    composite = Image.open(reskinned_composite_path).convert("RGBA")
    cw, ch = composite.size
    expected = (layout["composite_w"], layout["composite_h"])
    if (cw, ch) != expected:
        composite = composite.resize(expected, Image.LANCZOS)
        composite.save(reskinned_composite_path)
        cw, ch = expected

    response: dict = {
        "skin_name": skin_name,
        "method": method,
        "composite": str(composite_path.relative_to(project_dir)),
        "reskinned_composite": str(reskinned_composite_path.relative_to(project_dir)),
        "layout": layout,
    }

    if method == "atlas":
        sr = layout["snapshot_rect"]
        ar = layout["atlas_rect"]
        snap_half = composite.crop((sr["x"], sr["y"], sr["x"] + sr["w"], sr["y"] + sr["h"]))
        atlas_half = composite.crop((ar["x"], ar["y"], ar["x"] + ar["w"], ar["y"] + ar["h"]))
        snap_half_path = workdir / "reskinned_snapshot.png"
        atlas_half_path = workdir / "reskinned_atlas.png"
        snap_half.save(snap_half_path)
        atlas_half.save(atlas_half_path)
        response["reskinned_snapshot"] = str(snap_half_path.relative_to(project_dir))
        response["reskinned_atlas"] = str(atlas_half_path.relative_to(project_dir))

    return response
