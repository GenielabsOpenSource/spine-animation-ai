"""FastAPI app for Genie Reskin.

Single-project mental model: at any time the server has 0 or 1 open project.
All routes operate against that project, identified by its absolute path.
"""
from __future__ import annotations

import asyncio
import io
import json
import shutil
from pathlib import Path
from typing import Any, Optional

# Load .env early.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from PIL import Image

from . import projects, secrets_store, settings as settings_mod
from .ai import bria
from .ai.gemini import GeminiProvider
from .ai.sam_provider import SAMProvider
from .imaging import SlotEdit, apply_edit
from .logs import PipelineLogger
from .reskin.atlas_rebake import rebake_skin
from .reskin.pipeline import full_reskin
from .spine import atlas_reader
from .spine.atlas_repack import repack_atlas
from .spine.skin_writer import add_skin

# Saved API keys (~/.genie-reskin/secrets.json) take precedence over the .env
# that load_dotenv loaded above — push them into os.environ before any
# provider reads them.
secrets_store.apply_to_env()


app = FastAPI(title="Genie Reskin API")

# Vite dev server runs on :5173 by default and we run uvicorn on :8765 — open CORS
# for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ───────── In-memory state (single project) ─────────


class _State:
    project: projects.Project | None = None
    image_provider: GeminiProvider | None = None  # lazily constructed

    @classmethod
    def require_project(cls) -> projects.Project:
        if cls.project is None:
            raise HTTPException(400, "no project is open — POST /api/project/open first")
        return cls.project

    @classmethod
    def provider(cls) -> GeminiProvider:
        if cls.image_provider is None:
            cls.image_provider = GeminiProvider()
        return cls.image_provider

    @classmethod
    def logger(cls) -> PipelineLogger:
        p = cls.require_project()
        return PipelineLogger(p.path, p.workdir)


# ───────── Request models ─────────


class OpenProjectPayload(BaseModel):
    path: str


class GeneratePayload(BaseModel):
    skin_name: str
    prompt: str
    method: str = "atlas"  # "atlas" or "exploded"


class RebakePayload(BaseModel):
    skin_name: str


class InpaintSlotPayload(BaseModel):
    skin_name: str
    slot: str
    prompt: str


class RevertSlotsPayload(BaseModel):
    slots: list[str]
    revert: bool = True  # True = swap to original, False = restore reskin


# Mask is a binary PNG uploaded as raw bytes (see PUT /api/skin/{skin}/mask-image/{slot})


class EditPayload(BaseModel):
    slot: str
    skin_name: str  # which skin's parts to apply edits to
    edit: dict      # mirrors SlotEdit


class ExportPayload(BaseModel):
    skin_name: str
    edits: dict[str, dict] = {}  # {slot: SlotEdit-like dict}
    write_into_main_json: bool = False  # if True, modify Spine.json in place; else write Spine_{skin}.json


# ───────── Project ─────────


@app.post("/api/project/open")
def open_project(payload: OpenProjectPayload):
    try:
        p = projects.open_project(payload.path)
    except projects.MultipleProjectsError as e:
        # Folder has >1 Spine project triplet — frontend shows a picker.
        raise HTTPException(409, detail={
            "multi_project": True,
            "folder": str(e.folder),
            "candidates": [
                {"base": c.base, "display_name": c.display_name, "path": str(c.path)}
                for c in e.candidates
            ],
        })
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    _State.project = p
    return p.to_payload()


@app.get("/api/project/status")
def project_status():
    if _State.project is None:
        return {"open": False}
    return {"open": True, **_State.project.to_payload()}


@app.post("/api/project/snapshot")
async def project_snapshot(request: Request, skin_name: str = Query(...)):
    """Save a frontend canvas snapshot as the AI input for a specific skin.

    Saved to `.genie/snapshots/{skin_name}.png`. The reskin pipeline reads
    this file as the source image for Gemini.
    """
    project = _State.require_project()
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    snap_dir = project.workdir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out = snap_dir / f"{skin_name}.png"
    out.write_bytes(body)
    return {"ok": True, "size": len(body), "path": str(out.relative_to(project.path))}


@app.get("/api/project/file/{rel:path}")
def project_file(rel: str, v: Optional[str] = Query(None)):
    """Serve any file inside the open project (sandboxed to project.path).

    Path-style URL (vs query param) so PixiJS / browser caches infer the file
    type from the extension.

    For .atlas files served with a ?v= cache-buster, rewrite the page-image
    line(s) to include the same ?v= query. Spine atlases reference the PNG
    by bare filename and the spine runtime resolves it against the atlas
    URL — but URL resolution strips query strings, so without this rewrite
    Pixi/the browser keep serving the pre-rebake PNG after inpaint/mask
    edits even though the atlas itself reloads.
    """
    project = _State.require_project()
    full = (project.path / rel).resolve()
    if not str(full).startswith(str(project.path.resolve())):
        raise HTTPException(403, "path escapes project root")
    if not full.is_file():
        raise HTTPException(404, str(full))
    if v and full.suffix.lower() == ".atlas":
        text = full.read_text()
        out_lines = []
        rewrote = False
        for line in text.split("\n"):
            stripped = line.strip()
            if (not rewrote) and stripped and stripped.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ) and "?" not in stripped:
                out_lines.append(f"{stripped}?v={v}")
                rewrote = True
            else:
                out_lines.append(line)
        return Response(content="\n".join(out_lines), media_type="text/plain")
    return FileResponse(full)


# ───────── Reskin ─────────


@app.post("/api/reskin/rebake")
def reskin_rebake(payload: RebakePayload):
    """Slice the reskinned ATLAS half into per-region PNGs using SAM, repack
    into a per-skin atlas, and write `Spine-{skin}.json`.

    Atlas region rects come straight from the project's `.atlas` file —
    no live-spine bbox computation needed.
    """
    project = _State.require_project()
    try:
        return rebake_skin(
            project,
            payload.skin_name,
            sam_provider=SAMProvider(),
            pipeline_logger=_State.logger(),
        )
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


@app.get("/api/skin/{skin_name}/raw/{slot}")
def get_slot_raw(skin_name: str, slot: str):
    """Return the un-masked (full bbox) reskinned PNG for a slot — what the
    user paints/lassos on top of in the mask editor."""
    project = _State.require_project()
    raw = project.workdir / "skins" / skin_name / "extracted_raw" / f"{slot}.png"
    if not raw.exists():
        # Fall back to the masked extracted version so the editor still has
        # something to display (legacy skins generated before the split).
        legacy = project.workdir / "skins" / skin_name / "extracted" / f"{slot}.png"
        if legacy.exists():
            return FileResponse(legacy)
        raise HTTPException(404, f"raw not found for slot {slot!r}")
    return FileResponse(raw)


@app.get("/api/skin/{skin_name}/mask-image/{slot}")
def get_slot_mask_image(skin_name: str, slot: str):
    project = _State.require_project()
    f = project.workdir / "skins" / skin_name / "masks" / f"{slot}.png"
    if not f.exists():
        raise HTTPException(404, "no mask")
    return FileResponse(f)


@app.put("/api/skin/{skin_name}/mask-image/{slot}")
async def put_slot_mask_image(skin_name: str, slot: str, request: Request):
    """Save a binary mask PNG (8-bit greyscale) for a slot, then re-pack."""
    project = _State.require_project()
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    masks_dir = project.workdir / "skins" / skin_name / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_file = masks_dir / f"{slot}.png"
    mask_file.write_bytes(body)
    logger = _State.logger()
    raw_file = project.workdir / "skins" / skin_name / "extracted_raw" / f"{slot}.png"
    pre = logger.snapshot(raw_file, f"mask_{slot}_raw") if raw_file.exists() else None
    mask_snap = logger.snapshot(mask_file, f"mask_{slot}_mask")
    import time as _time
    t0 = _time.time()
    rebuilt = _rebuild_skin_atlas(project, skin_name)
    repacked_atlas = project.path / rebuilt.get("atlas", "")
    logger.record(
        "mask_save",
        skin_name=skin_name,
        slot=slot,
        params={"mask_bytes": len(body)},
        input_paths=[pre, mask_snap],
        output_paths=[logger.snapshot(repacked_atlas, f"atlas_{skin_name}")],
        duration_ms=(_time.time() - t0) * 1000,
    )
    return rebuilt


@app.post("/api/skin/{skin_name}/embedding/{slot}")
def get_slot_embedding(skin_name: str, slot: str):
    """Proxy to the SAM segmentation server's /image_segmentation route to
    get an image embedding for the slot's raw image. The embedding lives
    locally so the frontend can run mask decoding ONNX inference without
    re-uploading the image each time.

    Returns the embedding payload directly (the same JSON shape the SAM
    server returns), with the embedding bytes in base64.
    """
    import base64
    import numpy as np
    import requests

    project = _State.require_project()
    raw_path = project.workdir / "skins" / skin_name / "extracted_raw" / f"{slot}.png"
    if not raw_path.exists():
        legacy = project.workdir / "skins" / skin_name / "extracted" / f"{slot}.png"
        if legacy.exists():
            raw_path = legacy
        else:
            raise HTTPException(404, "no raw image for slot")

    sam_url = os.environ.get("SAM_SERVER_URL", "").rstrip("/")
    if not sam_url:
        raise HTTPException(503, "SAM_SERVER_URL not configured")

    # The SAM server fetches by URL (per the user's existing code). We need a
    # URL it can reach. Easiest: serve the raw via our backend and hope SAM
    # can reach our host. But our backend is bound to 127.0.0.1, so SAM can't.
    # Workaround: the SAM server's /image_segmentation requires `aws_image_url`,
    # which downloads. We POST a temporary URL that the SAM server can hit.
    #
    # Since SAM is on GCP and we're on localhost, the cleanest approach is to
    # extend the SAM server with a `image_base64` variant. For now, we cache
    # the embedding response so it's only computed once per (skin, slot).
    cache_path = project.workdir / "skins" / skin_name / "embeddings" / f"{slot}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Best-effort: try POSTing the raw PNG bytes. The user's SAM server does
    # accept either `aws_image_url` or `image_base64` (we asked them to add
    # the latter for /segment_with_boxes). If /image_segmentation only takes
    # a URL, this will 400 — caller catches it and the magic-select tool
    # falls back to disabled.
    raw_b64 = base64.b64encode(raw_path.read_bytes()).decode()
    try:
        r = requests.post(
            f"{sam_url}/image_segmentation",
            json={"image_base64": raw_b64},
            timeout=120,
        )
        if r.status_code == 400:
            # Try the legacy url-only shape with a data URL (some servers accept this)
            data_url = f"data:image/png;base64,{raw_b64}"
            r = requests.post(
                f"{sam_url}/image_segmentation",
                json={"aws_image_url": data_url},
                timeout=120,
            )
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"SAM /image_segmentation failed: {e}")

    payload = r.json()
    # The SAM server returns image_embedding as a Python list of floats; we
    # convert to base64 float32 bytes for compact transport to the browser.
    if isinstance(payload.get("image_embedding"), list):
        emb = np.asarray(payload["image_embedding"], dtype=np.float32)
        payload["image_embedding"] = base64.b64encode(emb.tobytes()).decode()
        payload["embedding_shape"] = list(emb.shape)
        payload["embedding_dtype"] = "float32"

    out = {"result": payload}
    cache_path.write_text(json.dumps(out))
    return out


def _transforms_path(project: projects.Project, skin: str) -> Path:
    """Where per-skin slot transform deltas live.

    Deltas (`{x, y}` in spine-local units) are applied on top of the loaded
    JSON's attachment x/y at render time — the source JSON files are never
    mutated, so reset always returns to disk-default.
    """
    return project.workdir / "transforms" / f"{skin}.json"


def _read_transforms(project: projects.Project, skin: str) -> dict[str, dict]:
    p = _transforms_path(project, skin)
    if not p.exists():
        return {}
    try:
        v = json.loads(p.read_text())
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _write_transforms(project: projects.Project, skin: str, transforms: dict) -> None:
    p = _transforms_path(project, skin)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(transforms, indent=2))


@app.get("/api/skin/{skin_name}/transforms")
def get_transforms(skin_name: str):
    project = _State.require_project()
    return _read_transforms(project, skin_name)


@app.put("/api/skin/{skin_name}/transforms")
async def put_transforms(skin_name: str, request: Request):
    """Replace the full transforms dict for a skin."""
    project = _State.require_project()
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "expected JSON object {slot: {x, y}}")
    _write_transforms(project, skin_name, body)
    return {"ok": True, "transforms": body}


@app.delete("/api/skin/{skin_name}/transforms")
def reset_transforms(skin_name: str):
    """Clear all transforms for a skin → reset to JSON default."""
    project = _State.require_project()
    p = _transforms_path(project, skin_name)
    if p.exists():
        p.unlink()
    return {"ok": True}


REVERTED_SLOTS_FILE = "reverted_slots.json"


def _read_reverted_slots(skin_dir: Path) -> set[str]:
    p = skin_dir / REVERTED_SLOTS_FILE
    if not p.exists():
        return set()
    try:
        v = json.loads(p.read_text())
        return set(v) if isinstance(v, list) else set()
    except Exception:
        return set()


def _write_reverted_slots(skin_dir: Path, slots: set[str]) -> None:
    skin_dir.mkdir(parents=True, exist_ok=True)
    (skin_dir / REVERTED_SLOTS_FILE).write_text(json.dumps(sorted(slots), indent=2))


def _slot_to_region(spine_json: dict, slot_name: str) -> str | None:
    """Return the atlas region a default-skin slot maps to. Mirror of
    atlas_rebake._region_for_slot — kept here to avoid the cross-import."""
    skins = spine_json.get("skins")
    atts = None
    if isinstance(skins, list):
        for s in skins:
            if s.get("name") == "default":
                atts = s.get("attachments", {}) or {}
                break
    elif isinstance(skins, dict):
        atts = skins.get("default", {}) or {}
    if not atts:
        return None
    slot_atts = atts.get(slot_name)
    if not slot_atts:
        return None
    att_key, att_meta = next(iter(slot_atts.items()))
    if isinstance(att_meta, dict) and att_meta.get("name"):
        return att_meta["name"]
    return att_key


def _rebuild_skin_atlas(project, skin_name: str) -> dict:
    """Re-pack the skin's extracted/ with current per-slot masks applied,
    then rewrite the per-skin Spine JSON. Used by mask-edit, inpaint, and
    revert-slots.

    Pixel source preference (per region):
      0. reverted_slots.json names this slot's region → project-root original
      1. extracted_raw/{region}.png + masks/{region}.png → multiply alpha
      2. extracted/{region}.png → use as-is (already SAM-masked)
    """
    import numpy as _np
    skin_dir = project.workdir / "skins" / skin_name
    extracted_dir = skin_dir / "extracted"
    raw_dir = skin_dir / "extracted_raw"
    masks_dir = skin_dir / "masks"

    baked = skin_dir / "baked"
    baked.mkdir(parents=True, exist_ok=True)
    for p in baked.glob("*.png"):
        p.unlink()

    reverted_slots_set = _read_reverted_slots(skin_dir)
    reverted_regions: set[str] = set()
    for s in reverted_slots_set:
        r = _slot_to_region(project.spine_json, s)
        if r:
            reverted_regions.add(r)

    seen: set[str] = set()

    for region in reverted_regions:
        original = project.path / f"{region}.png"
        if not original.exists():
            continue
        shutil.copy2(original, baked / f"{region}.png")
        seen.add(region)

    if raw_dir.exists():
        for raw in raw_dir.glob("*.png"):
            slot_stem = raw.stem
            if slot_stem in seen:
                continue
            seen.add(slot_stem)
            mask_path = masks_dir / f"{slot_stem}.png"
            if mask_path.exists():
                im = Image.open(raw).convert("RGBA")
                mask = Image.open(mask_path).convert("L")
                if mask.size != im.size:
                    mask = mask.resize(im.size, Image.NEAREST)
                arr = _np.asarray(im).copy()
                base_alpha = arr[..., 3].astype(_np.float32)
                mask_arr = _np.asarray(mask).astype(_np.float32) / 255.0
                arr[..., 3] = _np.clip(base_alpha * mask_arr, 0, 255).astype(_np.uint8)
                Image.fromarray(arr, mode="RGBA").save(baked / raw.name)
            else:
                shutil.copy2(raw, baked / raw.name)

    if extracted_dir.exists():
        for png in extracted_dir.glob("*.png"):
            if png.stem in seen:
                continue
            shutil.copy2(png, baked / png.name)

    atlas_name = f"{project.spine_json_path.stem}-{skin_name}"
    pack_result = repack_atlas(baked, project.path, atlas_name)
    name_map = pack_result.get("name_map", {})

    sam_slots_path = skin_dir / "sam_slots.json"
    sam_slots: list[str] = []
    if sam_slots_path.exists():
        try:
            sam_slots = json.loads(sam_slots_path.read_text()) or []
        except Exception:
            sam_slots = []

    skin_placements = {}
    region_names = {}
    for slot in sam_slots:
        if slot in reverted_slots_set:
            continue
        atlas_region = name_map.get(slot, slot)
        if atlas_region in pack_result["placements"]:
            skin_placements[slot] = pack_result["placements"][atlas_region]
            region_names[slot] = atlas_region

    new_spine = add_skin(
        project.spine_json,
        skin_name,
        placements=skin_placements,
        attachment_names=region_names,
    )
    skin_json_path = project.path / f"{project.spine_json_path.stem}-{skin_name}.json"
    skin_json_path.write_text(json.dumps(new_spine, indent=2))
    return {
        "ok": True,
        "applied": True,
        "atlas": pack_result["atlas"].name,
        "skin_spine_json": skin_json_path.name,
        "reverted_slots": sorted(reverted_slots_set),
    }


@app.post("/api/skin/{skin_name}/revert-slots")
def revert_slots(skin_name: str, payload: RevertSlotsPayload):
    """Toggle a list of slots between the reskinned version and the
    project-root original. `revert=True` swaps in originals; `revert=False`
    restores the reskinned version.
    """
    project = _State.require_project()
    skin_dir = project.workdir / "skins" / skin_name
    cur = _read_reverted_slots(skin_dir)
    for s in payload.slots:
        if payload.revert:
            cur.add(s)
        else:
            cur.discard(s)
    _write_reverted_slots(skin_dir, cur)
    return _rebuild_skin_atlas(project, skin_name)


@app.post("/api/skin/inpaint-slot")
async def inpaint_slot(payload: InpaintSlotPayload):
    """Per-slot AI Terminal: redraw ONE slot's texture in the active skin.

    Takes the slot's current PNG (SAM-masked from the previous Generate, or
    the original if this slot was never reskinned), sends it to Gemini with
    a prompt to redraw matching the original silhouette, and writes the
    result back. The atlas + per-skin Spine JSON are then re-packed/written
    so the canvas picks up the change without a full Generate.
    """
    project = _State.require_project()
    skin_dir = project.workdir / "skins" / payload.skin_name
    extracted_dir = skin_dir / "extracted"
    extracted_raw_dir = skin_dir / "extracted_raw"
    masks_dir = skin_dir / "masks"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    target = extracted_dir / f"{payload.slot}.png"
    if not target.exists():
        # Slot wasn't part of the original skin; seed it from the project's
        # original part PNG so we have a starting point for Gemini.
        original = project.path / f"{payload.slot}.png"
        if not original.exists():
            raise HTTPException(
                404,
                f"slot {payload.slot!r} has no current texture for skin {payload.skin_name!r}",
            )
        shutil.copy2(original, target)

    SLOT_PROMPT = (
        'Redraw this single character body part in the new style: "{prompt}".\n'
        '\n'
        'CRITICAL CONSTRAINTS:\n'
        '- Keep the EXACT same silhouette, outline, and shape as the input.\n'
        '- Output the part on a transparent (or pure white) background — no '
        'background scenery.\n'
        '- Match the input dimensions exactly. Do not crop or reframe.\n'
        '- Do not add any text, labels, or extra elements.\n'
    )
    prompt = SLOT_PROMPT.format(prompt=payload.prompt)

    logger = _State.logger()
    pre_inpaint = logger.snapshot(target, f"inpaint_{payload.slot}_input")

    import time as _time
    t0 = _time.time()
    gemini_meta: dict = {}
    try:
        await _State.provider().edit_image(
            target,
            prompt,
            negative_prompt=(
                "do not change the silhouette; do not add background scenery; "
                "do not add text or watermarks; do not crop the part"
            ),
            out_path=target,
            metadata=gemini_meta,
        )
    except Exception as e:
        padded_pil = gemini_meta.pop("padded_image_pil", None)
        padded_snap = (
            logger.snapshot(padded_pil, f"inpaint_{payload.slot}_padded")
            if padded_pil is not None else None
        )
        logger.record(
            "gemini_inpaint",
            skin_name=payload.skin_name,
            slot=payload.slot,
            params={"user_prompt": payload.prompt, **gemini_meta},
            input_paths=[pre_inpaint, padded_snap],
            duration_ms=(_time.time() - t0) * 1000,
            status="error",
            error=str(e),
        )
        raise
    padded_pil = gemini_meta.pop("padded_image_pil", None)
    padded_snap = (
        logger.snapshot(padded_pil, f"inpaint_{payload.slot}_padded")
        if padded_pil is not None else None
    )
    post_gemini = logger.snapshot(target, f"inpaint_{payload.slot}_gemini")
    logger.record(
        "gemini_inpaint",
        skin_name=payload.skin_name,
        slot=payload.slot,
        params={"user_prompt": payload.prompt, **gemini_meta},
        input_paths=[pre_inpaint, padded_snap],
        output_paths=[post_gemini],
        duration_ms=(_time.time() - t0) * 1000,
    )

    # Strip whatever background Gemini still painted in. Gemini's prompt
    # asks for transparent/white but it's unreliable; Bria gives clean alpha.
    t1 = _time.time()
    bria_available = bria.available()
    await bria.remove_background(target)
    post_bria = logger.snapshot(target, f"inpaint_{payload.slot}_bria")
    logger.record(
        "bria_remove_background",
        skin_name=payload.skin_name,
        slot=payload.slot,
        params={"available": bria_available},
        input_paths=[post_gemini],
        output_paths=[post_bria],
        duration_ms=(_time.time() - t1) * 1000,
        status="ok" if bria_available else "skipped",
    )

    # The inpaint output IS the new "raw" — Gemini already returns the part
    # on a transparent/white background, so no SAM mask is needed. Without
    # this, _rebuild_skin_atlas would walk extracted_raw/ first, multiply
    # the OLD pre-inpaint raw by the saved SAM mask, and silently discard
    # the new texture (which only lives in extracted/).
    extracted_raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, extracted_raw_dir / f"{payload.slot}.png")
    stale_mask = masks_dir / f"{payload.slot}.png"
    if stale_mask.exists():
        stale_mask.unlink()

    # Track this slot as part of the skin override now that we've edited it
    sam_slots_path = skin_dir / "sam_slots.json"
    sam_slots: list[str] = []
    if sam_slots_path.exists():
        try:
            sam_slots = json.loads(sam_slots_path.read_text()) or []
        except Exception:
            sam_slots = []
    if payload.slot not in sam_slots:
        sam_slots.append(payload.slot)
    sam_slots_path.write_text(json.dumps(sorted(sam_slots), indent=2))

    # Make sure all original parts are present in extracted/ so the atlas is
    # complete (matches what /api/reskin/rebake does).
    for original in project.path.glob("*.png"):
        if original.with_suffix(".atlas").exists():
            continue
        if original.stem.startswith("Spine-"):
            continue
        targ = extracted_dir / original.name
        if not targ.exists():
            shutil.copy2(original, targ)

    # Repack atlas (applying any per-slot masks) + write per-skin Spine JSON
    t2 = _time.time()
    rebuilt = _rebuild_skin_atlas(project, payload.skin_name)
    repacked_atlas = project.path / rebuilt.get("atlas", "")
    logger.record(
        "atlas_repack",
        skin_name=payload.skin_name,
        slot=payload.slot,
        params={"trigger": "inpaint_slot"},
        input_paths=[post_bria],
        output_paths=[logger.snapshot(repacked_atlas, f"atlas_{payload.skin_name}")],
        duration_ms=(_time.time() - t2) * 1000,
    )
    return {"ok": True, "slot": payload.slot, **rebuilt}


@app.post("/api/reskin/generate")
async def reskin_generate(payload: GeneratePayload):
    project = _State.require_project()
    workdir = project.workdir / "skins"

    atlas_path = sheet_path = None
    if payload.method == "atlas":
        pair = atlas_reader.find_atlas_pair(
            project.path, project.spine_json_path.stem
        )
        if pair is None:
            raise HTTPException(
                400,
                f"project has no original atlas+sheet pair under {project.path}; "
                "run scripts/explode_spine_atlas.py or open the original character folder",
            )
        atlas_path, sheet_path = pair

    s = settings_mod.load_settings()
    ref_path = (
        settings_mod.reference_image_path()
        if s.reference.enabled and settings_mod.has_reference_image()
        else None
    )
    ref_prompt = s.reference.prompt.strip() if s.reference.enabled else ""
    logger = _State.logger()
    import time as _time
    t0 = _time.time()
    try:
        result = await full_reskin(
            project_dir=project.path,
            workdir=workdir,
            skin_name=payload.skin_name,
            user_prompt=payload.prompt,
            image_provider=_State.provider(),
            method=payload.method,
            atlas_path=atlas_path,
            atlas_sheet_path=sheet_path,
            reference_image_path=ref_path,
            reference_prompt=ref_prompt,
            pipeline_logger=logger,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.record(
            "reskin_generate",
            skin_name=payload.skin_name,
            params={"method": payload.method, "user_prompt": payload.prompt},
            duration_ms=(_time.time() - t0) * 1000,
            status="error",
            error=str(e),
        )
        raise HTTPException(400, str(e))
    return result


# ───────── Per-slot edit preview ─────────


def _slot_edit_from_dict(d: dict) -> SlotEdit:
    return SlotEdit(
        hue_shift=float(d.get("hue_shift", 0.0)),
        sat_mult=float(d.get("sat_mult", 1.0)),
        light_shift=float(d.get("light_shift", 0.0)),
        brightness=float(d.get("brightness", 0.0)),
        contrast=float(d.get("contrast", 1.0)),
        rgb_balance=tuple(d.get("rgb_balance", (0.0, 0.0, 0.0))),
        dx=float(d.get("dx", 0.0)),
        dy=float(d.get("dy", 0.0)),
        rotation=float(d.get("rotation", 0.0)),
        scale=float(d.get("scale", 1.0)),
    )


def _resolve_slot_png(project: projects.Project, slot: str, skin_name: str) -> Path:
    """Locate the source PNG for a slot under a given skin.

    Order:
      1. .genie/skins/{skin_name}/extracted/{slot}.png  (fresh AI output)
      2. project.path/{slot}.png                         (default skin)
    """
    candidate = (
        project.workdir / "skins" / skin_name / "extracted" / f"{slot}.png"
    )
    if candidate.exists():
        return candidate
    fallback = project.path / f"{slot}.png"
    if fallback.exists():
        return fallback
    raise HTTPException(404, f"no PNG for slot {slot!r} in skin {skin_name!r}")


@app.post("/api/skin/edit-preview")
def edit_preview(payload: EditPayload):
    project = _State.require_project()
    src_path = _resolve_slot_png(project, payload.slot, payload.skin_name)
    img = Image.open(src_path).convert("RGBA")
    edited = apply_edit(img, _slot_edit_from_dict(payload.edit))
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ───────── Export ─────────


@app.post("/api/skin/export")
def export_skin(payload: ExportPayload):
    project = _State.require_project()
    skin_dir = project.workdir / "skins" / payload.skin_name
    extracted = skin_dir / "extracted"
    if not extracted.exists() or not any(extracted.glob("*.png")):
        raise HTTPException(
            400,
            f"no extracted parts for skin {payload.skin_name!r}; generate first",
        )

    # 1) Apply edits, write to a baked/ subdir
    baked = skin_dir / "baked"
    baked.mkdir(parents=True, exist_ok=True)
    for png in extracted.glob("*.png"):
        slot = png.stem
        edit_dict = payload.edits.get(slot, {})
        if edit_dict:
            edited = apply_edit(Image.open(png).convert("RGBA"), _slot_edit_from_dict(edit_dict))
        else:
            edited = Image.open(png).convert("RGBA")
        edited.save(baked / f"{slot}.png")

    # 2) Repack atlas
    atlas_name = f"atlas-{payload.skin_name}"
    pack_result = repack_atlas(baked, project.path, atlas_name)

    # 3) Add skin to Spine JSON
    new_spine = add_skin(
        project.spine_json,
        payload.skin_name,
        placements=pack_result["placements"],
    )

    if payload.write_into_main_json:
        out_path = project.spine_json_path
    else:
        out_path = project.spine_json_path.with_name(
            f"{project.spine_json_path.stem}_{payload.skin_name}.json"
        )

    out_path.write_text(json.dumps(new_spine, indent=2))

    # 4) Drop a folder of named PNGs at {project}/skins/{skin}/ so users can
    #    swap them in manually if desired. Matches the Spine multi-skin convention.
    skins_out = project.path / "skins" / payload.skin_name
    skins_out.mkdir(parents=True, exist_ok=True)
    for png in baked.glob("*.png"):
        (skins_out / png.name).write_bytes(png.read_bytes())

    return {
        "spine_json": str(out_path.relative_to(project.path)),
        "atlas_image": str(pack_result["image"].relative_to(project.path)),
        "atlas_meta": str(pack_result["atlas"].relative_to(project.path)),
        "skin_dir": str(skins_out.relative_to(project.path)),
        "files_copied": sorted(p.name for p in skins_out.glob("*.png")),
    }


# ───────── Settings ─────────


@app.get("/api/settings")
def get_settings():
    return settings_mod.to_payload(settings_mod.load_settings())


@app.put("/api/settings")
async def put_settings(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "expected JSON object")
    s = settings_mod.from_payload(body)
    settings_mod.save_settings(s)
    return settings_mod.to_payload(s)


@app.get("/api/secrets")
def get_secrets():
    """API keys / secret endpoints the user provides. Values come from saved
    overrides (~/.genie-reskin/secrets.json), seeded by the .env."""
    return {"secrets": secrets_store.status_payload()}


@app.put("/api/secrets")
async def put_secrets(request: Request):
    body = await request.json()
    updates = body.get("updates") if isinstance(body, dict) else None
    if not isinstance(updates, dict):
        raise HTTPException(400, "expected {\"updates\": {NAME: value}}")
    secrets_store.save(updates)
    return {"secrets": secrets_store.status_payload()}


@app.get("/api/settings/reference-image")
def get_reference_image():
    p = settings_mod.reference_image_path()
    if not p.is_file():
        raise HTTPException(404, "no reference image")
    return FileResponse(p)


@app.put("/api/settings/reference-image")
async def put_reference_image(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body")
    # Re-encode as PNG via Pillow so we accept any browser-supported format
    # and end up with a single deterministic on-disk format.
    try:
        img = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"unreadable image: {e}")
    p = settings_mod.reference_image_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p, format="PNG")
    return {"ok": True, "size": p.stat().st_size}


@app.delete("/api/settings/reference-image")
def delete_reference_image():
    p = settings_mod.reference_image_path()
    if p.exists():
        p.unlink()
    return {"ok": True}


# ───────── Pipeline logs ─────────


@app.get("/api/logs")
def get_logs(limit: int = 500):
    return {"events": _State.logger().read_events(limit=limit)}


@app.delete("/api/logs")
def clear_logs():
    n = _State.logger().clear()
    return {"ok": True, "cleared": n}


# Convenience: tiny health check
@app.get("/api/health")
def health():
    return {"ok": True}
