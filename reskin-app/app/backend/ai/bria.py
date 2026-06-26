"""fal.ai Bria RMBG-2.0 background removal.

Used to strip the background Gemini paints behind inpainted body parts in
the AI Terminal flow. Gemini's "transparent or pure white background"
prompt isn't reliable, so we run the result through Bria to get clean
alpha before the rebake atlas-packs the texture.

Requires FAL_KEY in the environment. If unset, `available()` returns False
and `remove_background()` is a no-op (logs a warning, leaves the file).
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import requests
from PIL import Image


BRIA_APP = "fal-ai/bria/background/remove"


def available() -> bool:
    return bool(os.environ.get("FAL_KEY"))


def _save_bria_output(image_path: Path, out_url: str) -> Path:
    """Common tail for sync + async paths: download, resize-back, save."""
    try:
        resp = requests.get(out_url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[bria] download failed: {e}", flush=True)
        return image_path
    try:
        cut = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        print(f"[bria] decode failed: {e}", flush=True)
        return image_path
    original = Image.open(image_path).convert("RGBA")
    if cut.size != original.size:
        cut = cut.resize(original.size, Image.LANCZOS)
    cut.save(image_path, format="PNG")
    return image_path


def remove_background_sync(image_path: Path) -> Path:
    """Sync version — use from non-async code paths (e.g. the rebake route)."""
    if not available():
        print("[bria] FAL_KEY not set — skipping background removal", flush=True)
        return image_path
    import fal_client
    try:
        url = fal_client.upload_file(str(image_path))
    except Exception as e:
        print(f"[bria] upload failed: {e}", flush=True)
        return image_path
    try:
        result = fal_client.subscribe(BRIA_APP, arguments={"image_url": url}, with_logs=False)
    except Exception as e:
        print(f"[bria] inference failed: {e}", flush=True)
        return image_path
    out_url = (result or {}).get("image", {}).get("url")
    if not out_url:
        print(f"[bria] no image url in response: {result}", flush=True)
        return image_path
    return _save_bria_output(image_path, out_url)


async def remove_background(image_path: Path) -> Path:
    """Strip the background of `image_path` in place via Bria RMBG-2.0.

    Returns the same path. Falls through silently (with a printed warning)
    if FAL_KEY is missing or the API call fails — the caller still has the
    original Gemini output to fall back on.
    """
    if not available():
        print("[bria] FAL_KEY not set — skipping background removal", flush=True)
        return image_path

    import fal_client

    try:
        # Upload the local file to fal storage so the API can fetch by URL.
        # upload_file_async hashes the bytes server-side — repeats are cheap.
        url = await fal_client.upload_file_async(str(image_path))
    except Exception as e:
        print(f"[bria] upload failed: {e}", flush=True)
        return image_path

    try:
        result = await fal_client.subscribe_async(
            BRIA_APP,
            arguments={"image_url": url},
            with_logs=False,
        )
    except Exception as e:
        print(f"[bria] inference failed: {e}", flush=True)
        return image_path

    out_url = (result or {}).get("image", {}).get("url")
    if not out_url:
        print(f"[bria] no image url in response: {result}", flush=True)
        return image_path

    return _save_bria_output(image_path, out_url)
