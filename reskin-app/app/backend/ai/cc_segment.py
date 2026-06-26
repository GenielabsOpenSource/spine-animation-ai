"""Background-removal + connected-components + IoU segmenter.

Drop-in alternative to SAMProvider for atlas_rebake. Pipeline:

  1. Strip the reskinned atlas's background via Bria (writes alpha for all
     non-background pixels).
  2. Find every connected component in the resulting alpha (8-connected).
  3. For each region (with known bbox in the atlas + the original part's
     silhouette from the project's per-part PNGs), compute IoU against
     every component and pick the highest-scoring one.
  4. Return per-region masks shaped like SAMProvider's output:
     `{slot_name: SamMask(slot, mask, score)}`, where `mask` is a full
     image-sized 8-bit greyscale PIL image (white = part, black = drop).

Falls back to a per-region rectangle mask if no component overlaps the
region's bbox. Set CC_SEGMENT_MIN_IOU to require a minimum match — under
that, we leave the slot un-masked (the rebake will use the raw crop).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


MIN_COMPONENT_PIXELS = 64  # ignore tiny noise blobs
DEFAULT_MIN_IOU = float(os.environ.get("CC_SEGMENT_MIN_IOU", "0.05"))


@dataclass
class CcMask:
    """Mirrors ai.sam_provider.SamMask so it's a drop-in replacement."""
    slot: str
    mask: Image.Image
    score: Optional[float]


class ComponentSegmenter:
    """Build with a callable that resolves a region name to its original
    silhouette mask (full atlas-sized boolean array). The rebake passes
    one in because where the silhouette comes from depends on whether
    we're in atlas mode (original .atlas regions) or exploded mode
    (per-part PNG placements)."""

    def __init__(self, *, min_iou: float = DEFAULT_MIN_IOU):
        self.min_iou = min_iou

    @property
    def available(self) -> bool:
        # Hard requirements: Bria (FAL_KEY) and OpenCV/numpy. Bria is
        # checked at request time so settings can flip on/off without
        # restart.
        from . import bria
        return bria.available()

    def segment_with_bboxes(
        self,
        image_path: Path,
        bboxes: dict[str, dict],
        *,
        original_silhouettes: dict[str, np.ndarray],
    ) -> dict[str, CcMask]:
        """Run Bria → label CCs → IoU-match to each region.

        `original_silhouettes` is a map slot/region name → boolean array
        sized to the atlas (True where the original part is opaque).
        """
        if not self.available:
            raise RuntimeError(
                "FAL_KEY not configured — set env var to enable bg_components segmentation"
            )

        cleaned = self._bria_clean(image_path)  # full-sized RGBA np array
        alpha = cleaned[..., 3]
        sheet_h, sheet_w = alpha.shape

        binary = (alpha > 16).astype(np.uint8)
        n_labels, labels = self._connected_components(binary)
        components = self._build_component_masks(labels, n_labels)
        print(
            f"[cc] {len(components)} components ≥ {MIN_COMPONENT_PIXELS} px "
            f"in {sheet_w}x{sheet_h} atlas",
            flush=True,
        )

        out: dict[str, CcMask] = {}
        for name in bboxes.keys():
            silhouette = original_silhouettes.get(name)
            if silhouette is None or not silhouette.any():
                continue
            best_iou = 0.0
            best_idx = -1
            for idx, comp in enumerate(components):
                iou = _iou(silhouette, comp)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0 or best_iou < self.min_iou:
                continue
            mask_arr = (components[best_idx].astype(np.uint8) * 255)
            out[name] = CcMask(
                slot=name,
                mask=Image.fromarray(mask_arr, mode="L"),
                score=float(best_iou),
            )
        print(f"[cc] matched {len(out)}/{len(bboxes)} regions", flush=True)
        return out

    # ───────── internals ─────────

    def _bria_clean(self, image_path: Path) -> np.ndarray:
        """Background-remove via Bria. We work on a copy in a temp file
        because Bria writes back in place, and we'd rather not mutate the
        reskinned atlas the caller is going to crop from."""
        from . import bria
        tmp = image_path.with_name(image_path.stem + "_bria_tmp" + image_path.suffix)
        Image.open(image_path).convert("RGBA").save(tmp)
        bria.remove_background_sync(tmp)
        cleaned = np.asarray(Image.open(tmp).convert("RGBA"))
        try:
            tmp.unlink()
        except OSError:
            pass
        return cleaned

    def _connected_components(self, binary: np.ndarray) -> tuple[int, np.ndarray]:
        try:
            import cv2
            n, labels = cv2.connectedComponents(binary, connectivity=8)
            return n, labels
        except Exception:
            # numpy fallback (rare — opencv is in requirements.txt)
            from scipy import ndimage  # type: ignore
            labels, n = ndimage.label(binary)
            return n + 1, labels

    def _build_component_masks(
        self,
        labels: np.ndarray,
        n_labels: int,
    ) -> list[np.ndarray]:
        # Label 0 is background.
        out: list[np.ndarray] = []
        for i in range(1, n_labels):
            m = labels == i
            if int(m.sum()) < MIN_COMPONENT_PIXELS:
                continue
            out.append(m)
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / max(1, union)


async def remove_background_async(image_path: Path) -> Path:
    """Async wrapper so callers in async contexts (the FastAPI handlers)
    can use the cleaner directly without the asyncio shenanigans in
    `_bria_clean`."""
    from . import bria
    return await bria.remove_background(image_path)
