"""GCP SAM segmentation provider.

Talks to the user's standalone SAM (Segment Anything) Flask server. The
expected endpoint:

    POST {SAM_SERVER_URL}/segment_with_boxes
    Content-Type: application/json
    Body:  { "image_base64": "...", "boxes": [{slot_id, x_min, y_min, x_max, y_max}, ...] }
    Returns: { "masks": [{slot_id, score, mask_b64}, ...] }

That endpoint runs `predictor.set_image()` ONCE per request, then prompts
the SAM predictor with each box — cheap fan-out after the embedding is
computed. So our reskin uses a single HTTP call regardless of slot count.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class SamMask:
    slot: str
    mask: Image.Image  # 8-bit greyscale, image-sized
    score: float | None


class SAMProvider:
    def __init__(self, base_url: str | None = None, timeout_s: int = 600):
        self.base_url = (base_url or os.environ.get("SAM_SERVER_URL") or "").rstrip("/")
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def segment_with_bboxes(
        self,
        image_path: Path,
        bboxes: dict[str, dict],
    ) -> dict[str, SamMask]:
        """Send all bboxes in ONE POST. Return {slot: SamMask} keyed by slot name."""
        if not self.available:
            raise RuntimeError(
                "SAM_SERVER_URL not configured — set env var to enable SAM segmentation"
            )

        import requests

        # Encode the image as base64 PNG so the server can decode without an
        # external fetch (works when our server is on localhost).
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        slot_order = list(bboxes.keys())
        boxes_payload: list[dict[str, Any]] = []
        for slot in slot_order:
            bb = bboxes[slot]
            boxes_payload.append({
                "slot_id": slot,
                "x_min": int(round(bb["x"])),
                "y_min": int(round(bb["y"])),
                "x_max": int(round(bb["x"] + bb["w"])),
                "y_max": int(round(bb["y"] + bb["h"])),
            })

        url = f"{self.base_url}/segment_with_boxes"
        print(f"[sam] POST {url} with {len(boxes_payload)} boxes", flush=True)
        r = requests.post(
            url,
            json={"image_base64": image_b64, "boxes": boxes_payload},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        payload = r.json()

        out: dict[str, SamMask] = {}
        for m in payload.get("masks") or []:
            slot = m.get("slot_id")
            if not slot:
                continue
            mask_bytes = base64.b64decode(m["mask_b64"])
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            out[slot] = SamMask(slot=slot, mask=mask_img, score=m.get("score"))
        print(f"[sam] got {len(out)} masks", flush=True)
        return out
