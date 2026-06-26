"""Stub for the user's SAM segmentation server.

The user said: "I have a segmentation server with SAM, I'll provide you access
to this when we get there." When that arrives, fill in `segment()` to POST the
image and parse a list of {bbox, centroid, area, mask} components.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import Component


class SAMProvider:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.environ.get("SAM_SERVER_URL")

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    async def segment(self, image_path: Path) -> list[Component]:
        if not self.base_url:
            raise RuntimeError(
                "SAM_SERVER_URL is not configured — set the env var to enable SAM"
            )
        # TODO: implement once the server URL + protocol are known.
        # Expected: POST {image} -> [{ "bbox": [...], "centroid": [...], "area": N, "mask_b64": ... }]
        raise NotImplementedError("SAMProvider.segment not implemented yet")
