"""Provider interface for AI image-edit and segmentation.

Lets us start with Gemini and slot in the user's SAM segmentation server (or
a SDXL+ControlNet provider) without touching the reskin pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Component:
    """A segmented region returned by a segmentation provider."""
    bbox: tuple[int, int, int, int]  # x, y, w, h in image pixel coords
    centroid: tuple[float, float]
    area: int
    mask: bytes | None = None  # optional 8-bit binary mask the size of bbox


class ImageEditProvider(Protocol):
    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        negative_prompt: str = "",
        out_path: Path | None = None,
    ) -> Path:
        """Reskin/edit an image given a text prompt. Returns the saved PNG path."""
        ...


class SegmentationProvider(Protocol):
    async def segment(self, image_path: Path) -> list[Component]:
        """Return labeled components for the image (one per body part, ideally)."""
        ...
