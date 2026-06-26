"""Global app settings — persisted to ~/.genie-reskin/settings.json.

Currently exposes the post-SAM erosion controls used by atlas_rebake to
shave the soft halo Gemini paints at part edges. Defaults preserve the
original size-tiered behaviour.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


SETTINGS_PATH = Path.home() / ".genie-reskin" / "settings.json"
REFERENCE_IMAGE_PATH = Path.home() / ".genie-reskin" / "reference.png"


@dataclass
class ErosionSettings:
    enabled: bool = True
    px_small: int = 0     # part side < small_threshold
    px_medium: int = 1    # < medium_threshold
    px_large: int = 2     # < large_threshold
    px_xlarge: int = 3    # otherwise
    small_threshold: int = 60
    medium_threshold: int = 200
    large_threshold: int = 500

    def px_for_side(self, side: int) -> int:
        if not self.enabled:
            return 0
        if side < self.small_threshold:
            return max(0, int(self.px_small))
        if side < self.medium_threshold:
            return max(0, int(self.px_medium))
        if side < self.large_threshold:
            return max(0, int(self.px_large))
        return max(0, int(self.px_xlarge))


@dataclass
class ReferenceSettings:
    """Optional style-context reference passed to the image model.

    When `enabled` and an image exists at REFERENCE_IMAGE_PATH, the full-
    reskin pipeline prepends the image to the multimodal contents and
    appends `prompt` to the generation prompt.
    """
    enabled: bool = False
    prompt: str = ""


# Segmentation methods used during rebake to cut individual parts out of
# the reskinned atlas. "sam" uses the user's SAM Flask server with bbox
# prompts. "bg_components" runs Bria background removal on the whole atlas
# then matches each region's original silhouette to the highest-IoU
# connected component — works without a SAM server and tends to be cleaner
# for parts whose silhouette closely matches the original layout.
SEGMENTATION_METHODS = ("sam", "bg_components")


@dataclass
class SegmentationSettings:
    method: str = "sam"


@dataclass
class Settings:
    erosion: ErosionSettings = field(default_factory=ErosionSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)


def reference_image_path() -> Path:
    return REFERENCE_IMAGE_PATH


def has_reference_image() -> bool:
    return REFERENCE_IMAGE_PATH.is_file()


def _coerce_erosion(raw: dict) -> ErosionSettings:
    defaults = ErosionSettings()
    return ErosionSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        px_small=int(raw.get("px_small", defaults.px_small)),
        px_medium=int(raw.get("px_medium", defaults.px_medium)),
        px_large=int(raw.get("px_large", defaults.px_large)),
        px_xlarge=int(raw.get("px_xlarge", defaults.px_xlarge)),
        small_threshold=int(raw.get("small_threshold", defaults.small_threshold)),
        medium_threshold=int(raw.get("medium_threshold", defaults.medium_threshold)),
        large_threshold=int(raw.get("large_threshold", defaults.large_threshold)),
    )


def _coerce_reference(raw: dict) -> ReferenceSettings:
    defaults = ReferenceSettings()
    return ReferenceSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        prompt=str(raw.get("prompt", defaults.prompt)),
    )


def _coerce_segmentation(raw: dict) -> SegmentationSettings:
    defaults = SegmentationSettings()
    method = str(raw.get("method", defaults.method))
    if method not in SEGMENTATION_METHODS:
        method = defaults.method
    return SegmentationSettings(method=method)


def load_settings() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()
    try:
        raw = json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    erosion_raw = raw.get("erosion")
    erosion = _coerce_erosion(erosion_raw) if isinstance(erosion_raw, dict) else ErosionSettings()
    reference_raw = raw.get("reference")
    reference = _coerce_reference(reference_raw) if isinstance(reference_raw, dict) else ReferenceSettings()
    segmentation_raw = raw.get("segmentation")
    segmentation = _coerce_segmentation(segmentation_raw) if isinstance(segmentation_raw, dict) else SegmentationSettings()
    return Settings(erosion=erosion, reference=reference, segmentation=segmentation)


def save_settings(settings: Settings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(_persist_payload(settings), indent=2))


def _persist_payload(settings: Settings) -> dict:
    """On-disk shape — no derived fields like `has_image`."""
    return {
        "erosion": asdict(settings.erosion),
        "reference": asdict(settings.reference),
        "segmentation": asdict(settings.segmentation),
    }


def to_payload(settings: Settings) -> dict:
    """API shape — includes derived `has_image` so the UI can show preview state."""
    return {
        "erosion": asdict(settings.erosion),
        "reference": {**asdict(settings.reference), "has_image": has_reference_image()},
        "segmentation": asdict(settings.segmentation),
    }


def from_payload(payload: dict) -> Settings:
    if not isinstance(payload, dict):
        return Settings()
    erosion_raw = payload.get("erosion")
    erosion = _coerce_erosion(erosion_raw) if isinstance(erosion_raw, dict) else ErosionSettings()
    reference_raw = payload.get("reference")
    reference = _coerce_reference(reference_raw) if isinstance(reference_raw, dict) else ReferenceSettings()
    segmentation_raw = payload.get("segmentation")
    segmentation = _coerce_segmentation(segmentation_raw) if isinstance(segmentation_raw, dict) else SegmentationSettings()
    return Settings(erosion=erosion, reference=reference, segmentation=segmentation)
