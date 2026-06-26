"""Gemini Nano Banana image-edit provider.

Ported from scripts/reskin_character.py. Wraps `google-genai`'s
generate_content with multimodal input and pulls the inline image bytes from
the response.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image


DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def _patch_image_config_for_image_size() -> None:
    """Older google-genai (≤1.47) ships an ImageConfig pydantic model that
    only declares `aspect_ratio`; passing `image_size` raises
    ExtraForbidden and the SDK silently falls back to "no image_config".
    Newer SDKs (1.69+) add the field but don't run on Python 3.9.
    Workaround: relax the model to allow extra fields so we can pass
    `imageSize` straight through. Serialization with by_alias still emits
    the camelCase wire shape (`imageConfig.imageSize`) Gemini expects.
    """
    try:
        from google.genai import types as _types
        cfg = dict(getattr(_types.ImageConfig, "model_config", {}) or {})
        if cfg.get("extra") == "allow":
            return
        cfg["extra"] = "allow"
        _types.ImageConfig.model_config = cfg
        _types.ImageConfig.model_rebuild(force=True)
    except Exception:
        # If the patch fails for any reason, fall through — edit_image will
        # detect the rejection at request-build time and degrade gracefully.
        pass


_patch_image_config_for_image_size()


class GeminiProvider:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set — get one at https://aistudio.google.com/app/apikey"
            )
        from google import genai
        self._client = genai.Client(api_key=api_key)
        return self._client

    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        negative_prompt: str = "",
        out_path: Path | None = None,
        reference_images: list[Path] = (),
        metadata: dict | None = None,
    ) -> Path:
        """Reskin via Gemini with strict 1:1 input/output pixel mapping.

        Pipeline:
          1. Pad the input image to the closest aspect ratio Gemini supports
             (white padding). Record the original area's bbox inside the
             padded canvas.
          2. Send the padded image with that aspect_ratio.
          3. Receive the reskinned image.
          4. Resize Gemini's output to the padded canvas's dimensions.
          5. Crop the resized output back to the original area's bbox.
          6. Save → output is exactly the same size as the input, with no
             reframing or scaling distortion. Bbox coords from the live
             skeleton apply 1:1 to the output.
        """
        from google.genai import types

        client = self._get_client()
        original = Image.open(image_path).convert("RGBA")
        orig_w, orig_h = original.size

        aspect, target_w, target_h, pad_x, pad_y = _pad_to_aspect_ratio(orig_w, orig_h)
        padded = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
        padded.paste(original, (pad_x, pad_y), original)

        contents: list = [prompt]
        if negative_prompt:
            contents.append(f"Negative prompt: {negative_prompt}")
        # Style-context references go first so the prompt's "FIRST input image
        # is a reference" wording matches the actual order the model sees.
        ref_sizes: list[list[int]] = []
        for ref_path in reference_images or ():
            try:
                ref_img = Image.open(ref_path).convert("RGBA")
            except Exception:
                continue
            ref_sizes.append([ref_img.width, ref_img.height])
            contents.append(ref_img)
        contents.append(padded)

        image_size = _pick_image_size(target_w, target_h)
        config_kwargs: dict = {"response_modalities": ["IMAGE", "TEXT"]}
        accepted_kwargs: dict = {}
        # Use camelCase kwargs so the value lands under the alias
        # name when ImageConfig is serialized with by_alias=True (which is
        # what the SDK uses to build the wire request). Without this, the
        # extra field would be sent as `image_size` (snake_case) and Gemini
        # would silently drop it.
        for ic_kwargs in (
            {"aspectRatio": aspect, "imageSize": image_size},
            {"aspectRatio": aspect},
            {},
        ):
            try:
                if ic_kwargs:
                    config_kwargs["image_config"] = types.ImageConfig(**ic_kwargs)
                accepted_kwargs = ic_kwargs
                break
            except Exception:
                continue

        # Surface every size manipulation + the actual model inputs so the
        # caller can attach this to its log event. Caller picks what to keep.
        if metadata is not None:
            metadata["model"] = self.model
            metadata["orig_size"] = [orig_w, orig_h]
            metadata["padded_size"] = [target_w, target_h]
            metadata["padding"] = [pad_x, pad_y]
            metadata["aspect_ratio"] = aspect
            metadata["image_size_picked"] = image_size
            metadata["image_size_actually_sent"] = (
                "imageSize" in accepted_kwargs or "image_size" in accepted_kwargs
            )
            metadata["image_config_kwargs_sent"] = accepted_kwargs
            metadata["reference_count"] = len(ref_sizes)
            metadata["reference_sizes"] = ref_sizes
            metadata["prompt_chars"] = len(prompt)
            metadata["prompt_preview"] = (
                prompt[:400] + "…" if len(prompt) > 400 else prompt
            )
            metadata["negative_prompt"] = negative_prompt
            metadata["contents_summary"] = _summarize_contents(contents)
            # Caller can snapshot this PIL image directly via PipelineLogger.
            metadata["padded_image_pil"] = padded.copy()

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        if out_path is None:
            out_path = image_path.with_name(image_path.stem + "_reskinned.png")

        for part in response.candidates[0].content.parts:
            if part.inline_data is None:
                continue
            # Decode response → resize to padded dims → crop padding away.
            import io
            generated = Image.open(io.BytesIO(part.inline_data.data)).convert("RGBA")
            if metadata is not None:
                metadata["model_output_size"] = [generated.width, generated.height]
                metadata["upscaled_to_padded"] = generated.size != (target_w, target_h)
            if generated.size != (target_w, target_h):
                generated = generated.resize((target_w, target_h), Image.LANCZOS)
            cropped = generated.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h))
            cropped.save(out_path, format="PNG")
            return out_path

        raise RuntimeError("Gemini did not return an image in its response")

    async def segment(self, image_path: Path):  # not implemented; SAM handles this
        raise NotImplementedError("GeminiProvider does not implement segment()")


# ─── Aspect ratio + size helpers ────────────────────────────────────────────

# Aspect ratios that Gemini Nano Banana / image-gen models accept.
_SUPPORTED_RATIOS: list[tuple[str, float]] = [
    ("1:1",   1 / 1),
    ("4:3",   4 / 3),
    ("3:4",   3 / 4),
    ("3:2",   3 / 2),
    ("2:3",   2 / 3),
    ("16:9",  16 / 9),
    ("9:16",  9 / 16),
    ("21:9",  21 / 9),
    ("9:21",  9 / 21),
]


def _pick_aspect_ratio(w: int, h: int) -> str:
    target = w / h if h > 0 else 1.0
    return min(_SUPPORTED_RATIOS, key=lambda kv: abs(kv[1] - target))[0]


def _pad_to_aspect_ratio(w: int, h: int) -> tuple[str, int, int, int, int]:
    """Pick the closest supported aspect ratio and compute padding.

    Returns (aspect_str, target_w, target_h, pad_x, pad_y) where (pad_x, pad_y)
    is where the original image is pasted inside the padded canvas. The
    padded canvas exactly matches the chosen aspect ratio so Gemini can
    return without re-cropping.
    """
    target_aspect_str = _pick_aspect_ratio(w, h)
    target_aspect = next(v for k, v in _SUPPORTED_RATIOS if k == target_aspect_str)
    in_aspect = w / h if h > 0 else 1.0

    if in_aspect >= target_aspect:
        # Input is wider than (or equal to) target → pad top + bottom
        target_w = w
        target_h = int(round(w / target_aspect))
        pad_x = 0
        pad_y = max(0, (target_h - h) // 2)
    else:
        # Input is taller → pad left + right
        target_h = h
        target_w = int(round(h * target_aspect))
        pad_y = 0
        pad_x = max(0, (target_w - w) // 2)

    return target_aspect_str, target_w, target_h, pad_x, pad_y


def _pick_image_size(w: int, h: int) -> str:
    """Map the input's longest side to one of Gemini's supported sizes."""
    longest = max(w, h)
    if longest >= 3000:
        return "4K"
    if longest >= 1500:
        return "2K"
    return "1K"


def _summarize_contents(contents: list) -> list[dict]:
    """Describe what was actually sent to generate_content, in order.

    Each entry is one item the model sees: text snippets are truncated to
    80 chars; images carry their dimensions. Lets the Logs tab show the
    exact ordering of references vs. the target image."""
    out: list[dict] = []
    for c in contents:
        if isinstance(c, str):
            out.append(
                {"kind": "text", "chars": len(c), "preview": c[:80] + ("…" if len(c) > 80 else "")}
            )
        elif isinstance(c, Image.Image):
            out.append({"kind": "image", "size": [c.width, c.height], "mode": c.mode})
        else:
            out.append({"kind": type(c).__name__})
    return out
