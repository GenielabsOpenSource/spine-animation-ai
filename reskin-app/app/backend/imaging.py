"""Non-destructive per-slot edits.

Each function takes an RGBA PIL Image, returns a new RGBA PIL Image. Pure —
no I/O. The frontend's `SlotEdit` shape mirrors these params; at export time
they're applied in order: color → transform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class SlotEdit:
    # Color
    hue_shift: float = 0.0       # degrees, -180..180
    sat_mult: float = 1.0        # 0..3, 1 = no change
    light_shift: float = 0.0     # -1..1 (additive in HLS lightness space)
    brightness: float = 0.0      # -1..1 additive
    contrast: float = 1.0        # 0..3, 1 = no change
    rgb_balance: tuple[float, float, float] = (0.0, 0.0, 0.0)  # additive offsets, -1..1
    # Transform
    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0  # degrees
    scale: float = 1.0


def _split_alpha(im: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    return arr[..., :3], arr[..., 3:4]


def _merge(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    rgb = np.clip(rgb, 0.0, 1.0)
    out = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGBA")


def _rgb_to_hls(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB → HLS. rgb in [0,1]. Returns (..., 3) with H in [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    l = (maxc + minc) / 2.0
    delta = maxc - minc

    s = np.zeros_like(l)
    mask = delta > 1e-9
    upper = l > 0.5
    s[mask & upper] = (delta / (2.0 - maxc - minc))[mask & upper]
    s[mask & ~upper] = (delta / (maxc + minc + 1e-9))[mask & ~upper]

    rc = np.where(delta > 1e-9, (maxc - r) / np.where(delta > 1e-9, delta, 1.0), 0.0)
    gc = np.where(delta > 1e-9, (maxc - g) / np.where(delta > 1e-9, delta, 1.0), 0.0)
    bc = np.where(delta > 1e-9, (maxc - b) / np.where(delta > 1e-9, delta, 1.0), 0.0)

    h = np.zeros_like(l)
    h = np.where((maxc == r) & mask, bc - gc, h)
    h = np.where((maxc == g) & mask, 2.0 + rc - bc, h)
    h = np.where((maxc == b) & mask, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0

    return np.stack([h, l, s], axis=-1)


def _hls_to_rgb(hls: np.ndarray) -> np.ndarray:
    h, l, s = hls[..., 0], hls[..., 1], hls[..., 2]

    def hue(p, q, t):
        t = t % 1.0
        out = np.where(t < 1 / 6, p + (q - p) * 6 * t, np.zeros_like(t))
        out = np.where((t >= 1 / 6) & (t < 1 / 2), q, out)
        out = np.where(
            (t >= 1 / 2) & (t < 2 / 3), p + (q - p) * (2 / 3 - t) * 6, out
        )
        out = np.where(t >= 2 / 3, p, out)
        return out

    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    r = hue(p, q, h + 1 / 3)
    g = hue(p, q, h)
    b = hue(p, q, h - 1 / 3)
    # Where saturation is zero, output is just luminance
    flat = np.stack([l, l, l], axis=-1)
    out = np.stack([r, g, b], axis=-1)
    return np.where(s[..., None] < 1e-9, flat, out)


def apply_color(im: Image.Image, edit: SlotEdit) -> Image.Image:
    rgb, alpha = _split_alpha(im)
    if (edit.hue_shift, edit.sat_mult, edit.light_shift) != (0.0, 1.0, 0.0):
        hls = _rgb_to_hls(rgb)
        hls[..., 0] = (hls[..., 0] + edit.hue_shift / 360.0) % 1.0
        hls[..., 1] = np.clip(hls[..., 1] + edit.light_shift, 0.0, 1.0)
        hls[..., 2] = np.clip(hls[..., 2] * edit.sat_mult, 0.0, 1.0)
        rgb = _hls_to_rgb(hls)

    if edit.contrast != 1.0:
        rgb = (rgb - 0.5) * edit.contrast + 0.5
    if edit.brightness != 0.0:
        rgb = rgb + edit.brightness
    if edit.rgb_balance != (0.0, 0.0, 0.0):
        rgb = rgb + np.array(edit.rgb_balance, dtype=np.float32)

    return _merge(rgb, alpha)


def apply_transform(im: Image.Image, edit: SlotEdit) -> Image.Image:
    """Rotate around the image center, scale, then translate by (dx, dy).

    The output canvas grows to fit the rotated content. dx/dy are applied AFTER
    the geometric transform — they shift the visual content within the
    expanded canvas.
    """
    out = im
    if edit.scale != 1.0:
        new_w = max(1, int(round(out.width * edit.scale)))
        new_h = max(1, int(round(out.height * edit.scale)))
        out = out.resize((new_w, new_h), Image.LANCZOS)
    if edit.rotation != 0.0:
        out = out.rotate(edit.rotation, resample=Image.BICUBIC, expand=True)
    if edit.dx != 0.0 or edit.dy != 0.0:
        # Place the (transformed) image on a padded canvas with the offset.
        pad_x = max(0, int(abs(edit.dx)))
        pad_y = max(0, int(abs(edit.dy)))
        canvas = Image.new(
            "RGBA",
            (out.width + pad_x * 2, out.height + pad_y * 2),
            (0, 0, 0, 0),
        )
        canvas.paste(out, (pad_x + int(edit.dx), pad_y + int(edit.dy)), out)
        out = canvas
    return out


def apply_edit(im: Image.Image, edit: SlotEdit) -> Image.Image:
    return apply_transform(apply_color(im, edit), edit)
