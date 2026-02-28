#!/usr/bin/env python3
"""
split_character.py — Generate a sprite-sheet atlas from a full character image
using Google Gemini image generation, then segment individual body parts via
OpenCV connected-components analysis.

Usage:
    python split_character.py <input_image> [--output-dir output_parts]
        [--atlas-out atlas.png] [--min-area 500] [--padding 12]
        [--bg-threshold 240]

Requires:
    pip install google-generativeai opencv-python Pillow numpy
    Environment variable GEMINI_API_KEY must be set.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image


def get_gemini_client():
    """Initialise the Gemini generative-AI client, or exit with a helpful
    error if the API key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY environment variable is not set.\n"
            "Get a free API key at: https://aistudio.google.com/app/apikey\n"
            "Then run:\n"
            "  export GEMINI_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key)
    return client


POSITIVE_PROMPT = (
    "A complete 2D game sprite sheet texture atlas for Spine animation of the "
    "exact character in the reference image. The character is completely "
    "deconstructed into separated, isolated body parts. Separated individual "
    "parts laid out flatly: isolated head, isolated torso, isolated upper arms, "
    "lower arms, hands, upper legs, lower legs, and feet. Spread out with clear "
    "space between every single body part. No overlapping parts. Clean solid "
    "white background. CRITICAL: Maintain the exact same art style, exact same "
    "shading, exact face, and exact color palette as the reference image. "
    "Identical style match, 2D game asset, flat layout, character design sheet."
)

NEGATIVE_PROMPT = (
    "3D, realistic, altered style, different art style, different face, "
    "redesign, overlapping parts, connected limbs, full body standing, dynamic "
    "pose, background scenery, shadows, gradients on background, messy layout, "
    "missing limbs, merged layers, text, watermarks."
)


def generate_atlas(client, input_image_path: str, atlas_out: str) -> str:
    """Send the reference image to Gemini and save the generated atlas PNG."""
    from google.genai import types

    ref_image = Image.open(input_image_path)

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            POSITIVE_PROMPT,
            f"Negative prompt: {NEGATIVE_PROMPT}",
            ref_image,
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extract the generated image from the response parts
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image_data = part.inline_data.data
            with open(atlas_out, "wb") as f:
                f.write(image_data)
            return atlas_out

    print("ERROR: Gemini did not return an image in its response.", file=sys.stderr)
    sys.exit(1)


def segment_parts(
    atlas_path: str,
    output_dir: str,
    min_area: int = 500,
    padding: int = 12,
    bg_threshold: int = 240,
) -> list[str]:
    """Detect individual parts in the atlas using connected-components analysis.

    Returns a list of saved part file paths.
    """
    img = cv2.imread(atlas_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: Could not read atlas image: {atlas_path}", file=sys.stderr)
        sys.exit(1)

    # Convert to RGBA if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Build a foreground mask: pixels whose RGB channels are all below the
    # background threshold are considered foreground.
    bgr = img[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY_INV)

    # Connected-components analysis (8-connectivity)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    os.makedirs(output_dir, exist_ok=True)

    saved: list[str] = []
    part_idx = 0
    h_img, w_img = img.shape[:2]

    for label_id in range(1, num_labels):  # skip background (label 0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        # Apply padding (clamped to image bounds)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, w_img)
        y2 = min(y + h + padding, h_img)

        # Crop the RGBA region
        crop = img[y1:y2, x1:x2].copy()

        # Zero-out pixels that don't belong to this component (make transparent)
        label_region = labels[y1:y2, x1:x2]
        component_mask = label_region == label_id
        crop[~component_mask] = [0, 0, 0, 0]

        out_path = os.path.join(output_dir, f"part_{part_idx:02d}.png")
        cv2.imwrite(out_path, crop)
        saved.append(out_path)
        part_idx += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sprite atlas from a character image using "
        "Gemini, then segment into individual body parts."
    )
    parser.add_argument("input_image", help="Path to the character reference image")
    parser.add_argument(
        "--output-dir",
        default="output_parts",
        help="Directory for cropped part PNGs (default: output_parts)",
    )
    parser.add_argument(
        "--atlas-out",
        default="atlas.png",
        help="Output path for the generated atlas PNG (default: atlas.png)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum component area in pixels to keep (default: 500)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=12,
        help="Padding in pixels around each cropped part (default: 12)",
    )
    parser.add_argument(
        "--bg-threshold",
        type=int,
        default=240,
        help="Grayscale threshold above which pixels are treated as background (default: 240)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        print(f"ERROR: Input image not found: {args.input_image}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Generate atlas ---
    print("[1/3] Generating atlas …")
    client = get_gemini_client()
    generate_atlas(client, args.input_image, args.atlas_out)
    print(f"      Atlas saved to {args.atlas_out}")

    # --- Step 2: Segment parts ---
    print("[2/3] Segmenting parts …")
    parts = segment_parts(
        args.atlas_out,
        args.output_dir,
        min_area=args.min_area,
        padding=args.padding,
        bg_threshold=args.bg_threshold,
    )
    print(f"      Found {len(parts)} parts → {args.output_dir}/")
    for p in parts:
        print(f"        • {os.path.basename(p)}")

    # --- Step 3: Done ---
    print("[3/3] Done ✓")
    print(f"\nParts are in: {args.output_dir}/")
    print("You can now feed them into position_parts.py (Step 1 of the Spine pipeline).")


if __name__ == "__main__":
    main()
