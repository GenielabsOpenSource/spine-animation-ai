#!/usr/bin/env python3
"""
generate_spine_player.py — Generate a self-contained HTML preview using the official Spine Web Player.

Embeds the skeleton JSON, atlas text, and atlas PNG as base64 data URIs via the
rawDataURIs configuration, so the resulting HTML file works standalone — no server needed.

Uses the official @esotericsoftware/spine-player from UNPKG CDN.

Usage:
    python3 generate_spine_player.py \
        --skeleton skeleton.json \
        --atlas skeleton.atlas \
        --atlas-image skeleton.png \
        --output preview.html \
        [--animation idle] \
        [--background "#1a1a2eff"] \
        [--skin default]

If no --atlas and --atlas-image are given but a --parts directory is provided,
the script will pack the parts into an atlas automatically.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path


def file_to_base64(path):
    """Read a file and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def json_to_base64(path):
    """Read a JSON file and return it as base64."""
    with open(path, "r") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


def text_to_base64(path):
    """Read a text file and return it as base64."""
    with open(path, "r") as f:
        content = f.read()
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


def find_atlas_images(atlas_path):
    """Parse an atlas file to find all referenced PNG filenames."""
    atlas_dir = os.path.dirname(os.path.abspath(atlas_path))
    images = []

    with open(atlas_path, "r") as f:
        lines = f.readlines()

    # The first line (or lines before the first region entry) contain page image filenames
    # Atlas format: image filename is a line that ends with .png (or other image ext)
    # followed by size:, format:, filter:, repeat: lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # A page image is the first non-empty line, or any line ending with an image extension
        # that is followed by "size:" on the next line
        if line and not line.startswith(" ") and not ":" in line:
            # Check if next line starts with "size:" indicating this is a page name
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("size:"):
                img_path = os.path.join(atlas_dir, line)
                if os.path.exists(img_path):
                    images.append((line, img_path))
                else:
                    print(f"  WARNING: Atlas references '{line}' but file not found at {img_path}")
        i += 1

    return images


def generate_html(skeleton_path, atlas_path, atlas_images,
                  animation=None, skin=None, bg_color="#1a1a2eff",
                  show_controls=True, title="Spine Animation Preview"):
    """Generate the complete HTML file with embedded Spine Web Player."""

    # Get filenames for rawDataURIs keys
    skel_filename = os.path.basename(skeleton_path)
    atlas_filename = os.path.basename(atlas_path)

    # Determine if JSON or binary
    is_json = skel_filename.lower().endswith(".json")
    skel_mime = "application/json" if is_json else "application/octet-stream"

    # Encode all assets
    skel_b64 = file_to_base64(skeleton_path)
    atlas_b64 = file_to_base64(atlas_path)

    # Build rawDataURIs object
    raw_data_entries = []
    raw_data_entries.append(
        f'        "{skel_filename}": "data:{skel_mime};base64,{skel_b64}"'
    )
    raw_data_entries.append(
        f'        "{atlas_filename}": "data:application/octet-stream;base64,{atlas_b64}"'
    )

    for img_name, img_path in atlas_images:
        img_ext = Path(img_path).suffix.lower()
        img_mime = "image/png" if img_ext == ".png" else "image/jpeg"
        img_b64 = file_to_base64(img_path)
        raw_data_entries.append(
            f'        "{img_name}": "data:{img_mime};base64,{img_b64}"'
        )

    raw_data_uris_js = ",\n".join(raw_data_entries)

    # Build config options
    config_lines = []
    config_lines.append(f'      skeleton: "{skel_filename}"')
    config_lines.append(f'      atlas: "{atlas_filename}"')

    if animation:
        config_lines.append(f'      animation: "{animation}"')

    if skin and skin != "default":
        config_lines.append(f'      skin: "{skin}"')

    config_lines.append(f'      backgroundColor: "{bg_color}"')
    config_lines.append(f'      showControls: {"true" if show_controls else "false"}')
    config_lines.append(f'      premultipliedAlpha: false')

    config_lines.append(f'      rawDataURIs: {{\n{raw_data_uris_js}\n      }}')

    # Error/success callbacks
    config_lines.append("""      success: function(player) {
        document.getElementById('status').textContent = 'Loaded successfully';
        document.getElementById('status').style.color = '#4ade80';
        // Log available animations
        var anims = player.skeleton.data.animations.map(function(a) { return a.name; });
        console.log('Available animations:', anims);
        var skins = player.skeleton.data.skins.map(function(s) { return s.name; });
        console.log('Available skins:', skins);
      }""")
    config_lines.append("""      error: function(player, reason) {
        document.getElementById('status').textContent = 'Error: ' + reason;
        document.getElementById('status').style.color = '#ef4444';
        console.error('Spine Player error:', reason);
      }""")

    config_js = ",\n".join(config_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>

<!-- Official Spine Web Player -->
<script src="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/iife/spine-player.js"></script>
<link rel="stylesheet" href="https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/spine-player.css">

<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 20px;
  }}
  h1 {{
    font-size: 1.5em;
    margin-bottom: 10px;
    color: #a8b2d1;
    letter-spacing: 0.04em;
  }}
  #status {{
    font-size: 0.85em;
    margin-bottom: 15px;
    color: #6b7da0;
    transition: color 0.3s;
  }}
  #player-container {{
    width: 700px;
    height: 600px;
    max-width: 95vw;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }}
  .info {{
    margin-top: 15px;
    font-size: 0.8em;
    color: #4a5568;
    text-align: center;
    max-width: 600px;
    line-height: 1.5;
  }}
  .info a {{ color: #6b8aad; text-decoration: none; }}
  .info a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div id="status">Loading Spine Player...</div>
<div id="player-container"></div>

<div class="info">
  Rendered with the official
  <a href="https://en.esotericsoftware.com/spine-player" target="_blank">Spine Web Player</a>.
  Use the controls to switch animations, adjust speed, and toggle debug views.
</div>

<script>
  new spine.SpinePlayer("player-container", {{
{config_js}
  }});
</script>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML preview using the official Spine Web Player"
    )
    parser.add_argument("--skeleton", required=True, help="Spine JSON or binary (.skel) file")
    parser.add_argument("--atlas", required=True, help="Spine .atlas file")
    parser.add_argument("--atlas-image", default=None,
                        help="Atlas PNG image (auto-detected from atlas if omitted)")
    parser.add_argument("--output", default="preview.html", help="Output HTML file")
    parser.add_argument("--animation", default=None, help="Default animation to play")
    parser.add_argument("--skin", default=None, help="Default skin")
    parser.add_argument("--background", default="#1a1a2eff", help="Background color (hex RGBA)")
    parser.add_argument("--title", default="Spine Animation Preview", help="Page title")
    parser.add_argument("--no-controls", action="store_true", help="Hide player controls")
    args = parser.parse_args()

    # Verify files exist
    for path, name in [(args.skeleton, "Skeleton"), (args.atlas, "Atlas")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    # Find atlas images
    atlas_images = []
    if args.atlas_image:
        img_name = os.path.basename(args.atlas_image)
        atlas_images.append((img_name, args.atlas_image))
    else:
        atlas_images = find_atlas_images(args.atlas)

    if not atlas_images:
        print("ERROR: No atlas images found. Specify --atlas-image or ensure atlas references valid PNGs.")
        sys.exit(1)

    print(f"Skeleton: {args.skeleton}")
    print(f"Atlas: {args.atlas}")
    for img_name, img_path in atlas_images:
        print(f"Atlas image: {img_name} ({os.path.getsize(img_path) / 1024:.1f} KB)")

    # Generate HTML
    print("Generating HTML with embedded Spine Web Player...")
    html = generate_html(
        skeleton_path=args.skeleton,
        atlas_path=args.atlas,
        atlas_images=atlas_images,
        animation=args.animation,
        skin=args.skin,
        bg_color=args.background,
        show_controls=not args.no_controls,
        title=args.title,
    )

    with open(args.output, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nPreview saved: {args.output} ({size_kb:.1f} KB)")
    print("Open in a browser to see your animation (requires internet for the Spine Player CDN).")


if __name__ == "__main__":
    main()
