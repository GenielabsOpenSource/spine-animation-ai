# 🎭 Spine Animation AI

> An AI-powered Claude skill for creating, rigging, and refining Spine 2D skeletal animations — from raw assets to interactive previews.

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Spine Version](https://img.shields.io/badge/Spine-4.2-blue.svg)](http://esotericsoftware.com)
[![Claude Skill](https://img.shields.io/badge/Claude-Skill-orange.svg)](SKILL.md)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Compatible-green.svg)](https://openclaw.ai)

---

<p align="center">
  <img src="demo/preview.gif" alt="Sombrero character idle animation" width="300"/>
  &nbsp;&nbsp;&nbsp;
  <img src="demo/editor-screenshot.png" alt="Part editor UI" width="480"/>
</p>

> **Live Demo →** [Open the interactive editor](https://GenielabsOpenSource.github.io/spine-animation-ai/demo/sombrero_editor.html)

---

## What Is This?

**Spine Animation AI** is a [Claude agent skill](SKILL.md) that lets AI handle the tedious parts of Spine 2D animation production:

- **Auto-position** body parts from reference images using SIFT + RANSAC
- **Build skeleton JSON** with proper bone hierarchy and draw order
- **Generate animations** (idle, walk, run, attack, wave, jump) using the 12 principles of animation
- **Produce interactive HTML5 previews** using the official Spine Web Player
- **Correct and refine** existing skeletons with AI-assisted offset adjustments

Think of it as a Spine rigging co-pilot. You provide the art assets; Claude does the math.

---

## Quick Start

### With OpenClaw / Claude Agent

Install the skill:

```bash
openclaw skills install spine-animation-ai
```

Then just describe what you want:

```
"Animate this sombrero character. I have the atlas and PNG.
 Give me idle, walk, and wave animations."
```

Claude will run the full pipeline and deliver a working `skeleton.json` + `preview.html`.

---

### Manual / Scripted Pipeline

**1. Auto-position parts from a reference image**

```bash
python3 scripts/position_parts.py \
  --reference assembled_character.png \
  --parts parts/ \
  --output layout.json \
  --debug debug/
```

**2. Build the Spine JSON**

```bash
python3 scripts/build_spine_json.py \
  --config layout.json \
  --output skeleton.json
```

**3. Pack a texture atlas**

```bash
python3 scripts/make_atlas.py \
  --parts parts/ \
  --output atlas/ \
  --name skeleton
```

**4. Generate a self-contained HTML preview**

```bash
python3 scripts/generate_spine_player.py \
  --skeleton skeleton.json \
  --atlas skeleton.atlas \
  --atlas-image skeleton.png \
  --output preview.html
```

Open `preview.html` in any browser — no server needed.

---

## AI Adjustment Format

When Claude analyzes a skeleton and recommends corrections, it outputs adjustments in this format:

```json
{
  "adjustments": {
    "right-arm": {
      "original_offset": { "x": -1.5, "y": 0 },
      "user_offset":     { "dx": -29.4, "dy": -84.1, "drot": 0 },
      "final_offset":    { "x": -30.9, "y": -84.1 }
    },
    "head": {
      "original_offset": { "x": 3, "y": 20.5 },
      "user_offset":     { "dx": -18.3, "dy": -2, "drot": 0 },
      "final_offset":    { "x": -15.3, "y": 18.5 }
    }
  },
  "draw_order": [
    "right-arm", "left-leg", "right-thigh", "right-leg",
    "left-thigh", "waist", "left-hand", "torso", "hat", "head"
  ]
}
```

Each entry tracks the original offset, the AI-suggested correction delta, and the applied final value.
This makes adjustments **reviewable, revertible, and composable**.

See [docs/adjustment-format.md](docs/adjustment-format.md) for full specification.

---

## Included Example: Sombrero Character

The `examples/sombrero/` directory contains a complete, working example:

| File | Description |
|------|-------------|
| `sombrero.json` | Full Spine skeleton with idle animation |
| `sombrero.atlas` | Texture atlas metadata |
| `sombrero.png` | Packed atlas spritesheet (10 parts) |
| `skeleton.json` | Skeleton-only version (no animation) |

The sombrero character has **10 body parts**, **16 bones**, and a 2-second idle loop with:
- Hip breathing bob
- Torso sway
- Head gentle rotation
- Hat follow-through
- Arm natural drift

Open [`demo/sombrero_idle.html`](demo/sombrero_idle.html) to see it in action.
Open [`demo/sombrero_editor.html`](demo/sombrero_editor.html) to interactively adjust part positions and export layout JSON.

---

## Project Structure

```
spine-animation-ai/
├── SKILL.md                        ← Claude agent skill definition
├── scripts/
│   ├── position_parts.py           ← SIFT+RANSAC auto-positioning
│   ├── build_spine_json.py         ← Spine JSON builder
│   ├── make_atlas.py               ← Texture atlas packer
│   └── generate_spine_player.py    ← HTML preview generator
├── references/
│   └── spine-json-spec.md          ← Spine format reference
├── examples/
│   └── sombrero/                   ← Full working example
│       ├── sombrero.json
│       ├── sombrero.atlas
│       └── sombrero.png
├── demo/
│   ├── sombrero_idle.html          ← Idle animation preview
│   ├── sombrero_editor.html        ← Interactive part editor
│   └── spine_animation_preview.html
└── docs/
    ├── getting-started.md
    ├── adjustment-format.md
    └── claude-prompting-guide.md
```

---

## How the Auto-Positioning Works

The `position_parts.py` script uses a two-phase algorithm:

### Phase 1: SIFT + RANSAC (primary)

1. Extract **SIFT keypoints** from each body part (alpha-masked to visible pixels)
2. Extract keypoints from the **assembled reference image**
3. Match with **FLANN** matcher + Lowe's ratio test
4. Estimate a **similarity transform** (translate + scale + rotate) via RANSAC
5. Accept if 4+ inlier matches are found; fall back otherwise

This is more robust than full homography for stylized game art with sparse features.

### Phase 2: Z-Order via Occlusion Analysis

For every overlapping pair of positioned parts:
1. Sample pixels in the overlap region
2. Compare each pixel to the reference image
3. The part whose color is closer to the reference is "on top"
4. Build a directed occlusion graph → topological sort → draw order

### Fallback: Template Matching

Parts too small for SIFT (accessories, tiny objects) fall back to
alpha-masked `TM_CCORR_NORMED` at multiple scales with background penalty scoring.

---

## Requirements

```
Python 3.9+
opencv-python >= 4.8
Pillow >= 10.0
numpy >= 1.24
```

Install dependencies:

```bash
pip install opencv-python Pillow numpy
```

---

## Claude Prompting Tips

The best prompts for this skill are specific about assets and intent:

✅ **Good:**
```
"I have separated body part PNGs for a robot character (head.png, torso.png,
 left-arm.png, right-arm.png, left-leg.png, right-leg.png) and a reference
 image showing the assembled character. Create idle and walk animations."
```

✅ **Also good:**
```
"Here's my Spine JSON. The right arm is positioned wrong — it should be
 lower and more to the left. Also add a wave animation."
```

❌ **Too vague:**
```
"Animate this character"
```

See [docs/claude-prompting-guide.md](docs/claude-prompting-guide.md) for more examples.

---

## Animation Presets

| Preset | Duration | Technique |
|--------|----------|-----------|
| `idle` | 2.0s loop | Overlapping sine waves on hip/torso/head |
| `walk` | 0.8s loop | Opposing arm-leg swing with hip bob |
| `run` | 0.5s loop | Exaggerated walk + forward lean + bounce |
| `wave` | 1.2s | Raise arm, oscillate forearm |
| `jump` | 1.0s | Anticipation squat → launch → land |
| `attack` | 0.6s | Windup → strike → follow-through |

All presets use **bezier easing** (`[0.25, 0, 0.75, 1]`) following the
[12 principles of animation](https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation).

---

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Ideas for contributions:
- New animation presets (dance, swim, fly, cast spell...)
- Support for non-humanoid rigs (animals, vehicles, abstract)
- Better occlusion detection for heavily layered characters
- Blender / Aseprite asset pipeline integration
- Spine Runtimes integration examples (Unity, Godot, Phaser)

---

## License

[MIT](LICENSE) — free for personal and commercial use.

---

## Credits

Built with:
- [Spine by Esoteric Software](http://esotericsoftware.com) — the industry-standard 2D animation tool
- [OpenCV](https://opencv.org) — SIFT feature detection and RANSAC
- [Claude AI](https://anthropic.com/claude) — the brain
- [OpenClaw](https://openclaw.ai) — the agent runtime

---

<p align="center">Made with ✨ by the community · Star if useful!</p>
