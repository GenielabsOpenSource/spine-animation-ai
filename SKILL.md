---
name: spine-animation
description: >
  Create Spine 2D skeletal animations from pre-existing character assets (separated body-part PNGs,
  atlas spritesheet, or a full character image). Use this skill whenever the user wants to animate
  a 2D character, create Spine JSON from existing art assets, rig a character with bones, build
  walk/idle/run/attack animations, produce an interactive Spine Web Player preview, or generate
  Spine-compatible export files (.json + .atlas + .png). Also trigger when the user mentions
  "Spine animation", "2D rigging", "skeletal animation", "bone animation", "cutout animation",
  "animate this character", "make this walk", "create walk cycle", or uploads separated character
  body parts and wants them animated. This skill handles the full pipeline: asset analysis,
  skeleton rigging, animation keyframing, Spine JSON export, and interactive HTML5 preview
  using the official Spine Web Player. Even if the user just uploads some PNGs and says
  "animate these", use this skill.
---

# Spine Animation Skill

Turn pre-existing 2D character assets into fully animated, interactive Spine animations.
This skill starts from **assets you already have** — separated body part PNGs, a texture atlas,
or both — and produces a complete Spine skeleton with animations and an interactive web preview.

## What You Need From the User

At minimum, one of these asset sets:

| Asset Set | What to Expect |
|-----------|---------------|
| **Separated body part PNGs** | Individual transparent PNGs for head, torso, arms, legs, etc. |
| **Texture atlas + atlas PNG** | A `.atlas` file + spritesheet `.png` (standard Spine export) |
| **Full character image** | A single image — Claude will help define part regions |
| **Existing Spine JSON** | An existing `.json` to add/modify animations |

The user should also say what animations they want (idle, walk, run, attack, wave, jump, etc.)

## Pipeline

```
User Assets → Analyze Parts → [Auto-Position if reference available] → Build Skeleton → Animate → Preview
```

### Step 1: Analyze the Assets

Look at the uploaded files:

1. **If separated PNGs**: Use Claude's vision to identify each part (head, torso, left-arm, etc.)
   and note their dimensions. Determine the bone hierarchy from the part names and visual layout.

2. **If atlas + spritesheet**: Parse the `.atlas` file to extract region names, positions, and sizes.
   Map region names to body parts.

3. **If full image only**: Use Claude's vision to identify body parts, then use PIL to crop them
   into separate part PNGs and generate an atlas. Use `scripts/make_atlas.py` for this.

4. **If existing Spine JSON**: Parse it, understand the skeleton structure, and add new animations.

5. **If separated PNGs + assembled reference image**: Run `scripts/position_parts.py` to auto-detect
   positions and z-order. This is the fastest path — it handles layout automatically.

### Step 1.5: Auto-Position Parts (when reference image is available)

If the user provides both the separated body-part PNGs **and** an assembled reference image showing
the character fully put together, use `position_parts.py` to automatically determine part placement:

```bash
python3 scripts/position_parts.py \
  --reference assembled_character.png \
  --parts parts_folder/ \
  --output layout.json \
  --debug debug/ \
  --min-matches 4 \
  --ratio 0.80
```

The algorithm uses two phases:

1. **SIFT + RANSAC similarity transform** — the primary method. Extracts SIFT keypoints from each
   part (alpha-masked to opaque pixels) and the reference image. Matches via FLANN with Lowe's ratio
   test (default 0.80), then estimates a similarity transform (4 DOF: translate + uniform scale +
   rotation) via `cv2.estimateAffinePartial2D` with RANSAC. This is more robust than full homography
   (8 DOF) for game art with sparse features. Works well for parts with sufficient texture — typically
   needs 4+ inlier matches. Falls back to template matching for tiny/featureless parts.

   Key tuning: SIFT uses `contrastThreshold=0.02, edgeThreshold=20` to detect more features on
   stylized game art. The `--ratio` flag controls Lowe's test strictness (lower = stricter).

2. **Z-order via pixel occlusion analysis** — for every overlapping pair of positioned parts, samples
   pixels in the overlap region and compares to the reference. The part whose pixel color is closer
   to the reference is "on top". Builds a directed occlusion graph and topologically sorts for draw order.

**Template matching fallback** — parts too small for SIFT (bobbers, tiny accessories) fall back to
alpha-masked `TM_CCORR_NORMED` at multiple scales with background penalty scoring.

After running, **check `debug/comparison.png`** — reference vs auto-composite side by side.
Per-part SIFT match visualizations are saved as `debug/sift_<partname>.jpg`.
Use Claude's vision to spot misalignments and adjust offsets before proceeding to skeleton building.

**Limitations:** heavily occluded parts (e.g. waist hidden behind torso, thighs behind belt) may
have too few visible pixels for reliable matching. When the debug comparison shows misplaced parts,
manually adjust those offsets or provide an SVG layout file for pixel-perfect positioning.

### Step 2: Build the Bone Hierarchy

Define bones based on the character anatomy. Read `references/spine-json-spec.md` for the complete
format. Standard humanoid hierarchy:

```
root
└── hip                    ← center of mass, pivot for whole body
    ├── torso              ← spine/chest
    │   ├── neck → head
    │   ├── left-upper-arm → left-lower-arm → left-hand
    │   └── right-upper-arm → right-lower-arm → right-hand
    ├── left-upper-leg → left-lower-leg → left-foot
    └── right-upper-leg → right-lower-leg → right-foot
```

Key rigging rules:
- **Bone positions** are relative to parent, in Spine coordinates (Y up)
- **Pivot points** (bone origins) should be at joint locations: shoulder, elbow, hip, knee
- **Bone length** is the distance to the child joint
- **Slots** define draw order: legs behind torso, torso behind arms, arms behind head

For non-humanoids, adapt the hierarchy to the creature's actual anatomy.

### Step 3: Create Animations

Use `scripts/build_spine_json.py` to generate the complete Spine JSON. The script includes
preset animation generators for common animations. Each uses proper animation principles:

**Animation presets available:**

| Preset | Description | Key Technique |
|--------|-------------|--------------|
| `idle` | Subtle breathing/sway, 1.5s loop | Overlapping sine waves on torso/head |
| `walk` | Walk cycle, 0.8s loop | Opposing arm-leg swing, hip bob |
| `run` | Run cycle, 0.5s loop | Exaggerated walk with lean + bounce |
| `wave` | Wave greeting, 1.2s | Raise arm, oscillate forearm |
| `jump` | Jump up & land, 1.0s | Anticipation squat → launch → land |
| `attack` | Melee swing, 0.6s | Windup → strike → follow-through |

Each preset uses **bezier curves** (not linear interpolation) for natural easing and follows
the 12 principles of animation (anticipation, follow-through, overlapping action, etc.)

You can also define **custom animations** by writing keyframe data directly in the config.

### Step 4: Generate the Interactive Preview

Use `scripts/generate_spine_player.py` to create a self-contained HTML file that uses the
**official Spine Web Player** (`@esotericsoftware/spine-player`) loaded from UNPKG CDN.

The preview embeds:
- Skeleton JSON as base64 data URI (via `rawDataURIs`)
- Atlas text as base64 data URI
- Atlas PNG as base64 data URI
- All body part images packed into the atlas

The Spine Web Player provides:
- Accurate rendering with WebGL (meshes, blending, clipping all work)
- Play/pause, timeline scrubbing, speed control
- Animation selector dropdown
- Skin selector (if skins are defined)
- Debug overlays (bones, regions, meshes, bounds)
- Fullscreen mode
- Bone dragging for interactive posing

### Step 5: Export Files

The final deliverables are:

| File | Purpose |
|------|---------|
| `skeleton.json` | Complete Spine JSON — loadable by any Spine Runtime |
| `skeleton.atlas` | Texture atlas metadata |
| `skeleton.png` | Packed texture atlas image |
| `preview.html` | Self-contained HTML preview with Spine Web Player |

The JSON + atlas + PNG triplet works with Unity, Godot, Unreal, Phaser, PixiJS, Three.js,
and all other Spine Runtimes.

## Script Reference

### `scripts/build_spine_json.py`
Generates the complete Spine JSON with skeleton, bones, slots, skins, and animations.

```bash
python3 scripts/build_spine_json.py --config config.json --output skeleton.json
```

Config format documented in the script header. Supports all animation presets + custom keyframes.

### `scripts/make_atlas.py`
Packs individual part PNGs into a Spine-compatible texture atlas.

```bash
python3 scripts/make_atlas.py --parts parts/ --output atlas/ --name skeleton
```

Produces `skeleton.png` (spritesheet) and `skeleton.atlas` (metadata).

### `scripts/generate_spine_player.py`
Bundles everything into a self-contained HTML preview using the official Spine Web Player.

```bash
python3 scripts/generate_spine_player.py \
  --skeleton skeleton.json \
  --atlas skeleton.atlas \
  --atlas-image skeleton.png \
  --output preview.html
```

## Practical Tips

1. **Part naming matters**: Name PNGs clearly (head.png, torso.png, left-upper-arm.png).
   The scripts map names to the skeleton automatically.

2. **Pivot points are everything**: If animation looks wrong, adjust bone positions. The joint
   (shoulder, elbow, hip, knee) should be exactly at the bone's origin.

3. **Draw order = slot order**: Lower-indexed slots are drawn behind higher-indexed ones.
   Put legs first, then torso, then arms, then head for a front-facing character.

4. **Start with idle**: Always test with an idle animation first to verify the rig before
   adding complex animations. If idle looks good, the skeleton is correct.

5. **Bezier curves not linear**: Always use `[0.25, 0, 0.75, 1]` or similar easing curves
   for organic characters. Linear interpolation looks robotic.

6. **Atlas vs. individual images**: The Spine Web Player needs an atlas. If the user provides
   individual PNGs, pack them first with `make_atlas.py`.

## Spine JSON Format Quick Reference

See `references/spine-json-spec.md` for the full format specification. Quick cheat sheet:

```json
{
  "skeleton": { "spine": "4.2.0", "width": 400, "height": 600 },
  "bones": [
    { "name": "root" },
    { "name": "hip", "parent": "root", "y": 200 }
  ],
  "slots": [
    { "name": "torso", "bone": "torso", "attachment": "torso" }
  ],
  "skins": [{ "name": "default", "attachments": { ... } }],
  "animations": {
    "idle": {
      "bones": {
        "torso": {
          "rotate": [
            { "time": 0, "angle": 0 },
            { "time": 0.75, "angle": 2, "curve": [0.25, 0, 0.75, 1] },
            { "time": 1.5, "angle": 0, "curve": [0.25, 0, 0.75, 1] }
          ]
        }
      }
    }
  }
}
```
