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
  skeleton rigging, animation keyframing, Spine JSON export, and interactive HTML5 preview.
---

# Spine Animation Skill

Turn pre-existing 2D character assets into fully animated, interactive Spine animations.

## Step 0: Set Up Scripts

This skill includes Python scripts that do the heavy lifting. Claude MUST write them to disk
before use. Each script is embedded below — Claude should save them to `/home/claude/spine-scripts/`
at the start of every session.

```bash
mkdir -p /home/claude/spine-scripts
pip install opencv-python Pillow numpy --break-system-packages -q
```

### Embedded Scripts

The following scripts are auto-injected from the repository's `scripts/` directory.
**Claude: read these carefully, then write each one to `/home/claude/spine-scripts/`**
before running the pipeline.

<!-- EMBED:scripts/position_parts.py -->

<!-- EMBED:scripts/build_spine_json.py -->

<!-- EMBED:scripts/make_atlas.py -->

<!-- EMBED:scripts/generate_spine_player.py -->

After writing the scripts, verify:
```bash
ls /home/claude/spine-scripts/
# Should show: position_parts.py  build_spine_json.py  make_atlas.py  generate_spine_player.py
```

---

## What You Need From the User

At minimum, one of these asset sets:

| Asset Set | What to Expect |
|-----------|---------------|
| **Separated body part PNGs** | Individual transparent PNGs for head, torso, arms, legs, etc. |
| **Separated PNGs + reference image** | Parts + an assembled character image → enables auto-positioning |
| **Texture atlas + atlas PNG** | A `.atlas` file + spritesheet `.png` (standard Spine export) |
| **Full character image** | A single image — Claude will help define part regions |
| **Existing Spine JSON** | An existing `.json` to add/modify animations |

The user should also say what animations they want (idle, walk, run, attack, wave, jump, etc.)

## Full Pipeline

```
User Assets → Analyze Parts → [Auto-Position if reference] → Build Skeleton → Animate → Preview
```

### Step 1: Analyze the Assets

Look at the uploaded files:

1. **If separated PNGs**: Use Claude's vision to identify each part (head, torso, left-arm, etc.)
   and note their dimensions. Determine the bone hierarchy from the part names and visual layout.

2. **If atlas + spritesheet**: Parse the `.atlas` file to extract region names, positions, and sizes.

3. **If full image only**: Use Claude's vision to identify body parts, then crop into separate PNGs.

4. **If existing Spine JSON**: Parse it, understand the skeleton, and add new animations.

5. **If separated PNGs + assembled reference image**: Run `position_parts.py` for auto-layout.

### Step 1.5: Auto-Position Parts (when reference image is available)

If the user provides both separated body-part PNGs **and** an assembled reference image:

```bash
python3 /home/claude/spine-scripts/position_parts.py \
  --reference assembled_character.png \
  --parts parts_folder/ \
  --output layout.json \
  --debug debug/ \
  --min-matches 4 \
  --ratio 0.80
```

**Algorithm — SIFT + RANSAC similarity transform:**

1. Extracts SIFT keypoints from each part (alpha-masked) and the reference image
2. Matches via FLANN with Lowe's ratio test (default 0.80)
3. Estimates similarity transform (4 DOF: translate + scale + rotation) via
   `cv2.estimateAffinePartial2D` — more robust than full homography for game art
4. SIFT tuning for stylized art: `contrastThreshold=0.02, edgeThreshold=20`
5. Template matching fallback for tiny/featureless parts, using SIFT-derived median scale
6. Z-order via pairwise occlusion voting

After running, **check `debug/comparison.png`** to verify positioning accuracy.
Per-part SIFT match visualizations: `debug/sift_<partname>.jpg`.

**Limitations:** Heavily occluded parts (e.g., thighs behind a belt) may need manual correction.
Compare composite vs reference with Claude's vision and adjust layout JSON offsets.

### Step 2: Build Bone Hierarchy

Standard humanoid skeleton:

```
root
└── hip
    ├── torso
    │   ├── neck → head → hat/hair
    │   ├── left-shoulder → left-upper-arm → left-lower-arm → left-hand
    │   └── right-shoulder → right-upper-arm → right-lower-arm → right-hand
    ├── left-upper-leg → left-lower-leg → left-foot
    └── right-upper-leg → right-lower-leg → right-foot
```

**Coordinate system:** Spine uses Y-up, origin at character's feet center.
- `spine_x = pixel_x - center_x`
- `spine_y = bottom_y - pixel_y`

**Bone positions** are RELATIVE to parent:
- `relative_pos = child_world_pos - parent_world_pos`

**Attachment offsets** are relative to their bone:
- `att_offset = image_center_world_pos - bone_world_pos`

### Step 3: Build Spine JSON

Spine JSON v4.2 structure:

```json
{
  "skeleton": { "hash": "...", "spine": "4.2", "width": 500, "height": 950 },
  "bones": [
    { "name": "root", "x": 0, "y": 0, "length": 0 },
    { "name": "hip", "parent": "root", "x": 0, "y": 410, "length": 30 }
  ],
  "slots": [
    { "name": "back-arm", "bone": "left-arm-bone", "attachment": "back-arm" }
  ],
  "skins": [{
    "name": "default",
    "attachments": {
      "slot-name": {
        "attachment-name": { "x": 5, "y": -10, "width": 100, "height": 200 }
      }
    }
  }],
  "animations": {
    "idle": {
      "bones": {
        "hip": {
          "translate": [
            { "time": 0, "x": 0, "y": 0, "curve": [0.25, 0, 0.75, 1] },
            { "time": 1.0, "x": 0, "y": 3 },
            { "time": 2.0, "x": 0, "y": 0 }
          ]
        }
      }
    }
  }
}
```

**Slots** define draw order — first slot is drawn first (back), last is front.

### Step 4: Create Animations

**Keyframe format:**
```json
{ "time": 0.0, "value": 0, "curve": [0.25, 0, 0.75, 1] }
```

The `curve` is a cubic bezier `[cx1, cy1, cx2, cy2]`. Use `[0.25, 0, 0.75, 1]` for ease-in-out.

**Animation presets:**

| Preset | Duration | Key Technique |
|--------|----------|---------------|
| idle | 2.0s loop | Hip ±3px translate, torso ±1° rotate, head ±1.5° counter-sway |
| walk | 0.8s loop | Opposing arm-leg swing, hip ±5px bob, torso ±3° lean |
| run | 0.5s loop | Exaggerated walk + 5° forward lean + ±8px bounce |
| wave | 1.2s | Shoulder -45°, forearm oscillate ±15° |
| jump | 1.0s | Squat → launch → air → land (4 phases) |
| attack | 0.6s | Windup → strike → follow-through (3 phases) |

**Key principles:**
- Offset timing between related bones (head peaks 0.1s after torso = follow-through)
- Larger movements on larger bones (hip > torso > head)
- All loops must return to starting values

### Step 5: Build Atlas

```bash
python3 /home/claude/spine-scripts/make_atlas.py \
  --parts parts_folder/ \
  --output . \
  --name character_name
```

Outputs: `character_name.png` (spritesheet) + `character_name.atlas` (metadata).

### Step 6: Generate Preview

For a **self-contained HTML Canvas preview** (recommended):
Build the HTML directly in Python with base64-embedded images, bone system, bezier interpolation,
and animation loop. No external dependencies needed.

For an **official Spine Web Player preview**:
```bash
python3 /home/claude/spine-scripts/generate_spine_player.py \
  --skeleton character.json \
  --atlas character.atlas \
  --atlas-image character.png \
  --output preview.html
```

### Step 7: Interactive Editor (Optional)

Build an HTML editor that allows the user to fine-tune part positions:
- Click to select parts (purple dashed border + glow)
- Drag to reposition in real-time
- Arrow keys for 1px nudge (Shift for 10px)
- Side panel with numeric X/Y/rotation inputs
- Draggable z-order list
- Export button producing layout corrections JSON:

```json
{
  "adjustments": {
    "part-name": {
      "original_offset": { "x": 0, "y": 0 },
      "user_offset": { "dx": 5.2, "dy": -3.1, "drot": 0 },
      "final_offset": { "x": 5.2, "y": -3.1 }
    }
  },
  "draw_order": ["back-part", "...", "front-part"]
}
```

This JSON can be fed back to Claude to apply corrections to the skeleton.
