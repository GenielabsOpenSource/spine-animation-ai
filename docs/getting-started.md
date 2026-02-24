# Getting Started with Spine Animation AI

This guide walks you through your first animation from scratch.

## Prerequisites

- Python 3.9+
- `pip install opencv-python Pillow numpy`
- Claude (via OpenClaw, Claude.ai, or API) with the `SKILL.md` loaded

---

## Option A: I have separated body part PNGs + a reference image (fastest path)

This is the best-case scenario. The SIFT algorithm will auto-position everything.

**Step 1: Organize your files**

```
my-character/
├── head.png
├── torso.png
├── left-upper-arm.png
├── left-lower-arm.png
├── right-upper-arm.png
├── right-lower-arm.png
├── left-thigh.png
├── left-leg.png
├── right-thigh.png
├── right-leg.png
└── reference.png   ← the assembled character
```

**Step 2: Auto-position**

```bash
python3 scripts/position_parts.py \
  --reference my-character/reference.png \
  --parts my-character/ \
  --output layout.json \
  --debug debug/
```

Check `debug/comparison.png` — it shows the reference vs your auto-assembled composite.
If parts look misaligned, adjust `--ratio` (try 0.75 for more aggressive matching)
or manually edit `layout.json` offsets.

**Step 3: Ask Claude**

```
Here's my layout.json [attach]. The reference image is reference.png [attach].
Build a Spine skeleton with idle and walk animations.
Output skeleton.json.
```

---

## Option B: I have an atlas + spritesheet

**Step 1: Point Claude at your files**

```
I have skeleton.atlas [attach] and skeleton.png [attach].
Build a Spine skeleton with idle animation.
Use the region names from the atlas as bone names.
```

Claude will parse the atlas, set up the skeleton, and output `skeleton.json`.

**Step 2: Generate preview**

```bash
python3 scripts/generate_spine_player.py \
  --skeleton skeleton.json \
  --atlas skeleton.atlas \
  --atlas-image skeleton.png \
  --output preview.html
```

Open `preview.html` in Chrome/Firefox.

---

## Option C: I have an existing Spine JSON and want to fix it

```
Here's my skeleton.json [attach]. The right arm is positioned too high.
Adjust it down by about 80 units and give me the corrected JSON.
Also add a wave animation.
```

Claude will output a corrected JSON and an `adjustments.json` showing exactly what changed.

---

## Understanding the Output Files

| File | What it is |
|------|-----------|
| `skeleton.json` | Spine-compatible skeleton. Load in Spine editor, Unity, Godot, etc. |
| `skeleton.atlas` | Atlas metadata. Required alongside skeleton.json for any Spine runtime. |
| `skeleton.png` | The packed texture. Referenced by the atlas. |
| `adjustments.json` | Record of what the AI changed. Useful for review and iteration. |
| `preview.html` | Self-contained HTML preview. Open in any browser — no server needed. |

---

## Troubleshooting

**position_parts.py fails with "not enough matches"**
- Try lowering `--ratio` to 0.75 or even 0.70
- Make sure your body-part PNGs have alpha transparency (transparent background)
- Parts with very little texture (solid flat colors) may not have enough SIFT keypoints — use `--min-matches 2`

**The composite looks mostly right but one part is wrong**
- Manually edit `layout.json` — find the part's `x`/`y` offset and adjust
- Or tell Claude: "The left arm is off by about 30px to the right"

**preview.html shows a white screen**
- Make sure the atlas PNG path in the atlas file matches the actual filename
- Check the browser console for errors

**Spine editor says JSON is invalid**
- Make sure you're using Spine 4.2 (this skill targets 4.2 format)
- Check `references/spine-json-spec.md` for format details
