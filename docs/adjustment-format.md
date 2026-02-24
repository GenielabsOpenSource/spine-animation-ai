# Adjustment Format Reference

When Claude analyzes a Spine skeleton and suggests corrections, it outputs an **adjustment JSON**
that documents every change made. This format is designed to be:

- **Readable** — humans can inspect and understand each change
- **Reversible** — you can revert any individual adjustment
- **Composable** — multiple adjustment rounds can be stacked
- **Auditable** — you know exactly what the AI changed and why

---

## Top-Level Structure

```json
{
  "adjustments": { ... },
  "draw_order": [ ... ]
}
```

---

## `adjustments` Object

Each key is a **part name** (matching the slot name in the Spine JSON).

```json
"part-name": {
  "original_offset": { "x": float, "y": float },
  "user_offset":     { "dx": float, "dy": float, "drot": float },
  "final_offset":    { "x": float, "y": float }
}
```

### Fields

| Field | Description |
|-------|-------------|
| `original_offset.x` | The attachment's x offset in the original Spine JSON |
| `original_offset.y` | The attachment's y offset in the original Spine JSON |
| `user_offset.dx` | Horizontal correction to apply (positive = right) |
| `user_offset.dy` | Vertical correction to apply (positive = up in Spine coords) |
| `user_offset.drot` | Rotation correction in degrees (positive = counter-clockwise) |
| `final_offset.x` | `original_offset.x + user_offset.dx` |
| `final_offset.y` | `original_offset.y + user_offset.dy` |

### Zero adjustment

Parts that don't need correction still appear with zero deltas:

```json
"torso": {
  "original_offset": { "x": 25.5, "y": 105 },
  "user_offset":     { "dx": 0, "dy": 0, "drot": 0 },
  "final_offset":    { "x": 25.5, "y": 105 }
}
```

This is intentional — it confirms the part was reviewed, not skipped.

---

## `draw_order` Array

A list of part names ordered **back-to-front** (index 0 = furthest back, last = frontmost).

```json
"draw_order": [
  "right-arm",
  "left-leg",
  "right-thigh",
  "right-leg",
  "left-thigh",
  "waist",
  "left-hand",
  "torso",
  "hat",
  "head"
]
```

This maps to the **slot order** in the Spine JSON — earlier slots are drawn behind later ones.

---

## Applying Adjustments

The adjustments are applied directly to the Spine JSON's `skins` section:

```json
"skins": [{
  "name": "default",
  "attachments": {
    "right-arm": {
      "right-arm": {
        "x": -30.9,   ← final_offset.x
        "y": -84.1,   ← final_offset.y
        "width": 97,
        "height": 240,
        ...
      }
    }
  }
}]
```

You can apply them programmatically or ask Claude:

```
Here's adjustments.json [attach] and skeleton.json [attach].
Apply the adjustments and give me the corrected skeleton.json.
```

---

## Iteration Workflow

Round 1: Auto-position → adjustments-v1.json  
Round 2: Review in editor → adjustments-v2.json (new deltas, from v1 final as origin)  
Round 3: Fine-tune → adjustments-v3.json  

Each round creates a new adjustments file. The chain of files is your full edit history.

---

## Coordinate System

Spine uses **Y-up** coordinates (unlike HTML canvas which is Y-down):
- Positive X = right
- Positive Y = up
- Positive rotation = counter-clockwise

When adjusting offsets, keep this in mind. If something looks too high on screen, use a **negative dy**.
