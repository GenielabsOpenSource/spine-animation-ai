# Claude Prompting Guide for Spine Animation AI

Getting great results from Claude depends on giving it the right context. This guide shows what works.

---

## The Golden Rule

**Be specific about assets, intent, and constraints.**

Claude can't see your files unless you attach them. It doesn't know your target engine
unless you say. The more context, the better the output.

---

## Prompt Templates

### Starting from scratch (separated PNGs)

```
I have separated body part PNGs for a [description] character:
- head.png, torso.png, left-upper-arm.png, right-upper-arm.png,
  left-lower-arm.png, right-lower-arm.png, left-thigh.png,
  right-thigh.png, left-leg.png, right-leg.png

[Attach all PNGs + reference assembled image if available]

Please:
1. Auto-position the parts using position_parts.py
2. Build a Spine 4.2 skeleton JSON
3. Create idle and walk animations
4. Generate a preview.html

The character should look [style notes: cartoony / realistic / cute / etc.]
```

---

### Starting from an atlas

```
I have a Spine atlas for my character [attach skeleton.atlas + skeleton.png].
The parts are: [list part names or say "use the atlas region names"].

Please:
1. Build a skeleton JSON with the standard humanoid bone hierarchy
2. Add idle, walk, and attack animations
3. The character is [description: side-scrolling game hero / top-down RPG / etc.]
```

---

### Fixing an existing skeleton

```
Here's my Spine JSON [attach skeleton.json].

Problems I can see:
- The right arm is about 80px too high
- The head seems rotated wrong (looks tilted)
- The draw order puts the left leg in front of everything — should be behind torso

Please:
1. Fix these issues
2. Output corrected skeleton.json
3. Output adjustments.json showing exactly what changed
```

---

### Adding an animation to an existing skeleton

```
Here's my skeleton.json [attach]. It already has an idle animation.

Please add:
- A wave animation (right arm waves, 1.5s, 2 cycles)
- A jump animation (full body, includes anticipation squat, 1.0s)

Keep the existing idle animation unchanged.
```

---

### Iterating after reviewing the preview

```
I reviewed preview.html and here's what needs fixing:

1. The hat floats about 20px above the head — pull it down
2. The walk cycle arms swing too wide — reduce the arm rotation range by 50%
3. The idle animation speed feels too fast — slow it down by 20%

Here's the current skeleton.json [attach].
Output a corrected skeleton.json and updated adjustments.json.
```

---

## What Claude Needs to Know

| Information | Why it matters | Example |
|-------------|---------------|---------|
| Part names | Maps to skeleton slots | "head, torso, left-arm, right-arm, waist, left-leg, right-leg" |
| Target engine | May affect JSON format | "Unity with Spine Unity runtime", "Phaser 3", "just preview" |
| Canvas/viewport size | Sets skeleton bounds | "600x900 pixel canvas" |
| Animation style | Affects easing and timing | "snappy and energetic", "slow and floaty" |
| Reference image | Enables auto-positioning | [attach assembled_character.png] |
| Current issues | Tells Claude what to fix | "the arm is too high by ~50px" |

---

## Common Mistakes

### ❌ Asking without attaching files

```
Can you fix my skeleton? The arm is wrong.
```

Claude can't see `skeleton.json` unless you attach it.

---

### ❌ Being too vague about the problem

```
The animation looks bad.
```

Better:

```
The walk cycle arms don't swing — they're stuck at their rest position.
Also the hip doesn't bob. Here's the JSON [attach].
```

---

### ❌ Not specifying the Spine version

Spine 4.x and 3.x have different JSON formats. Default assumption is **4.2** unless you say otherwise.

---

## Tips for Non-Standard Characters

**Quadrupeds (animals, horses, dogs):**
```
This is a dog character. The bone hierarchy should be:
root → spine → neck → head
              → tail
      → front-left-leg (shoulder → lower-leg → paw)
      → front-right-leg (same)
      → back-left-leg (hip → lower-leg → paw)
      → back-right-leg (same)
```

**Vehicles:**
```
This is a car. I want it to have:
- A wheel-spin animation (all 4 wheels rotating)
- A suspension-bounce animation (body rises/falls slightly)
No humanoid hierarchy — treat each wheel as an independent bone.
```

**Abstract/custom:**
```
This is a jellyfish. It has a dome body and 8 tentacles.
Create an idle animation where the dome pulses and tentacles trail.
```

---

## Working with the Interactive Editor

The `demo/sombrero_editor.html` is a useful reference for understanding what adjustments
look like in practice:

1. Open it in a browser
2. Click a body part to select it
3. Drag to reposition, or use the numeric inputs for precision
4. Click "Export Layout JSON" when satisfied
5. Paste that JSON to Claude: *"Apply these offsets to my skeleton.json"*

This gives you a visual, tactile way to figure out the right numbers before asking Claude to apply them.
