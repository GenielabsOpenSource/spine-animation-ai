# Spine JSON Format Quick Reference

Condensed reference for generating valid Spine 4.2 JSON programmatically.
Full spec: https://en.esotericsoftware.com/spine-json-format

## Top-Level Structure
```json
{ "skeleton": {}, "bones": [], "slots": [], "skins": [], "animations": {} }
```

## Skeleton
```json
"skeleton": { "hash": "abc", "spine": "4.2.0", "x": -200, "y": 0, "width": 400, "height": 600 }
```

## Bones (parent-before-child order!)
```json
"bones": [
  { "name": "root" },
  { "name": "hip", "parent": "root", "x": 0, "y": 200, "length": 30 }
]
```
Defaults: x=0, y=0, rotation=0, scaleX=1, scaleY=1, length=0

## Slots (array order = draw order, lower index = drawn behind)
```json
"slots": [{ "name": "torso", "bone": "torso", "attachment": "torso" }]
```

## Skins & Region Attachments
```json
"skins": [{ "name": "default", "attachments": {
  "slotName": { "attachmentName": { "width": 120, "height": 200, "x": 0, "y": 60 } }
}}]
```

## Animation Bone Timelines
```json
"animations": { "idle": { "bones": {
  "torso": {
    "rotate": [
      { "time": 0, "value": 0, "curve": [0.19, 0, 0.56, 2] },
      { "time": 0.75, "value": 2 }
    ],
    "translate": [
      { "time": 0, "x": 0, "y": 0, "curve": [0.19, 0, 0.56, 0, 0.19, 0, 0.56, 3] },
      { "time": 0.75, "x": 0, "y": 3 }
    ]
  }
}}}
```

Rotation values are keyed as `value`. (`angle` is the Spine 3.8 spelling; a
4.x runtime reads it as 0 and nothing turns.)

### Curve types
- Omitted = linear | `"stepped"` = hold | array of numbers = bezier
- A keyframe's curve describes the segment that **starts** at it. The last
  keyframe of a timeline therefore never carries one.

### Bezier control points
Two rules trip people up when writing 4.x JSON by hand:

**One bezier per animated property, not per keyframe.** `rotate` drives a
single property and takes 4 numbers. `translate`, `scale` and `shear` drive
two (x then y) and take 8. Supplying 4 where 8 are needed makes the runtime
read past the end of the array; the resulting NaN propagates through the bone
transforms and the skeleton stops rendering after the first frame.

**Control points are absolute, not normalized.** `cx` is a time and `cy` is a
value, in the same units as the surrounding keyframes -- not a 0..1 fraction
of the segment. To place a standard ease on a segment running from
`(t1, v1)` to `(t2, v2)`:

```
cx1 = t1 + (t2 - t1) * 0.25    cy1 = v1 + (v2 - v1) * 0
cx2 = t1 + (t2 - t1) * 0.75    cy2 = v1 + (v2 - v1) * 1
```

Normalized fractions from the 3.8-era format put the handles outside their
own segment, which makes the motion lurch rather than ease.

## Atlas File Format
```
skeleton.png
size: 512,512
format: RGBA8888
filter: Linear,Linear
repeat: none
regionName
  rotate: false
  xy: 0, 0
  size: 120, 200
  orig: 120, 200
  offset: 0, 0
  index: -1
```
