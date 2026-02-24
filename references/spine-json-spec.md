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
      { "time": 0, "angle": 0 },
      { "time": 0.75, "angle": 2, "curve": [0.25, 0, 0.75, 1] }
    ]
  }
}}}
```

### Curve types
- Omitted = linear | `"stepped"` = hold | `[cx1, cy1, cx2, cy2]` = bezier
- Common: ease `[0.25, 0, 0.75, 1]`, in `[0.42, 0, 1, 1]`, out `[0, 0, 0.58, 1]`

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
