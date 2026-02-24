#!/usr/bin/env python3
"""
build_spine_json.py — Generate a complete Spine-compatible JSON skeleton with animations.

Accepts a config describing bones, slots, attachments, and desired animations,
then outputs a valid Spine 4.2 JSON file.

Usage:
    python3 build_spine_json.py --config config.json --output skeleton.json

Config JSON format:
{
  "skeleton": {
    "name": "my-character",
    "width": 400,
    "height": 600
  },
  "bones": [
    { "name": "root" },
    { "name": "hip", "parent": "root", "x": 0, "y": 200, "length": 30 },
    { "name": "torso", "parent": "hip", "length": 120 },
    ...
  ],
  "slots": [
    { "name": "torso", "bone": "torso", "attachment": "torso" },
    ...
  ],
  "attachments": {
    "torso": { "width": 120, "height": 200, "x": 0, "y": 60 },
    ...
  },
  "animations": ["idle", "walk", "wave", "jump", "run", "attack"],
  "custom_animations": {
    "my-custom": { "bones": { "head": { "rotate": [...] } } }
  }
}
"""

import argparse
import json
import hashlib
import sys

# ─── Bezier curve presets ────────────────────────────────────────────────────
EASE = [0.25, 0, 0.75, 1]        # Standard ease in-out (most common)
EASE_IN = [0.42, 0, 1, 1]        # Accelerate from rest
EASE_OUT = [0, 0, 0.58, 1]       # Decelerate to rest
EASE_BOUNCE = [0.34, 1.56, 0.64, 1]  # Slight overshoot
EASE_FAST = [0.4, 0, 0.2, 1]     # Quick but smooth

def _has(bone_names, *names):
    """Check if any of the given bone names exist."""
    return any(n in bone_names for n in names)

def _kf(time, angle=None, x=None, y=None, curve=EASE):
    """Build a keyframe dict, omitting None values."""
    kf = {"time": round(time, 4)}
    if angle is not None:
        kf["angle"] = round(angle, 2)
    if x is not None:
        kf["x"] = round(x, 2)
    if y is not None:
        kf["y"] = round(y, 2)
    if curve:
        kf["curve"] = curve
    return kf


# ─── Animation Generators ───────────────────────────────────────────────────

def gen_idle(B):
    """Idle breathing/sway. Subtle, loopable. ~1.6s"""
    bones = {}
    D = 1.6

    for name, angle_amp, phase in [
        ("torso", 1.5, 0.5), ("neck", 1.0, 0.55), ("head", -2.0, 0.6)
    ]:
        if name in B:
            bones[name] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(D * phase, angle_amp),
                _kf(D, 0),
            ]}

    # Subtle torso lift
    if "torso" in B:
        bones.setdefault("torso", {})["translate"] = [
            _kf(0, x=0, y=0, curve=None),
            _kf(D * 0.5, x=0, y=1.5),
            _kf(D, x=0, y=0),
        ]

    # Gentle arm sway
    for side in ["left", "right"]:
        s = 1 if side == "left" else -1
        for part, amp, ph in [
            (f"{side}-upper-arm", s * 1.5, 0.5),
            (f"{side}-lower-arm", s * 1.0, 0.55),
        ]:
            if part in B:
                bones[part] = {"rotate": [
                    _kf(0, 0, curve=None), _kf(D * ph, amp), _kf(D, 0),
                ]}

    return {"bones": bones} if bones else {}


def gen_walk(B):
    """Walk cycle. Opposing arm-leg motion, hip bob. ~0.8s"""
    bones = {}
    D = 0.8
    Q = D / 4  # quarter

    if "hip" in B:
        bones["hip"] = {
            "translate": [
                _kf(0, x=0, y=0, curve=None),
                _kf(Q, x=0, y=3), _kf(Q*2, x=0, y=0),
                _kf(Q*3, x=0, y=3), _kf(D, x=0, y=0),
            ],
            "rotate": [
                _kf(0, 0, curve=None),
                _kf(Q, -2), _kf(Q*2, 0), _kf(Q*3, 2), _kf(D, 0),
            ],
        }

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(Q, 3), _kf(Q*2, 0), _kf(Q*3, -3), _kf(D, 0),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(Q, -1.5), _kf(Q*2, 0), _kf(Q*3, 1.5), _kf(D, 0),
        ]}

    # Legs: left forward at t=0, right forward at t=D/2
    for side, phase_shift in [("left", 0), ("right", 0.5)]:
        p = phase_shift * D
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        foot = f"{side}-foot"

        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -25 if phase_shift == 0 else 25, curve=None),
                _kf(Q, 0), _kf(Q*2, 25 if phase_shift == 0 else -25),
                _kf(Q*3, 0), _kf(D, -25 if phase_shift == 0 else 25),
            ]}
        if lower in B:
            # Back leg straight, front leg bent
            bones[lower] = {"rotate": [
                _kf(0, 5 if phase_shift == 0 else 35, curve=None),
                _kf(Q, 35), _kf(Q*2, 35 if phase_shift == 0 else 5),
                _kf(Q*3, 5), _kf(D, 5 if phase_shift == 0 else 35),
            ]}

    # Arms: oppose legs
    for side, phase_shift in [("left", 0.5), ("right", 0)]:
        upper = f"{side}-upper-arm"
        lower = f"{side}-lower-arm"

        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -20 if phase_shift == 0 else 20, curve=None),
                _kf(Q, 0), _kf(Q*2, 20 if phase_shift == 0 else -20),
                _kf(Q*3, 0), _kf(D, -20 if phase_shift == 0 else 20),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, -10 if phase_shift == 0 else -30, curve=None),
                _kf(Q, -20), _kf(Q*2, -30 if phase_shift == 0 else -10),
                _kf(Q*3, -20), _kf(D, -10 if phase_shift == 0 else -30),
            ]}

    return {"bones": bones} if bones else {}


def gen_run(B):
    """Run cycle. Exaggerated walk, forward lean, bigger bounce. ~0.5s"""
    bones = {}
    D = 0.5
    Q = D / 4

    if "hip" in B:
        bones["hip"] = {
            "translate": [
                _kf(0, x=0, y=0, curve=None),
                _kf(Q, x=0, y=6), _kf(Q*2, x=0, y=-2),
                _kf(Q*3, x=0, y=6), _kf(D, x=0, y=0),
            ],
            "rotate": [
                _kf(0, 0, curve=None),
                _kf(Q, -3), _kf(Q*2, 0), _kf(Q*3, 3), _kf(D, 0),
            ],
        }

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 8, curve=None),  # Constant forward lean
            _kf(Q, 12), _kf(Q*2, 8), _kf(Q*3, 12), _kf(D, 8),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, -6, curve=None),  # Compensate for torso lean
            _kf(Q, -8), _kf(Q*2, -6), _kf(Q*3, -8), _kf(D, -6),
        ]}

    for side, ph in [("left", 0), ("right", 0.5)]:
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -35 if ph == 0 else 40, curve=None),
                _kf(Q, 0), _kf(Q*2, 40 if ph == 0 else -35),
                _kf(Q*3, 0), _kf(D, -35 if ph == 0 else 40),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, 10 if ph == 0 else 50, curve=None),
                _kf(Q, 50), _kf(Q*2, 50 if ph == 0 else 10),
                _kf(Q*3, 10), _kf(D, 10 if ph == 0 else 50),
            ]}

    for side, ph in [("left", 0.5), ("right", 0)]:
        upper = f"{side}-upper-arm"
        lower = f"{side}-lower-arm"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, -30 if ph == 0 else 30, curve=None),
                _kf(Q, 0), _kf(Q*2, 30 if ph == 0 else -30),
                _kf(Q*3, 0), _kf(D, -30 if ph == 0 else 30),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, -20 if ph == 0 else -50, curve=None),
                _kf(Q, -35), _kf(Q*2, -50 if ph == 0 else -20),
                _kf(Q*3, -35), _kf(D, -20 if ph == 0 else -50),
            ]}

    return {"bones": bones} if bones else {}


def gen_wave(B):
    """Waving greeting. Raise right arm, oscillate forearm. ~1.2s"""
    bones = {}
    D = 1.2

    if "right-upper-arm" in B:
        bones["right-upper-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -130, curve=EASE_OUT),
            _kf(D - 0.2, -130, curve=None),
            _kf(D, 0, curve=EASE_IN),
        ]}
    if "right-lower-arm" in B:
        bones["right-lower-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -30, curve=EASE_OUT),
            _kf(0.4, 20), _kf(0.6, -20), _kf(0.8, 20), _kf(1.0, -20),
            _kf(D, 0, curve=EASE_IN),
        ]}
    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.2, -3), _kf(D - 0.2, -3, curve=None), _kf(D, 0),
        ]}
    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.3, 5), _kf(D - 0.2, 5, curve=None), _kf(D, 0),
        ]}

    return {"bones": bones} if bones else {}


def gen_jump(B):
    """Jump: anticipation squat → launch → air → land → settle. ~1.0s"""
    bones = {}
    D = 1.0

    if "hip" in B:
        bones["hip"] = {"translate": [
            _kf(0, x=0, y=0, curve=None),
            _kf(0.15, x=0, y=-20, curve=EASE_IN),     # squat
            _kf(0.35, x=0, y=70, curve=EASE_OUT),      # launch
            _kf(0.55, x=0, y=65, curve=None),           # float
            _kf(0.80, x=0, y=-10, curve=EASE_IN),       # land impact
            _kf(D, x=0, y=0, curve=EASE_OUT),            # settle
        ]}

    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.15, 8, curve=EASE_IN),     # lean forward in squat
            _kf(0.35, -5, curve=EASE_OUT),    # extend in air
            _kf(0.80, 5, curve=EASE_IN),      # absorb landing
            _kf(D, 0, curve=EASE_OUT),
        ]}

    if "head" in B:
        bones["head"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.15, 5), _kf(0.35, -8), _kf(0.80, 3), _kf(D, 0),
        ]}

    for side in ["left", "right"]:
        s = 1 if side == "left" else -1
        arm = f"{side}-upper-arm"
        if arm in B:
            bones[arm] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, s*10), _kf(0.35, s*-50, curve=EASE_OUT),
                _kf(0.80, s*8, curve=EASE_IN), _kf(D, 0),
            ]}
        upper = f"{side}-upper-leg"
        lower = f"{side}-lower-leg"
        if upper in B:
            bones[upper] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, 20),   # squat bend
                _kf(0.35, -15),  # extend
                _kf(0.55, 10),   # tuck in air
                _kf(0.80, 15),   # absorb
                _kf(D, 0),
            ]}
        if lower in B:
            bones[lower] = {"rotate": [
                _kf(0, 0, curve=None),
                _kf(0.15, -30), _kf(0.35, 10), _kf(0.55, -15),
                _kf(0.80, -20), _kf(D, 0),
            ]}

    return {"bones": bones} if bones else {}


def gen_attack(B):
    """Melee attack: wind-up → strike → follow-through. ~0.6s"""
    bones = {}
    D = 0.6

    if "right-upper-arm" in B:
        bones["right-upper-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, 40, curve=EASE_IN),      # wind up (pull back)
            _kf(0.25, -80, curve=EASE_FAST),   # strike forward
            _kf(0.4, -60, curve=None),          # follow through
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "right-lower-arm" in B:
        bones["right-lower-arm"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, -40, curve=EASE_IN),
            _kf(0.25, 10, curve=EASE_FAST),
            _kf(0.4, -5, curve=None),
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "torso" in B:
        bones["torso"] = {"rotate": [
            _kf(0, 0, curve=None),
            _kf(0.1, -8, curve=EASE_IN),   # lean back
            _kf(0.25, 12, curve=EASE_FAST), # lunge forward
            _kf(0.4, 5, curve=None),
            _kf(D, 0, curve=EASE_OUT),
        ]}
    if "hip" in B:
        bones["hip"] = {"translate": [
            _kf(0, x=0, y=0, curve=None),
            _kf(0.1, x=-5, y=-5, curve=EASE_IN),
            _kf(0.25, x=10, y=2, curve=EASE_FAST),
            _kf(D, x=0, y=0, curve=EASE_OUT),
        ]}

    return {"bones": bones} if bones else {}


PRESETS = {
    "idle": gen_idle,
    "walk": gen_walk,
    "run": gen_run,
    "wave": gen_wave,
    "jump": gen_jump,
    "attack": gen_attack,
}


# ─── Spine JSON Builder ──────────────────────────────────────────────────────

def build_spine_json(config):
    """Build a complete Spine JSON structure from config."""
    bone_names = {b["name"] for b in config["bones"]}

    skel_meta = config.get("skeleton", {})
    data_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]

    spine = {
        "skeleton": {
            "hash": data_hash,
            "spine": "4.2.0",
            "x": -(skel_meta.get("width", 400) // 2),
            "y": 0,
            "width": skel_meta.get("width", 400),
            "height": skel_meta.get("height", 600),
            "images": "./images/",
        },
        "bones": config["bones"],
        "slots": config.get("slots", []),
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {},
    }

    # Build attachments for default skin
    attachments = config.get("attachments", {})
    for slot in config.get("slots", []):
        att_name = slot.get("attachment", slot["name"])
        if att_name in attachments:
            spine["skins"][0]["attachments"][slot["name"]] = {
                att_name: attachments[att_name]
            }

    # Generate preset animations
    for anim_name in config.get("animations", ["idle"]):
        if anim_name in PRESETS:
            data = PRESETS[anim_name](bone_names)
            if data:
                spine["animations"][anim_name] = data
        else:
            print(f"  WARNING: Unknown animation preset '{anim_name}', skipping")

    # Merge custom animations
    for name, data in config.get("custom_animations", {}).items():
        spine["animations"][name] = data

    return spine


def main():
    parser = argparse.ArgumentParser(description="Build Spine JSON skeleton with animations")
    parser.add_argument("--config", required=True, help="Skeleton configuration JSON")
    parser.add_argument("--output", default="skeleton.json", help="Output Spine JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Building: {config.get('skeleton', {}).get('name', 'unnamed')}")
    spine_json = build_spine_json(config)

    with open(args.output, "w") as f:
        json.dump(spine_json, f, indent=2)

    print(f"Saved: {args.output}")
    print(f"  Bones: {len(spine_json['bones'])}")
    print(f"  Slots: {len(spine_json['slots'])}")
    print(f"  Animations: {list(spine_json['animations'].keys())}")


if __name__ == "__main__":
    main()
