# Contributing to Spine Animation AI

Thank you for your interest in contributing! This project is community-driven and welcomes all skill levels.

## Ways to Contribute

### 🐛 Bug Reports
Found a script that crashes or produces bad output? Open an issue with:
- The command you ran
- Your Python version (`python3 --version`)
- The error message or bad output
- A minimal example (attach files if possible)

### ✨ New Features
Before opening a large PR, open an issue first to discuss. Some ideas we'd love to see:

- **New animation presets** — dance, swim, fly, cast-spell, crouch, death
- **Non-humanoid rigs** — animals, creatures, vehicles, abstract shapes
- **Better occlusion detection** — for heavily layered / occluded parts
- **Asset pipeline integrations** — Blender exporter, Aseprite → atlas, LibreSprite
- **Runtime examples** — Unity C#, Godot GDScript, Phaser 3, PixiJS
- **Improved SKILL.md** — better Claude prompting patterns, edge cases

### 📖 Documentation
Documentation PRs are always welcome — fix typos, clarify instructions, add examples.

---

## Getting Started

```bash
git clone https://github.com/your-username/spine-animation-ai
cd spine-animation-ai

# Install Python dependencies
pip install opencv-python Pillow numpy

# Run the example
python3 scripts/position_parts.py --help
```

---

## Code Style

- **Python**: PEP 8. Max line length 100. Type hints for new functions.
- **Docstrings**: Use Google-style docstrings for public functions.
- **Scripts**: Each script should be self-contained with a `--help` that explains all flags.
- **Commit messages**: Conventional Commits preferred (`feat:`, `fix:`, `docs:`, `refactor:`)

---

## Testing New Scripts

Before submitting, test your script against the included sombrero example:

```bash
python3 scripts/your_new_script.py \
  --input examples/sombrero/sombrero.json \
  --output /tmp/test_output/
```

Include a note in your PR about what you tested.

---

## Pull Request Checklist

- [ ] Script runs without errors on the sombrero example
- [ ] `--help` flag works and explains all options
- [ ] README updated if adding a new script or major feature
- [ ] SKILL.md updated if Claude's behavior should change
- [ ] No large binary files added (use Git LFS for PNGs > 1MB)

---

## Project Philosophy

This skill should feel like a knowledgeable co-pilot, not a magic black box.
The output should be **inspectable** (JSON you can read), **adjustable** (easy to tweak),
and **portable** (works with any Spine runtime, not just our preview).

When in doubt: prefer clarity over cleverness, and correctness over speed.

---

Questions? Open an issue or start a discussion. We're friendly here. 🙂
