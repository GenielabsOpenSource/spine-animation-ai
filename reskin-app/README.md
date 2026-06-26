# Genie Spine Reskin

AI-powered reskinning for Spine 2D character projects. Loads an existing Spine
rig, renders the character via the live `spine-pixi-v8` runtime, and lets you
generate a new skin in one click — Gemini reskins the rendered character, SAM
slices the result back into per-slot textures, and the new skin reassembles on
the rig with no rectangular bleed between slots.

## What you get

- **Open project** — point at a folder containing `Spine.json` + `*.atlas` +
  `*.png`. The rig renders in a PixiJS canvas with the original skin.
- **+ New skin** — describe the look (`emerald and silver royal robes`) and
  click Generate. The pipeline:
    1. Snapshots the live canvas as a clean static reference.
    2. Pads the snapshot to Gemini's nearest supported aspect ratio so the
       output maps 1:1 back to input pixels.
    3. Sends to Gemini Nano Banana for the global reskin.
    4. Crops back to the original dimensions.
    5. Computes per-slot bboxes by toggling slot visibility on the live spine
       instance.
    6. Calls the SAM-3 server (one HTTP call) with all bboxes; receives one
       precise mask per slot.
    7. Repacks a per-skin atlas + writes a per-skin `Spine-{skin}.json` with
       the new skin entry.
    8. The canvas reloads spine-pixi against the per-skin atlas → the rigged
       character renders with the AI-reskinned per-slot textures.
- **Mask editor** — brush, eraser, lasso, and SAM-powered Magic Select for
  refining any slot's mask after the fact.
- **AI Terminal** — per-slot inpainting. Pick a slot, type a prompt, Gemini
  redraws just that part preserving the silhouette.
- **Per-slot edits** — non-destructive HSL/RGB/contrast/transform sliders.
- **Export** — writes the new skin's atlas + per-skin Spine JSON next to your
  original project so you can open it in Spine 2D.

## Project layout

```
app/
├── backend/          FastAPI server
│   ├── server.py     Routes for project, reskin, mask, export
│   ├── projects.py   Project model + open logic
│   ├── ai/           Gemini provider + SAM provider
│   ├── reskin/       compose / slice / pipeline
│   ├── spine/        parser, skin_writer, atlas_repack
│   └── imaging.py    HSL/RGB/transform for per-slot edits
└── frontend/         Vite + React + TypeScript + spine-pixi-v8
    ├── src/components/
    │   ├── canvas/SpineCanvas.tsx       Live rig render
    │   ├── Sidebar.tsx                   Slot list + visibility/lock
    │   ├── right-panel/PartEditor.tsx   Edits + AI Terminal + Mask launcher
    │   └── modals/                       New skin / AI Terminal / Mask
    └── src/styles/
        ├── genie-tokens.css   ← design system tokens (do not edit)
        └── index.css

design-system/genie-studio/  — design tokens, icons, brand mark, README

scripts/
├── explode_spine_atlas.py   Ingests a packed Spine atlas → per-region PNGs
└── make_atlas.py             Row-based bin packer (used by atlas_repack)
```

## Setup

### Requirements

- Python 3.9+ (3.10+ recommended)
- Node 18+
- A Gemini API key (`GEMINI_API_KEY`)
- A SAM-3 segmentation server reachable over HTTP (`SAM_SERVER_URL`)
- Optional: an Anthropic API key for the chat sidebar (`ANTHROPIC_API_KEY`)

### Env vars

Copy `app/.env.example` to `app/.env` and fill in:

```
GEMINI_API_KEY=...
SAM_SERVER_URL=http://your-sam-host:30231
ANTHROPIC_API_KEY=...
```

### Install + run

```bash
# Backend
cd app/backend
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.backend.server:app --host 127.0.0.1 --port 8765 --app-dir ../..

# Frontend (separate terminal)
cd app/frontend
npm install
npm run dev
```

Open http://localhost:5173.

### Ingesting a Spine project

If your Spine project uses an old `bounds:`-style atlas (e.g. exports from
older tools), run the explode script once to produce a standard Spine 4.x
atlas plus per-region PNGs:

```bash
python scripts/explode_spine_atlas.py \
  --char-dir /path/to/CharacterFolder \
  --base-name CharacterAtlasBaseName \
  --output-dir /wherever/you/want/the/genie-project
```

Open the output dir in the app.

### Magic Select

To enable SAM-based magic select in the mask editor, drop the SAM ONNX
decoder file at:

```
app/frontend/public/magic_cut.onnx
```

The frontend probes it at runtime; if it's missing, magic select shows a
clear hint and the other tools still work.

## Design system

This project uses the Genie Studio design system at
`design-system/genie-studio/`. Read `CLAUDE.md` and the design system's
`README.md` before touching any UI — never invent colors, type, spacing, or
components not grounded in the system.
