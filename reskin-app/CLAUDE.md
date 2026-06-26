# Design system

This project uses the Genie Studio design system at `design-system/genie-studio/`.

- Tokens: `design-system/genie-studio/colors_and_type.css` — import this in your global stylesheet.
- Icons: `design-system/genie-studio/assets/icons/` — 24×24 stroke SVGs.
- Reference compositions: `design-system/genie-studio/ui_kits/web/index.html`.
- Read `design-system/genie-studio/README.md` before designing anything.

Never invent colors, type, spacing, or components not grounded in this system.

## Voice

Calm, plain, second-person, sentence case, no emoji, no exclamation marks.
Empty states: short and concrete. Loading states: noun + ellipsis (`Generating…`).
Affordance labels: verbs (`Edit`, `Save`, `Generate`, `Open project`).
Inline `+` for primary creation (`+ New skin`).

## Visual rules

- **Light theme only** (no dark mode). Default surface `--charcoal-0` (white); text `--charcoal-8`.
- **One brand colour**, used sparingly: Genie blue `--genie-4` (#45b3fd) — primary buttons, focus rings, links, brand mark, thinking shimmer.
- **Borders**: 1px `--charcoal-2`. Slightly stronger `--charcoal-2-5` for nav/outline buttons that need contrast.
- **Shadows**: only the three canonical ones (`--shadow-card`, `--shadow-button`, `--shadow-attachment`). No glow, no coloured shadows.
- **Radii ramp**: 4 / 6 / 8 / 12 / 16 / 9999. Buttons 8px, inputs 6px, side panel 16px, pill buttons fully round.
- **Spacing**: 4px base. No margins for layout — padding + flex/grid + gap only.
- **Type**: SF Pro stack (system fonts) or Inter fallback. Weights 300/400/500/600 only — no bold, no italics. Body 14px (`text-sm`). Headers 16/20/24/30 px depending on level.

## Iconography

Bespoke 24×24 stroke icons live at `design-system/genie-studio/assets/icons/`. They are SVG, 1.5px stroke, `stroke="currentColor"`. Use these first; fall back to `lucide-react` at the same visual weight if a glyph isn't covered.

Never mix in Heroicons, Phosphor, Material, or emoji-as-icons.

## When implementing UI

Refer to `design-system/genie-studio/ui_kits/web/index.html` for layout patterns:
- Floating chrome over canvas: rounded white capsules with `bg-background/95` + `shadow-card` + `border-charcoal-2-5`.
- Side panel: fixed-width, `p-6`, `border-radius: 16px`, soft white-ish surface.
- Cards: white surface + 1px `--charcoal-2` border + `--shadow-card`.

Always read `design-system/genie-studio/README.md` for content fundamentals (tone, casing, copy patterns) before writing any UI strings.
