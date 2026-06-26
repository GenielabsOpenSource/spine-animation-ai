# Genie Studio — Agent Skill

This is a design system extracted from the **Genie Studio** web app (codename: Creative Cortex), an AI canvas-based image creation tool by Genie Labs. Use these tokens, components, and conventions when designing for or extending the product.

## When to use this skill

Reach for this skill when you're:
- Designing **new screens, flows, or features** for Genie Studio (the web app at `/boards`, `/board/:id`)
- Building **marketing or onboarding surfaces** in the Genie Studio visual language
- Making **internal tools** that should match the Genie brand
- Recreating Genie Studio UI in mocks, prototypes, or hi-fi designs

Don't use it for: unrelated Genie Labs products, marketing collateral that calls for its own visual identity, or anything that needs full-bleed photography / heavy gradients (the system explicitly avoids those).

## Where to look

| Need | File |
|---|---|
| Color + type tokens (CSS vars) | `colors_and_type.css` |
| Tone of voice, copy patterns | `README.md` → CONTENT FUNDAMENTALS |
| Spacing, radii, shadows, motion | `README.md` → VISUAL FOUNDATIONS |
| Iconography rules (bespoke + Lucide) | `README.md` → ICONOGRAPHY |
| Brand mark | `assets/genie-icon.svg` |
| Product icons (24×24 SVG, currentColor) | `assets/icons/` |
| Token review cards | `preview/` |
| Composed screens (Boards + Canvas + Chat) | `ui_kits/web/index.html` |

## Hard rules (memorize)

1. **One brand color, used sparingly.** Genie blue `#45b3fd` (`--genie-4`) for primary buttons, focus, links, brand. Charcoal carries the rest.
2. **Sentence case everywhere.** "+ New board", not "+ New Board".
3. **No emoji. Ever.** Not in copy, not as icons.
4. **No marketing gradients.** The thinking shimmer is the *only* expressive gradient. Don't invent purple/blue washes.
5. **Padding + flex/grid + gap. No layout margins.**
6. **Icons are stroke 1.5px, currentColor, 24×24** — bespoke set in `assets/icons/`, otherwise Lucide. Never Heroicons / Phosphor / Material.
7. **Floating chrome over canvas = translucent white pill.** `bg-white/95 + backdrop-blur + border-charcoal-2.5 + shadow-card`. Never use vignette/protection gradients.
8. **Soft shadows only.** Three canonical: `shadow-card`, `shadow-button`, `shadow-attachment`. No inner, no colored, no glow.
9. **Light mode only.** No dark mode in production.
10. **Inter is a stand-in for SF Pro.** Real product looks slightly tighter. Don't add other display fonts.

## Copy templates

- **CTA:** `+ New board`, `+ New`, `Share`
- **Empty state:** `No boards yet` / `Create your first board to get started`
- **Loading:** `Creating…` (U+2026, never `...`)
- **Error toast:** `Failed to <verb> <noun>` (e.g. `Failed to save board name`)
- **Page header:** noun + short noun-phrase subtitle (`Boards` / `Your canvas workspaces`)
- **Tab labels:** `Genie`, `Asset Library`, `Image History`
- **Affordance labels:** verbs only (`Edit`, `Resize`, `Remove BG`, `Split layers`, `Download`)

## Default component recipes

- **Button (primary):** `h-8 px-4 rounded-lg bg-charcoal-7 text-white text-sm font-medium`, hover → `bg-charcoal-6`.
- **Card:** `bg-white border border-charcoal-2 rounded-lg shadow-button`, hover → `shadow-md, border-charcoal-2.5`.
- **Toolbar (floating over canvas):** vertical capsule, 4px inner padding, 36px buttons, 8px radius. Hover → `highlight-2`. Active → `highlight-4`.
- **Side panel:** fixed 400px right, full height, 24px page padding, inner panel = `bg-white/92 + backdrop-blur + 16px radius + shadow-card`.
- **Prompt input:** 12px-radius bordered group, textarea + addon row with attachment / layout icon-buttons on left, 32px send button (`bg-charcoal-7`) on right.
- **Thinking shimmer:** `Inter 300`, gradient `genie-4 → mag-4(0.75) → genie-4`, 3s linear infinite, `background-clip:text`.

## Caveats

- SF Pro substituted with Inter; production text is slightly tighter.
- No Figma — component specs come from reading the React source.
- No dark mode coverage.
- No marketing/landing surfaces in source.
