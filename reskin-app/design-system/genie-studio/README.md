# Genie Studio Design System

A design system extracted from the **Genie Studio** web app (internal project name: **Creative Cortex**) — a canvas-based AI image creation tool by Genie Labs.

## What is Genie Studio?

Genie Studio is a single-page React app where users **create AI-generated imagery on an infinite 2D canvas**. The product is built around a few core concepts:

- **Boards** — named workspaces. Each board has its own canvas state and conversation history. Boards are listed on `/boards` as a card grid.
- **Canvas** — a Konva-powered infinite 2D surface. Users place *frames* (named regions), *creation images* (AI outputs), *text* nodes, and arrange them freely. Toolbar on the left: Move (V), Hand (H), Frame, Text, Add image.
- **Genie chat** — the AI side panel on the right. Users describe what they want; the agent streams back imagery onto the canvas. Chat supports file attachments, image-reference attachments (drag a node from canvas into chat), and `@`-mentions of tagged objects.
- **Creations** — image outputs land on the canvas with hover affordances: Edit, Resize, Remove BG, Split layers, Download.

The "Genie" half of the brand is friendly + curious. The "Studio" half signals *real creative work* — this is not a toy. Visually it leans crisp and quiet: lots of white, one strong brand blue (`#45b3fd`), restrained typography, soft card shadows.

## Sources

This system was reverse-engineered from the production web codebase only. No Figma was provided.

- **Codebase (read-only mount):** `web/` — Vite + React 19 SPA
  - Tokens: [`web/src/index.css`](web/src/index.css) — Tailwind v4 `@theme` block (lifted verbatim into `colors_and_type.css`)
  - UI primitives: `web/src/components/ui/` (Button, Tabs, Select, Popover, Tooltip, Menu, …) — built on Base UI + CVA
  - Canvas: `web/src/components/canvas/` — Konva renderer, toolbar, icon set
  - Chat: `web/src/components/chat/` — SSE streaming, prompt input, attachments, thinking shimmer
  - Boards: `web/src/components/boards/`, `web/src/routes/boards/`
  - App index: `web/index.html` (`<title>Genie Studio</title>`, favicon → `genie_Icon.svg`)

The backend ("Creative Cortex") is a separate FastAPI service. This design system covers the **web client only**.

---

## Index

| File / folder | Purpose |
|---|---|
| `README.md` | This file |
| `SKILL.md` | Cross-compatible Agent Skill manifest |
| `colors_and_type.css` | All color + type tokens (mirror of `web/src/index.css`) |
| `assets/genie-icon.svg` | Brand mark — the lamp swoosh |
| `assets/icons/` | 28 product SVG icons (canvas tools, frame chrome, creation actions, chat) |
| `assets/image-placeholder.svg` | Placeholder graphic used by canvas creation slots |
| `preview/` | Design-system review cards (rendered in the Design System tab) |
| `ui_kits/web/index.html` | Composed surfaces — Boards index, Canvas with floating chrome, Genie chat panel |

There are **no slide templates** in this system — Genie Studio doesn't ship presentation surfaces.

---

## CONTENT FUNDAMENTALS

The product copy is **calm, plain, second-person, and very low on adjectives**. The voice is a quiet collaborator, not a hype assistant — no exclamation marks, no rocket emoji, no "Let's go!".

### Tone
- **Direct and instructional.** "Describe what you'd like to create." "Your canvas workspaces." "Create your first board to get started."
- **Second-person ("you").** Never "we" or "I". The agent doesn't refer to itself by name in copy chrome — only the tab is labeled `Genie`.
- **Sentence case everywhere.** Buttons read "+ New board", "Back to canvas", "Boards" — never `New Board` or `BOARDS`.
- **No greetings, no sign-offs.** The chat opens with a textarea placeholder, not a "Hi! How can I help?".
- **Quiet error states.** Toasts use Sonner with `richColors`; copy is matter-of-fact: "Failed to create board", "Failed to switch conversation".

### Specific patterns
- **Empty states are short and concrete.** `No boards yet` / `Create your first board to get started`.
- **Loading states are nouns + ellipsis.** `Creating…` (note: U+2026, not three dots).
- **Status bar copy is single words / short phrases.** `Boards`, `Genie`, `Asset Library`, `Image History`, `Share`.
- **Affordance labels are verbs.** `Move tool`, `Hand tool`, `Edit`, `Resize`, `Remove BG`, `Split layers`, `Download`.
- **Inline `+` for primary creation.** `+ New board`, `+ New` (chat). The `+` is part of the label, not an icon.

### Emoji
- **Never.** No emoji appear anywhere in product copy or component defaults. Don't introduce them.

### Casing
- **Sentence case** for all UI strings, toasts, headers.
- **PascalCase** only for the brand label `Genie` and proper nouns (`Auth0`, `Konva`).
- **Title Case** is not used.

### Examples to reuse
- Page header: `Boards` / subtitle: `Your canvas workspaces`
- Empty list: `No boards yet` / `Create your first board to get started`
- CTA: `+ New board`
- Card meta: `Edited 3 days ago`
- Chat placeholder: `Describe what you'd like to create`
- Tab labels: `Genie`, `Asset Library`, `Image History`
- Toast (error): `Failed to save board name`

---

## VISUAL FOUNDATIONS

The system is **light, airy, and content-first**. Imagery (creations on the canvas) is the protagonist; everything else is chrome that gets out of the way.

### Color
- **One brand color**, used sparingly: Genie blue `#45b3fd` (`--genie-4`). Appears in primary buttons, focus rings, links, the brand mark, and the thinking shimmer.
- **Charcoal scale** (`charcoal-0` → `charcoal-8`) carries the entire grayscale UI — borders, text, surfaces, dividers. Pure white (`charcoal-0`) is the default surface; pure-ish black (`charcoal-8`, `#101010`) is the default text.
- **Highlight tints** (`highlight-1` → `highlight-4`) — pale near-white blues used *only* for hover/active/selected states on toolbar buttons. They read as "almost white with a hint of cool" on the canvas.
- **Accent scales kept in reserve.** `mag-*` (magenta), `mala-*` (red), `bina-*` (green) exist for destructive, error, and success states respectively, but the product surfaces them rarely. A typical screen uses charcoal + one shot of genie blue, full stop.
- **Soft "Bgs" palette** (`--bgs-2` through `--bgs-7`: cream, mustard, peach, sky, pistachio, pink) — these are the **frame fill colors** users can choose for canvas frames. They are intentionally low-saturation, gallery-friendly.
- **No dark mode in production.** The CSS contains a `@custom-variant dark` hook but the app ships light-only.

### Type
- **Family:** SF Pro Display / SF Pro Text first, falling back through system UI. Inter is loaded as a webfont (`@fontsource-variable/inter`) and is used specifically for the **thinking shimmer** in chat — everywhere else, system SF wins on Apple devices and Segoe/Helvetica on others. We've substituted Inter for the SF stack in this design system since SF Pro is not licensable for distribution; **flag**: in-product text should look slightly tighter than what we render here.
- **Weights:** 300 (thinking shimmer only), 400 (body), 500 (medium / button labels), 600 (semibold / headers). No 700+, no italics.
- **Letter-spacing:** default. Headers do not use negative tracking.
- **Sizes:** the smallest size in product is `--text-xxs` (10px), reserved for kbd shortcuts. Body is 14px (`text-sm`), card titles 14px medium, page H1 24px semibold.
- **Line-height:** Tailwind defaults — 1.5 body, 1.25 headers.

### Spacing
- **4px base.** All spacing snaps to multiples of 4. Common rungs: 4, 6, 8, 12, 16, 24, 28, 40 px.
- **Compact density on chrome, generous on content.** The canvas toolbar, prompt input addon row, and tab bar are tight (`gap-1`, `gap-1.5`). The boards list grid uses `gap-4`. Page padding is `px-6 py-10`.
- **No margins for layout.** The codebase explicitly bans margin-driven layout — everything is padding + flex/grid + gap.

### Backgrounds
- **Plain white** (`#ffffff`) for the canvas page background; **soft off-white** (`charcoal-1`, `#f9f9f9`) for the boards list page.
- **Canvas surface** itself is `--canvas` `#f3f4f5` — a very pale neutral that lets the colorful frames pop.
- **No full-bleed photography. No textures. No patterns.** The system is illustration-free outside the brand mark.
- **Gradients are rare and quiet.** The board card placeholder uses `bg-linear-to-br from-charcoal-1 to-charcoal-2` (an extremely subtle white-to-light-gray). The thinking shimmer is the only "expressive" gradient — a horizontal genie-blue → magenta sweep. **Do not invent purple/blue marketing gradients.**

### Animation
- **Motion library:** Motion (Framer Motion v12).
- **Thinking shimmer:** 3s linear infinite — the most visible motion in the app.
- **Otherwise: subtle CSS transitions** on hover/focus. `transition-all` is common but durations come from Tailwind defaults (~150ms). Cards use simple transitions on `box-shadow` and `border-color`.
- **No bounces, no springs in chrome.** Konva canvas uses pure functional transforms (zoom-to-point, etc) without easing flourishes.
- **Streaming text uses `Streamdown`** — incremental render with subtle reveal.

### Hover states
- **Lighten or shift hue toward genie blue.** Buttons darken slightly (`bg-genie-7 → bg-genie-6`); outline buttons gain a `genie-3` border + `genie-4/10` fill; nav buttons swap charcoal-3 borders for charcoal-4 + a slight shadow.
- **Toolbar buttons go to `highlight-2`** — pale blue, almost white.
- **Cards bump shadow + tighten border** (`hover:shadow-md`, `hover:border-charcoal-2.5`).
- **Tabs:** `hover:bg-charcoal-3/15 hover:text-charcoal-8` — a charcoal wash, not a color shift.

### Press / active states
- **Active = darker fill or `genie-2/90` background.** No scale transforms, no shrinking. The `:active` state usually drops the shadow (`active:shadow-none`) to reinforce "pressed".
- Outline button active: border becomes transparent, fill goes to `genie-2/90`, text to `genie-6`.

### Borders
- **Standard border:** 1px `--charcoal-2` (`#e3e3e3`). Slightly darker `--charcoal-2.5` (`#b9b7b6`) for buttons that need a touch more contrast (outline, nav).
- **Focus ring:** 1px `ring-ring/50` (genie-4 at 50%) plus a 1px `border-ring`. Crisp, not glowy.
- **Dashed borders** appear in empty states (`border-dashed border-charcoal-2`).

### Shadows
- Three canonical shadows, all extremely soft:
  - `--shadow-card` `-1px 4px 8px rgba(0,0,0,0.08)` — side panels, popovers, surfaces. Note the **negative X** — light comes from the right.
  - `--shadow-button` `0 1px 2px 1px rgba(0,0,0,0.04)` — buttons and inputs.
  - `--shadow-attachment` `0 2px 4px rgba(0,0,0,0.08)` — chat attachments.
- **No inner shadows. No colored shadows. No glow effects.**

### Capsules vs protection gradients
- **Capsules**, never protection gradients. The canvas overlays floating chrome (toolbars, popovers, board name field) on top of canvas content using **rounded white surfaces with `bg-background/95` + `shadow-card` + `border-charcoal-2.5`** — i.e. tinted-white pills, not vignette gradients. This is consistent across canvas toolbar, board name toolbar, and side panel.

### Layout rules / fixed elements
- **Canvas page:** chrome is positioned `absolute` over the Konva stage. Top-left has the hamburger nav + board name (`top-7 left-7`); right edge has the side panel (`top-0 right-0 h-full p-6`); the canvas toolbar is centered vertically near the left edge.
- **Side panel:** fixed width (`SIDE_PANEL_WIDTH_PX`, 400 px in the source — confirm), full viewport height with `p-6` from the screen edge.
- **Page max-width** for content lists: `max-w-7xl` mx-auto with `px-6` gutters.

### Transparency & blur
- **Backdrop blur** is used **only** on the side panel (`bg-background/90 backdrop-blur-sm`) and on `nav` variant buttons. It signals "floating over canvas content" — never used on full-page surfaces.
- **Translucent white surfaces** (`bg-charcoal-0/50`, `bg-background/95`) are common for chrome that overlays the canvas, so canvas content peeks through faintly.

### Imagery
- The product itself **doesn't ship marketing imagery**; user-generated images dominate. When we need placeholders, the codebase's `image-placeholder.svg` is the canonical empty-state graphic.
- Imagery on the canvas is presented as-is — **no filters, no overlays, no rounded corners** unless the user adds them.

### Corner radii
- A clean ramp: 4 / 6 / 8 / 12 / 16 / 9999. Most product chrome uses **8px** (buttons, toolbar). Inputs are **6px**. The chat composer is **12px**. The side panel is **16px**. Pill buttons (`nav`, `canvasPopover`) are **fully round**.

### Cards
- **Cards = white surface + 1px charcoal-2 border + `shadow-card` + 8–12px radius.** No drop shadows beyond the canonical card shadow. Hover lifts shadow to `shadow-md`.

### Iconography summary
See **ICONOGRAPHY** section below.

---

## ICONOGRAPHY

Genie Studio uses two complementary icon systems, both stroke-based.

### 1. Bespoke 24×24 SVG set (canvas + chat chrome)
Lives in `web/src/components/canvas/icons/` and `web/src/components/chat/icons/`. Imported via `vite-plugin-svgr` with the `?react` suffix and rendered inline as React components. Currents:
- **Stroke-based, 1.5px stroke width.** `stroke-linecap="round"`, `stroke-linejoin="round"`. Minimal fills — usually only when the icon needs to read at very small sizes (e.g. `cursor.svg` is filled, `cursor-fill.svg` is the alternate).
- **24×24 viewBox**, except the chat icons (`send`, `attachment`, `layout`) which are 16×16.
- **`stroke="currentColor"`** so they inherit text color from their parent. Icons should always be sized via Tailwind size utilities, not via the SVG `width`/`height` attributes.
- **Style:** clean, even-thickness, geometric — close in spirit to Lucide and Phosphor "regular" but custom-drawn. Slightly chunkier than Lucide's default 1px feel.

We've copied **all** of these into `assets/icons/`:

```
add, align-center, align-left, align-right, attachment, chevron-down, copy,
creation-image-{download,edit,remove-bg,resize,split-layers},
cursor, cursor-fill, frame, frame-background-transparent, frame-label,
frame-toolbar-options, hamburger, hand, key-ref-frame, keyart-label, layout,
send, text, zoom
```

### 2. Lucide React (fallback / general UI)
For everything not covered by the bespoke set, the codebase uses [`lucide-react`](https://lucide.dev). Examples seen in source: `Loader2Icon`, `SquareIcon`, `XIcon` (in the prompt input submit button). Stroke style matches the bespoke set well enough that mixing them reads consistently.

**Recommendation:** if you need an icon not in `assets/icons/`, pull it from Lucide at the same 1.5px-feel size (`size-4` / `size-5` Tailwind). Don't mix in Heroicons, Phosphor, or Material icons — the visual grammar will clash.

### What's *not* used
- **No icon font.** Icons are inline SVGs only.
- **No emoji as icons.** Anywhere.
- **No unicode icon glyphs** (no ✓, no ▾). Caret/check/etc are SVGs.
- **No PNG icons.**

### Brand mark
`assets/genie-icon.svg` — the swooping lamp/genie wisp shape, filled in `genie-4` blue. 452×452 viewBox, single-path. Used as the favicon and for any "Genie" branding moment. **Do not recolor it** outside the genie-* scale.

---

## CAVEATS

- **SF Pro substitution.** SF Pro Display/Text is Apple-only and not licensable for distribution. We've fallen back to **Inter** in this design system. In production on macOS/iOS, the real product looks slightly tighter and slightly more rounded than what these previews render. If you need pixel-perfect fidelity, install SF Pro locally.
- **No Figma was provided**, so component specs come from reading source. State variants (focus-visible, disabled, data-popup-open, etc) are captured from CVA definitions but not all combinations have been visually verified.
- **No dark mode coverage.** Production ships light-only.
- **No marketing/landing surfaces** were in the codebase — this is a logged-in app only.
