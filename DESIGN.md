# DESIGN.md: Weld Quality and Defect Classifier

> Spec first, code second. A theme for a Streamlit inference app.
> Reference: `kaopu-xiaopu.github.io/web-design` (Dark Editorial, teal and orange).
> Deliberate deviation: the target medium is a Streamlit app (Python, sandboxed
> HTML), so the skill's L2 and L3 GSAP/WebGL/scroll-story red lines do not apply.
> Interaction tier is L1 (refined hover and soft CSS entrance), applied via injected
> CSS plus `st.set_page_config`. No hardcoded hex in the app: all colors are CSS vars.

---

## 1. Visual Theme and Atmosphere
Style: Dark Editorial crossed with Industrial Inspection Tool.
Keywords: precise, technical, trustworthy, calm, instrument-grade, readable.
Tone: dark but not cold, technical but not cyber. A radiography viewer, not a game.
Feel: looking at a weld through an inspection panel, dark field with bright findings.
Interaction Tier: L1, soft fade-in on content, smooth hover on controls and cards.

## 2. Color Palette and Roles
```css
:root {
  --bg: #0a0b0e; --bg-2: #07080b;
  --surface-1: #111217; --surface-2: #171921; --surface-3: #1f222d; --surface-hover: #242836;
  --border: rgba(67,70,81,.5); --border-strong: rgba(67,70,81,.9); --border-accent: rgba(94,234,212,.3);
  --text-1: #ebecef; --text-2: #c6c9d2; --text-3: #8d909c; --text-4: #60636f;
  --accent-cool: #5EEAD4; --accent-cool-hover: #8CF5E3; --accent-cool-soft: rgba(94,234,212,.12);
  --accent-warm: #FB923C; --accent-warm-hover: #FDB874; --accent-warm-soft: rgba(251,146,60,.12);
  --gradient-key: linear-gradient(135deg,#5EEAD4 0%,#FB923C 100%);
  /* Class semantics map to the three weld classes */
  --cls-bad: #F87171;    /* bad-weld  */
  --cls-defect: #FB923C; /* defect (warm accent) */
  --cls-good: #5EEAD4;   /* good-weld (cool accent) */
  --accent-cool-rgb: 94,234,212; --accent-warm-rgb: 251,146,60;
}
```
Rules: accents only on CTA, active, key words, findings. Large areas stay surface.
Gradient (`--gradient-key`) only on the page title, one spot. Class colors are the
single source of truth for the legend, the detection table, and the summary chips.

## 3. Typography Rules
```css
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400..700&family=Instrument+Serif:ital@1&family=JetBrains+Mono:wght@400;500&display=swap');
--font-ui: 'DM Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
--font-serif: 'Instrument Serif', Georgia, serif;   /* italic key words only */
--font-mono: 'JetBrains Mono', ui-monospace, Menlo, monospace;
```
| Role | Font | Size | Weight | LH | Tracking |
|------|------|------|--------|----|----------|
| App title (H1) | DM Sans | clamp(30px,4vw,46px) | 700 | 1.08 | -0.02em |
| Section H2 | DM Sans | 22px | 600 | 1.2 | -0.01em |
| Eyebrow label | DM Sans | 12px | 600 | 1.4 | 0.08em UPPER |
| Body | DM Sans | 15px | 400 | 1.6 | 0 |
| Caption / small | DM Sans | 13px | 500 | 1.5 | 0 |
| Numbers / mono | JetBrains Mono | 14px | 500 | 1.4 | 0 |

Rules: headings DM Sans weight 600 or more. Instrument Serif italic only for the
word "defect" in the title and subtitle (at most 2 uses). Confidence values and
mask percentages use mono (instrument feel).
Never: Arial, Times New Roman, Comic Sans, default Streamlit serif fallback.

## 4. Component Stylings (states included)
- Buttons and download: pill, `--accent-cool` fill, `color:var(--bg)`. Hover: `--accent-cool-hover` plus translateY(-1px). Focus: 3px `--accent-cool-soft` ring. Active: translateY(0). Disabled: `--surface-3`, `--text-4`, no transform.
- Cards (image panels, metrics): `--surface-1`, `1px --border`, radius 14px. Hover: `--surface-hover` with border `--border-strong`.
- Sliders: track `--surface-3`, filled `--accent-cool`, thumb `--accent-cool` with soft glow on focus.
- Tabs: inactive `--text-3`, active `--text-1` with a `--accent-cool` indicator, hover `--text-2`.
- Detection table: header `--surface-2` with `--text-4` uppercase micro labels, rows `--surface-1`, mono numerics, class name colored by class.
- Class chips and legend: pill, 11px dot in the class color, label `--text-2`, bg `--surface-2`.

## 5. Layout Principles
Max content width 1100px, centered. 8-point spacing scale (8/16/24/40). The compare
view splits into equal columns per model and stacks on narrow screens. Sidebar holds
controls only. Generous vertical rhythm (about 40px between sections).

## 6. Depth and Elevation
Flat first. `--shadow-1: 0 1px 2px rgba(0,0,0,.4)` on cards, `--shadow-2: 0 8px 30px
rgba(0,0,0,.45)` on the annotated result panel only. Accent glow `0 0 0 3px
var(--accent-cool-soft)` on focused controls. No heavy drop shadows.

## 7. Animation and Interaction (L1)
- Content fade-in-up on load: `@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1}}`, 0.4s ease, applied once to the main block.
- Hover transitions 0.18s cubic-bezier(.2,0,0,1) on buttons, cards, tabs.
- `prefers-reduced-motion`: disable fadeUp and transforms, keep color transitions.
- No scroll-jacking, no JS motion libraries (not available in Streamlit's main pane).

## 8. Do's and Don'ts
Do: keep the field dark so findings pop, use class colors consistently everywhere,
use mono for all numbers, keep one gradient accent (the title), keep touch targets
44px or larger, label every control, keep the legend visible.
Don't (avoid): hardcoding hex in app.py (use the CSS vars), coloring large areas
with teal or orange, using more than one gradient, using emoji as the only class
indicator (use color dots), a light theme (it kills contrast on radiographs), the
default Streamlit red primary, body text below 15px, and motion without a
reduced-motion fallback.

## 9. Responsive Behavior
Breakpoints: 820px and up for side-by-side compare, below 820px stacked. The sidebar
collapses to Streamlit's native drawer on mobile. Touch targets 44x44px or larger.
No horizontal overflow at 600px and below, the title uses clamp(). The detection
table scrolls inside its own container, never the page.
