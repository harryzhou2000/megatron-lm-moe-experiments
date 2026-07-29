# NVIDIA-light Beamer template

This project is a LaTeX Beamer visual adaptation of the supplied
`NVIDIA_PPT_Light_External_v2.potx`. It provides a 16:9 light theme with
NVIDIA Sans typography, the template's extracted color palette, title artwork,
wide margins, charcoal titles, transition artwork, and source/page footer treatment.

It is intentionally a reusable starter project rather than an export of the
PowerPoint itself. The included `main.tex` is a short working example based on
the adjacent Sparser-MoE content note; replace its copy and frames with your
own deck.

## Build

The project uses XeLaTeX because it can access system fonts, including NVIDIA Sans.

```sh
cd notes/final_presentation/beamer
make
open build/main.pdf
```

If the managed NVIDIA Sans files are unavailable, the theme falls back to Helvetica Neue. Use
`make clean` to remove build output and `make watch` while authoring.

## Theme primitives

- `\NvidiaSetFooter{...}` sets a small persistent footer label.
- `\NvidiaSource{...}` adds a source note to one standard slide.
- `\NvidiaTransitionSlide{title}` creates a template-derived image transition.
- `\NvidiaMetric{value}{label}{scope}` creates a compact result callout.
- `NvidiaEvidence` is a restrained green-led evidence annotation.
- `\NvidiaTextImageFrame{title}{subtitle}{image}{source}{copy}` creates a text-left,
  image-right page based on the template's one-up alternative layout.
- `\NvidiaImageTextFrame{title}{subtitle}{image}{source}{copy}` creates the mirrored
  image-left, text-right page.
- `\NvidiaWideImageFrame{title}{subtitle}{image}{source}{caption}` creates a centered
  16:9 one-up page for a large image, chart, or product visual.

For image-layout frames, pass a concise source string in the fourth argument.
Use an image whose aspect ratio suits the intended slot (1:1 or 4:5 for
side-by-side pages, 16:9 for the one-up page). Keep visual labels readable at
presentation distance; move a dense chart to a wide page instead of shrinking
it into a side column.

The title, transition, and logo assets in `assets/` are extracted from the supplied
PowerPoint template or its canonical PDF export. `../NVIDIA_PPT_Light_External_v2.pdf`
is the canonical exported reference used to verify the layout. NVIDIA names and marks
remain subject to NVIDIA's applicable brand and trademark rules; this project
is a visual approximation, not an official NVIDIA deliverable.
