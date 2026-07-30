# Presentation Visual QA Playbook

This is a reusable checklist for the `beamer_presentation` deck. It records the
failure modes found while producing **Sparser MoE Explorations**, why they occur,
and the smallest reliable repair. Apply it after every material content or layout
change; do not wait until the final compilation.

## Required review loop

1. Rebuild vector assets and the deck.

       cd notes/final_presentation/beamer_presentation
       make plots
       make

2. Treat the LaTeX log as a layout test: resolve every `Overfull \vbox` and
   `Overfull \hbox` associated with a visible slide.

3. Render the PDF to page images and inspect at approximately presentation scale.
   Review a contact sheet for consistency, then inspect dense pages individually
   at higher resolution. Text extraction alone cannot establish visual quality.

       pdftoppm -r 150 -png build/main.pdf /tmp/deck/page

4. Check the opening, every section transition/Contents pair, each diagram and
   code page, all result plots/tables, the conclusions page, and the closing page.
   A second reviewer should independently check the final PDF.

## Failure modes and repairs

### 1. Body overflow, clipped titles, and footer collisions

**Symptoms**

- A title is clipped at the top or appears too close to the page edge.
- Bottom prose overlaps, hides, or visually crowds the page number/footer label.
- A page looks acceptable at low resolution but produces an `Overfull \vbox`.

**Cause**

The body exceeds the usable Beamer frame height. Columns, figures, display math,
and implicit paragraph skips accumulate; centered frame placement can make the
failure appear as title clipping rather than a bottom-only overflow.

**Repair**

- First remove unnecessary vertical space; shorten or split prose before shrinking
  everything.
- Give figures an explicit maximum height that leaves a footer reserve. Use
  `keepaspectratio` and verify the source PDF has no large white margin.
- Compact local `\vspace` values and table row spacing. If necessary, split a
  data-dense slide into two pages. Use Beamer's `[shrink=<n>]` only as a final,
  local fallback.
- Put dense frames at the top (`[t]`) only when it fixes the actual placement;
  do not globally alter frame alignment to repair one slide.
- Re-render the specific page and confirm both the left footer label and NVIDIA
  wordmark remain visible.

**Avoid**

Do not fix a body overflow by moving the global frame title or footline template.
That shifts every slide and typically creates new failures elsewhere.

### 2. Diagrams that are ugly, inconsistent, overlapping, or badly proportioned

**Symptoms**

- Flow arrows cover labels, have misplaced heads, or point to the wrong buffer
  state.
- A pipeline has inconsistent spacing, terminology, or visual hierarchy.
- A ring-buffer diagram lacks a clear enclosing buffer, producer/consumer roles,
  or state progression.
- SVG text is technically present but too small to read when embedded.

**Cause**

The diagram was treated as decoration rather than as a precise view of a
source-level dataflow. Drawing directly inside a constrained LaTeX frame also
makes iteration and visual inspection difficult.

**Repair**

- Validate the real call and kernel sequence before drawing. Make every arrow
  correspond to a real transfer or dependency.
- Build the diagram offline as SVG, inspect it directly, and insert its PDF
  conversion as vector art. Do not use in-LaTeX plotting for these diagrams.
- Use one layout grammar per diagram family: left-to-right dataflow, consistent
  box sizes, one semantic color per state/type, and arrows with clear direction.
- Put the shared-memory ring at the center of dispatch/combine designs; surround
  it with a subtle outer buffer boundary. Connect arrows to the specific state
  box, not the ring's general vicinity.
- Break long labels intentionally; increase diagram font size rather than relying
  on a large canvas scaled down into the slide.
- Inspect the SVG/PDF before compiling the deck, then inspect the rendered page.

### 3. Abnormal font sizes and aggressive metric callouts

**Symptoms**

- A number is almost title-sized and dominates the slide.
- Plot labels, code, or table text is too small to be read from a screen.
- Math uses a different visual language from the surrounding prose.

**Repair**

- Reserve title-scale text for frame titles. Large metrics should be about
  1.5 times normal prose, not title size.
- Use compact code only for essential skeleton lines. If a correct explanation
  requires smaller text than can be read, split the code across slides.
- Increase plot source font sizes before scaling the full figure down. For a
  complete sweep, split panels across slides rather than creating unreadable
  four-panel charts.
- Keep prose in the NVIDIA sans family and use serif math consistently. Use one
  font scale for headings, normal prose, captions, tables, code, and metrics.

### 4. Header-to-rule spacing and inconsistent card rhythm

**Symptoms**

- A small green rule sits visibly far below its heading.
- The same heading/rule pair has different spacing on different cards.
- A global rule macro change fixes one page but makes others cramped.

**Cause**

Heading, `\par`, and rule macros each introduce vertical glue. Cards/minipages
also have independent local baselines.

**Repair**

- Keep heading-to-rule spacing explicit and local, for example:

       {\NvidiaSansMedium Heading}\NvidiaTightRule{.6pt}\vspace{.04in}

- Use the same local sequence in every card that shares the visual grammar.
- Tighten the next element separately; do not make a broad theme-level adjustment
  to compensate for one unusually tall paragraph.

### 5. Abnormally enlarged line spacing after tables, equations, or prose

**Symptoms**

- A normal paragraph suddenly has a large gap between lines.
- A table-to-prose transition introduces unexplained whitespace.
- A display formula changes the spacing of the next paragraph.

**Cause**

LaTeX layout state leaked outside its intended scope: `\arraystretch`,
`\fontsize`, `\baselineskip`, `\parskip`, display skips, or an open
grouping interacts with a following `\par`.

**Repair**

- Scope table and font changes with braces:

       {\renewcommand{\arraystretch}{1.12}
        \fontsize{8}{9.5}\selectfont
        \begin{tabularx}{\linewidth}{...}
        ...
        \end{tabularx}\par}

- Start following prose in an explicit, local text group with
  `\setlength{\parskip}{0pt}` and `\noindent` when compact rhythm is needed.
- Set `\abovedisplayskip` and `\belowdisplayskip` locally around dense equations.
- Prefer paragraph breaks (`\par`) over stacked `\\` in prose. Do not use a
  table's `\\` as a general line-break tool.
- Re-render the whole page: the failure often appears one block after its cause.

### 6. Columns and minipages that look like mismatched cards

**Symptoms**

- Column titles/rules do not align.
- One column begins much lower because of an invisible baseline or paragraph gap.
- Boxes make a configuration slide look heavy or cluttered.

**Repair**

- Use top-aligned `columns` and top-aligned `minipage`s for card-like layouts.
- Prefer open minipages with shared heading/rule treatment over decorative boxes
  when the page is explanatory rather than categorical.
- Align both columns to a common first baseline and tune their local spacing;
  avoid forcing equal visual height with empty vertical fill.

### 7. Result plots that are complete but unreadable

**Symptoms**

- Legends, tick labels, and line identities are too small at projection scale.
- Token-count labels overlap or plot annotations sit inside bars.
- Checkpoint colors are hard to distinguish.

**Repair**

- Generate figures as PDF vector graphics. Avoid PNG unless a raster asset is
  intrinsically required.
- Increase source font sizes, line widths, and marker sizes in the plotting
  script; do not only enlarge the image in LaTeX.
- Split a four-token sweep into two-token pages when necessary. Keep the same
  y-scale and color/legend mapping across the pair so comparison remains valid.
- Put bar values above bars and use vivid, perceptually distinct checkpoint colors.
- State the measured shape, hardware/topology, metric, and comparison scope in
  the subtitle or a concise caption.

### 8. Inconsistent transition pages, footers, and page hierarchy

**Symptoms**

- Section titles sit too high, too low, or wrap unexpectedly on a transition page.
- The footer's left label fails to align with the title margin.
- A Sources/provenance line appears as visual noise at every slide bottom.

**Repair**

- Use one transition template and place its single-line title at a stable,
  upper-right position. Keep a clickable Contents slide after each top-level
  section change.
- Keep page number, footer label, and wordmark inside fixed reserved widths. Align
  the footer label to the same left grid as frame titles.
- Keep detailed source paths in the source/Markdown record or an optional Sources
  slide, not in the visible footline of ordinary slides.

### 9. Content correctness hidden by attractive visuals

**Symptoms**

- A simplified cost model changes the technical claim.
- A diagram uses a plausible but incorrect kernel/dataflow name.
- A microbenchmark result is presented as an end-to-end causal gain.

**Repair**

- Separate source-level behavior, profiler measurements, isolated benchmarks,
  and end-to-end results. State the metric and scope on every result slide.
- Re-read the implementation before drawing a pipeline or quoting a kernel.
- Preserve key caveats in concise prose: latent MoE reduces exchange payload
  rather than expert work; dense metadata representation and all-gather backend
  are independent choices.

## Exit criteria

The deck is ready only when:

- the final PDF has been rebuilt from the current source;
- no visible frame has overflow, clipping, footer collision, or accidental wrap;
- diagrams are source-faithful, vector-based, and readable at presentation scale;
- plots expose all requested data with readable axes/legends;
- typography, headings/rules, and paragraph rhythm are consistent; and
- an independent visual review reports no concrete defect.
