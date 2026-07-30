# Making Sparser MoE Practical — Beamer deck

This directory contains the complete internship final presentation. It is a
40-page, 16:9 XeLaTeX Beamer deck built from the adjacent NVIDIA-light template.
The deck covers the 2,304-expert Qwen3-Next stress model, fused-router algorithm
and kernel work, compact routing metadata, HybridEP sparse-path optimization and
tuning, Megatron Core integration, and matched MoE-only and full-model results.

`BRIEF.md` records the requested narrative and evidence constraints.
`../sparser_moe_final_presentation_content.md` is the persistent technical
content source.

## Build

The project uses XeLaTeX so it can access the managed NVIDIA Sans fonts.

```sh
cd notes/final_presentation/beamer_presentation
make
open build/main.pdf
```

Regenerate all result plots before building:

```sh
make plots
make
```

The plotting script reads the checked-in CSV snapshots under `data/` and writes
PDF and PNG assets under `assets/generated/`. It expects the repository virtual
environment at `../../../.venv`; override `PYTHON` if needed:

```sh
make plots PYTHON=/path/to/python
```

Use `make clean` to remove LaTeX build products and `make watch` while editing.
If managed NVIDIA Sans is unavailable, the theme falls back to Helvetica Neue.

## Data and claim scope

- Router plots use matched B300 trace-back and P3R records.
- HybridEP plots separate controlled eight-GPU microbenchmarks from historical
  isolated experiments.
- Full-model plots use the staged Qwen3-Next image-sweep medians.
- MoE-only plots use the canonical matched matrix with full graph and paged
  stash enabled for both baseline and optimized variants.
- Theoretical B300 roofs are explicitly labeled as optimistic references, not
  achieved performance.

The deck deliberately does not add isolated kernel speedups to predict training
throughput. Each slide footer identifies the relevant local record, PR, or
official model/hardware source.

## Layout source

The title, transition, and logo assets in `assets/` came from the supplied
NVIDIA PowerPoint template or its canonical PDF export. NVIDIA names and marks
remain subject to NVIDIA brand and trademark rules; this is a visual adaptation,
not an official template.
