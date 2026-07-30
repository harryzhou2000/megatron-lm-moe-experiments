# Sparser MoE Final Presentation Build Brief

## Goal

Create an NVIDIA-light LaTeX Beamer deck for the internship final presentation.
Use `notes/final_presentation/sparser_moe_final_presentation_content.md` as the
persistent technical source of truth, then update claims with the newer datasets
listed below.

## Intended narrative

1. Introduce MoE architecture basics, representative current open MoE models,
   and the trend toward sparser MoE enabled by latent MoE.
2. Establish the main test model: Qwen3-Next-derived latent MoE with hidden
   dimension 512, 2,304 experts, top-k 36, and EP72.
3. Present the initial timeline analysis:
   - CPU/launch overhead from many small kernels;
   - fused-router degradation at large expert count and top-k;
   - expert-parallel communication overhead.
   Explain CPU-overhead mitigations: larger MBS, full-iteration CUDA Graph,
   fused grouped MLP and related MoE kernels, and fused metadata/dispatcher
   preprocessing. Router and communication bottlenecks motivate the following
   optimization sections.
4. Explain fused-router work by optimization stage:
   - initial large-top-k radix support;
   - P3/P3R loop fusion, async loading, persistent grid, packed radix histogram,
     static score dispatch, and backward improvements;
   - compact dense routing-map output and backward support.
   Show matched trace-back comparisons and detailed P3R progression. Add
   reference bounds for B300 HBM bandwidth and top-k CUDA-core execution.
5. Explain HybridEP tuning and optimization:
   - ballot-based permute/unpermute;
   - expert-local probability transfer;
   - combine-stage/group/batch/SM tuning;
   - int16 dense route input and bitset scan;
   - custom all-gather only if its isolated result is positive and clearly
     scoped.
   End with the controlled eight-GPU microbenchmark.
6. Explain Megatron Core integration:
   - paged stash, full-iteration graph capture, 1f1b overlap, and correct memory
     release are colleagues' work and must be attributed as such;
   - this work contributes dense-routing capability detection and plumbing
     across TE, Megatron Core, and HybridEP.
7. Close with canonical MoE-only and/or full-model results for large-EP,
   large-expert workloads. Include the 2,304-expert EP72 and 2,048-expert EP64
   Qwen3-Next models, showing both full-stack and staged gains.

## Required evidence

- `data/trace-back-comparisons/`
- `data/router_fix_p3R_*.csv` and related `data/` router records
- `notes/final_presentation/sparser_moe_final_presentation_content.md`
- `/Users/harry/projects/agentic-mcore-dev/notes/moe_perf_canonical_matrix_20260730/README.md`
- `/Users/harry/projects/agentic-mcore-dev/notes/qwen3_next_2606_image_sweep_20260728/README.md`
- `/Users/harry/projects/agentic-mcore-dev/notes/qwen_tflops_gain/`

Generate new plots when necessary. Match the green NVIDIA bar-chart style used
by `qwen_tflops_gain`, keep baseline/optimized semantics consistent, and label
hardware, benchmark scope, warmup, and metric.

## Content constraints

- Keep one claim per slide and use takeaway-style titles.
- Use top-level Beamer sections and insert a current-section roadmap after each
  section transition, including the appendix.
- Distinguish isolated kernel results, staged stack comparisons, MoE-only
  measurements, and full-model training results.
- Do not present historical snapshots as controlled additive ablations.
- Do not claim theoretical bandwidth or execution bounds as achieved results.
- Omit unrelated multimodal, MIMO, and pipeline-parallel work.
- Keep failed ideas only when they explain a design decision.
- Cite local records, PRs, model papers/cards, and benchmark methodology.

## Deliverables and QA

- Deck source and assets under `notes/final_presentation/beamer_presentation/`.
- Compiled `notes/final_presentation/beamer_presentation/build/main.pdf`.
- All plots reproducible from scripts and local data.
- Render every page to an image, inspect every slide at full size, and fix
  clipping, overlap, unreadable labels, inconsistent legends, and source text.
