# Evidence notebook: work since 2026/06/29

This directory records the evidence used to prepare the next Top 5 Things entry and the
full Sparser MoE retrospective. It is intentionally more explicit than the final T5T so
that performance scope, PR state, and historical-vs-controlled comparisons remain auditable.

Evidence snapshot date: **2026/08/26**.

## Requested deliverables

1. `notes/top5_things_2026_08_26.md`: concise update since the 2026/06/29 T5T.
2. `notes/sparser_moe_full_work_summary.md`: detailed, start-to-finish Sparser MoE record
   with links and embedded figures.

## Sources inspected

### Prior T5T and presentation material

- [Top 5 Things 2026/06/29](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/top5_things_2026_06_29.md)
- [Confluence Top 5 transcript](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/top5_confluence_transcript.md)
- Live Confluence page: https://
- [Final Sparser MoE presentation content](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/sparser_moe_final_presentation_content.md)
- [Internship midpoint Sparser MoE report](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/internship_midpoint_sparser_moe_progress.md)

### Primary Sparser MoE technical records

- [TE fused-router optimization](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_optimization.md)
- [Final TE fused-router results](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/te_fused_router_p3R_results.md)
- [HybridEP sparse optimization and NCU analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid-ep-sparse-opt-new.md)
- [Rebased HybridEP correctness and bandwidth](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/hybrid_ep_rebased_bw_results.md)
- [Dense scan benchmarks](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/isolated-scan-bench.md)
- [Direct-permute NCU analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/direct_permute_ncu_analysis.md)
- [NVL72 CUDA-graph and skew analysis](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/reports/nvl72_cg_vs_nocg_report.md)
- [CUDA-graph `record_stream()` investigation](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/record_stream_removal.md)
- July MoE-only benchmark evidence: the validated measurements are reproduced in the
  tables below; raw environment and job provenance is intentionally not republished.
- July 30 canonical GB200/GB300 matrix evidence: the matched results are reproduced below
  with their full-model versus module-only scope preserved.

### Kimi K3 and multimodal records

- [Kimi K3 Stable LatentMoE implementation](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/kimi_k3_moe_implementation.md)
- [K3 Quantile Balancing router histogram results](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/945e8fe0091b315e7328111fff9877cab4eb65df/notes/kimi_k3_qb_router_histogram_results.md)
- Local and `nvhzdc7` history catalogs in `~/projects/summarize_agents`, queried for
  Sparser MoE, HybridEP, Kimi K3, Quantile Balancing, SiTU-GLU, latent RMSNorm,
  Qwen3.5 mock vision, and variable-length records. Only work outcomes are carried into
  the T5T; history/tool details are deliberately omitted.

## Delta since the 2026/06/29 entry

The June 29 page named TE router PRs #3012 and #3129 and left downstream integration as
follow-through work. The later work closed or advanced the remaining layers:

- DeepEP dense top-k scan https:// was
  created on 2026/06/29 and merged on 2026/08/05.
- MCore dense-route propagation https://
  (`dev`) and https:// (`main`) were
  opened on 2026/08/18. Both are open and mergeable at this snapshot.
- TE https:// and
  https:// were already merged
  before the June 29 page. Do not describe them as newly merged after June 29; describe
  them as the completed TE layer of the now-wrapped stack.

## Sparser MoE performance evidence

### Historical target progression

| Stack snapshot | Approx. TFLOP/s/GPU | Scope |
| --- | ---: | --- |
| Initial profile | ~92 | Historical 2,304E/top-36/EP72 run |
| Initial radix router | ~158 | Historical evolving stack |
| Refined HybridEP sparse stack | ~310 | Historical evolving stack |
| Later dense scan/bitset stack | ~400 | Historical evolving stack |

These values show the bottleneck moving across the system. They are **not** a controlled
92-to-400 ablation and must not be presented as one.

### Matched full-model results

Median TFLOP/s/GPU, paged stash + full-iteration CUDA graph + expert 1F1B:

| System | Shape | No tune/no radix | Full sparse stack | Gain |
| --- | --- | ---: | ---: | ---: |
| GB200 | 2,304E / EP72 | 125.3 | 403.7 | 3.22x |
| GB200 | 2,048E / EP64 | 159.4 | 421.0 | 2.64x |
| GB300 | 2,304E / EP72 | 127.2 | 419.7 | 3.30x |
| GB300 | 2,048E / EP64 | 162.2 | 435.1 | 2.68x |

The stage data also separate tuned HybridEP, initial router #2821, deeper router #3012,
and the full sparse stack; use the final presentation note for the complete table.

### Later MoE-module matrix

All rows completed 160 MoE performance iterations with full-iteration CUDA graph and
paged stash. This is MoE-module-only, not full-model training.

| Shape | Base | Optimized | Gain |
| --- | ---: | ---: | ---: |
| NT3 Super estimate | 511.8 | 553.65 | +8.2% |
| Nemotron Ultra proxy | 1072.0 | 1123.8 | +4.8% |
| Qwen397 EP64 | 1014.4 | 1198.3 | +18.1% |
| Qwen397 Sparser EP64 | 826.2 | 1041.5 | +26.1% |
| Qwen40 Sparser EP72 | 234.5 | 484.5 | +106.6% |

The trend, not any single row, is the main takeaway: the optimization stack matters more
as routing and expert-parallel overhead occupy more of the iteration.

## Sparser MoE contribution map

- **TE router:** radix top-k, async/double-buffered score loading, persistent launch,
  fused backward loops, packed histograms, static score dispatch.
- **Compact metadata:** optional int16 `[T,K]` selected-expert indices replace bool
  `[T,E]` route metadata for the target, a 32x logical reduction at the documented shape.
- **HybridEP traffic:** send only destination-local probability columns; do not imply
  probabilities can be removed from training.
- **Pre/post processing:** warp-ballot traversal visits selected local experts instead of
  testing every slot.
- **Combine:** coordinated search over SM count, FIFO depth, tokens/group, and reduction
  batching; combine remains structurally harder because it pulls and reduces remote data.
- **Dense scan:** local-expert and rank bitsets turn repeated sparse membership checks into
  register bit tests.
- **Negative results:** direct-permute amplifies NVLink writes; fused combine serializes
  consumers; warp-pruned scan adds more control overhead than it removes.
- **Runtime analysis:** full-iteration graph + expert 1F1B exposed CUDA allocator and
  cross-rank skew behavior. Preserve the diagnosis, but do not claim the custom allocator
  workaround as the final upstream solution.

## Kimi K3 evidence and live PR state

The implemented K3 MoE slice consists of:

1. SiTU-GLU with beta1=4 and beta2=25, including standalone/shared-expert and fused
   grouped-expert paths plus safe FP32 fallback semantics.
2. Global-batch Quantile Balancing using Top-(k+1), persistent FP32 bounds/bias,
   int32 histograms, and distributed step-finalization.
3. `RMSNorm -> W_up` after routed latent aggregation, before restoring the full hidden
   dimension; shared experts stay full-width.

Merged lower-stack PRs:

- cuDNN Frontend https:// and
  https://: grouped SiTU-GLU and API fixes.
- TE https://: QB router histogram paths.
- TE https://: SiTU-GLU activation.

Open MCore `dev`/`main` pairs:

- SiTU-GLU: https:// /
  https://.
- Quantile Balancing: https:// /
  https://.
- Latent RMSNorm: https:// /
  https://.

Validation highlights:

- TE scaled-activation matrix: 188 passed, 68 expected skips, zero failures.
- SiTU-GLU focused MCore suites: 41 passed on `dev`, 60 passed on `main`.
- QB distributed coverage included EP4, EP8, and TP4/EP2 topologies.
- Fused TE QB histogram accumulation was 4.6x-8.8x faster than the PyTorch QB reference
  in the focused B300 cases; distinguish this kernel comparison from model throughput.
- K3 MoE-only QB rows were approximately neutral-to-positive versus the selected control
  (about 1.01x in the presentation’s GB200/GB300 summary), so support/correctness is the
  primary result rather than a large throughput claim.

## Qwen3.5 empirical mock-vision replay evidence

- Strict public JSONL v1: `format_version`, `llm_sequence_length`, and exactly one of
  `vision_tokens_per_image` or `image_sizes`; empty arrays mean text-only.
- Variable-length packed THD uses real `cu_seqlens`, separate padded cumulative lengths,
  and a `padding_mask`; BSHD pads to the batch target length.
- Mixed batches remove dummy vision rows; all-text batches retain one row for eager vision
  execution.
- Eight-rank focused tests passed 38 tests per rank.
- A three-step Qwen MoE/GDN EP8 run passed with MBS4/GBS32, finite LM/load-balancing loss,
  and zero skipped or NaN iterations.
- Fork-local draft PR: https://,
  open, draft, and mergeable at this snapshot.

## Additional T5T candidate

The Qwen3-Next 2,304E/2,048E legacy launcher was ported into readable model/recipe YAML,
render-checked token-for-token against the source configuration, and incorporated into the
canonical GB200/GB300 MoE matrix. This is worth a separate short point because it turns the
one-off Sparser MoE experiment into reusable regression and performance coverage.

## Claim guardrails for final writing

- Never describe the historical 92 -> 400 sequence as a controlled speedup.
- Label MoE-module-only numbers separately from full-model training.
- Probability traffic is reduced to selected/local columns, not eliminated.
- Dense metadata and custom all-gather are independent improvements.
- TE #3012/#3129 were already merged by 2026/06/29.
- State live PR status as of 2026/08/26; do not imply open MCore work is merged.
- Describe the K3 deliverable as K3 **MoE support**, not a complete K3 model implementation.
- Do not mention coding agents, session IDs, history databases, machines, containers, or
  internal workflow details in the T5T.
