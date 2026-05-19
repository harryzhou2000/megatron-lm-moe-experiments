# Internship Midpoint Progress: Sparser MoE on MCore

## Summary

My internship project has focused on exploring and optimizing a very sparse MoE training configuration in MCore/Megatron-LM. The target configuration is based on a Qwen3-Next-style architecture for GB200 NVL72 systems, with EP72, 2304 total experts (32 local experts per rank), topk=36, and latent MoE hidden dimension 512.

At the midpoint, the project has moved from initial benchmarking and bottleneck identification to concrete kernel optimizations in Transformer Engine and DeepEP. The most important full-model throughput progression is: before the fused router optimization, the EP72 Sparser MoE run was around 92 TFLOP/s; after the merged fused router optimization, it reached about 158 TFLOP/s; with the later HybridEP sparse optimizations and related stack improvements, the current best result is 320+ TFLOP/s.

## Project Context

The initial exploration started from Robin Zhang's EP72 Sparser MoE model configuration and early profiling showed that EP overhead, rather than GEMM alone, was the main limiter. The dominant problem areas became TE fused router kernels at large expert count and large topk, HybridEP dispatch/combine/allgather/permute/unpermute behavior for small latent-MoE tokens, and the interaction between MoE 1F1B overlap, full-iteration CUDA graph capture, and PyTorch CUDA memory semantics.

## Major Contribution 1: TE Fused Router Optimizations

I first optimized Transformer Engine's fused router path for large expert count and topk. The initial upstreamable work replaced the original repeated max-scan topk path with radix selection and other forward-kernel improvements. This fixed the large-E/topk performance degradation in kernels used by `FusedTopkScoreFunction` and `FusedComputeScoresForMoEAuxLoss`.

This merged TE PR was not only useful for the Sparser MoE exploration. It was also required by Xiaowei Ren's Nemotron performance work, making the first router optimization an upstream contribution with immediate value outside my local benchmark.

Impact:

- TE PR merged: <https://github.com/NVIDIA/TransformerEngine/pull/2821>
- For the Sparser MoE config (2304 experts, topk=36), the forward router kernels improved by more than 10x.
- The full-model training result improved from roughly 92 TFLOP/s before the fused router optimization to about 158 TFLOP/s after it.

I then continued with a second phase of fused router optimization on the `hhanyu/router_fix_p3R` branch. This work targets both forward and backward kernels, including fused preprocess/backward loops across topk and aux-loss paths, reduced backward shared-memory use by eliminating large temporary buffers, async global-to-shared loading with double buffering, persistent-grid launch sizing and vectorized output helpers, radix histogram/register-pressure improvements, and static score-function template paths to eliminate dynamic score dispatch.

Measured effective bandwidth improvements over the router P2 baseline include:

| Kernel | Pass | Config | Improvement |
| --- | --- | --- | --- |
| topk | fprop | 2304/36 | +32.8% |
| topk | bprop | 2304/36 | +261.7% |
| aux_loss | fprop | 2304/36 | +39.9% |
| aux_loss | bprop | 2304/36 | +104.5% |

This second phase is still in development and validation, but it has already shown that the fused router backward path can be made much more competitive for large expert counts.

## Major Contribution 2: HybridEP Sparse Optimizations

After the first router work, profiling showed that HybridEP became the major bottleneck. The target latent-MoE case has small token payloads (H=512) and very sparse local expert activity: topk=36 over 8 ranks means only about 4.5 active local experts per token on average, while several kernels still scanned all 32 local expert slots.

The HybridEP work on the `hhanyu/hybrid-ep-sparse-opt` branch addressed this from several directions: reducing dispatched/combined probability traffic to the expert-rank-local part instead of transferring the full expert set, adding dense routing-map paths based on `topk_idx` for sparse MoE, tuning dispatch/combine SM counts and combine pipeline parameters, adding ballot/ffs skipping in permute and unpermute so the kernels visit only active local experts, and investigating TMA-based NVL allgather and allgather/scan behavior for large-E sparse routing maps.

Representative microbenchmark results:

| Metric | Baseline | Sparse-Opt | Speedup |
| --- | ---: | ---: | ---: |
| dispatch kernel, E=32 TOPK=36 | 185.1 us | 103.4 us | 1.79x |
| permute kernel, E=32 TOPK=36 | 393.6 us | 94.4 us | 4.17x |
| unpermute kernel, E=32 TOPK=36 | 267.3 us | 131.6 us | 2.03x |
| dispatch+permute API | 599.8 us | 218.0 us | 2.75x |

With the documented tuned combine configuration, the sparse-opt path achieved:

- dispatch+permute: 218 us, about 2.8x faster than the tuned HybridEP baseline.
- combine+unpermute: 399 us, about 1.3x faster than the tuned HybridEP baseline.
- Total dispatch/combine path: 617 us vs 1134 us, about 1.84x end-to-end.

Early HybridEP work gave a 1.3x TFLOP/s gain on the NVL72 full model, and the refined sparse optimizations later moved the full 2304-expert EP72 setup from the post-router 158 TFLOP/s level to the current best of 320+ TFLOP/s.

## Major Contribution 3: Record-Stream / CUDA Graph Investigation

I also investigated a memory blow-up when combining MoE 1F1B overlap with full-iteration CUDA graph capture. The observed behavior was that combined 1F1B with full-iteration CUDA graph used roughly 190 GB to 265+ GB per GPU in cases where eager 1F1B used about 108 GB to 110 GB.

The investigation traced the inflation to PyTorch CUDA allocator behavior around `record_stream()`: during CUDA graph capture, blocks tagged with cross-stream uses can become deferred and unavailable for reuse inside the graph private pool. I worked through why `record_stream()` is still semantically necessary in eager multi-stream execution, and why blindly removing it is unsafe.

This work is now best treated as a diagnosis and negative result rather than a custom optimization to carry forward. After further verification, the custom record-stream removal path was deemed unnecessary because the correct PyTorch option can address the practical issue without taking on risky allocator semantics changes. The value of the work was in identifying the root cause, separating eager-mode correctness from CUDA-graph memory behavior, and avoiding an unsafe patch.

## Additional Investigations and Negative Results

Several investigations produced useful "do not do this for the target workload in its current form" results, while still leaving clear future optimization directions.

Direct-permute dispatch attempted to skip the local permute kernel by writing directly into expert-grouped output buffers over NVLink. The implementation and NCU analysis showed this is not favorable for the target H=512, topk=36 case: direct writes introduce K-times NVLink write amplification, scattered writes, metadata overhead, and an L2 capacity cliff. For H=512, K=8 was already 2.2x slower than non-direct dispatch plus permute; for K=36 the path is clearly uneconomical. It may still be worth revisiting for larger hidden dimensions, much smaller topk, or a design that avoids K-times remote write amplification.

Fused dispatch+permute and fused combine+unpermute were also investigated. For latent MoE H=512, the standalone non-fused path is faster because fused mode suffers from SM allocation constraints, chunk-flag polling, and low useful work per block. The fused direction remains a potential optimization point if the architecture can be changed to allocate SMs more effectively, reduce polling overhead, or target larger hidden dimensions where per-token overhead is better amortized.

The combine kernel tuning work also clarified current limits. For H=512, the tuned combine kernel reaches about 246 us / 271 GB/s output bandwidth, with remaining cost dominated by per-token barriers, shared-memory movement, and TMA issue overhead. For H=7168, combine reaches about 675 GB/s and is primarily limited by G2S TMA issue rate. Further improvement likely requires a deeper kernel restructuring rather than only parameter tuning.

## Collaboration and Communication

I have been communicating progress regularly with Chandler Zhou, Tong Liu, Robin Zhang, Jiajie Yao, Dennis Liu, and others. The work has also involved technical discussion with domain experts, including Robin Zhang for the original EP72 Sparser MoE benchmark setup and direct dispatch ideas, Tong Liu for HybridEP and dense routing-map direction, Xiaowei Ren for the Nemotron performance requirement motivating the merged TE router PR, and Pingtian Li for discussion of `record_stream()` semantics.

The TE router work has already produced an upstream merged PR, and the ongoing router P3 and HybridEP sparse branches are structured as follow-on optimization work that can be reviewed and integrated once validation is complete.

## Self-Evaluation Framing

At midpoint, I would summarize my impact as taking ownership of a performance critical MoE routing/communication path, moving it from profiler observations to measured kernel improvements and an upstream TE contribution. The clearest project-level outcome is that the target Sparser MoE training run improved from around 92 TFLOP/s before fused router optimization, to about 158 TFLOP/s after the merged TE router PR, and then to 320+ TFLOP/s with the later HybridEP sparse optimizations and full-stack improvements.

The strongest parts of the work are:

- I identified the right bottlenecks for the target workload instead of only optimizing generic MoE kernels.
- I produced an upstream TE change required by both the Sparser MoE exploration and Xiaowei Ren's Nemotron performance work.
- I continued beyond the merged PR into deeper forward/backward fused-router optimization work.
- I delivered HybridEP sparse optimizations that materially improved both microbenchmarks and full-model NVL72 throughput.
- I used profiling to reject attractive but uneconomical ideas, such as direct-permute for H=512/topk=36, and turned those negative results into constraints for future designs.
- I connected microbenchmark improvements to full-model NVL72 throughput, with the overall trajectory moving from 92 TFLOP/s to 320+ TFLOP/s.

The main remaining risks are integration and validation. The second TE router optimization phase and the DeepEP sparse-opt path need more correctness testing, full-model benchmarking, and cleanup before they are ready to upstream or become default paths.

## Next Steps

- Finish validating the TE fused router P3 changes, especially backward correctness and performance across topk/expert-count regimes.
- Package the HybridEP sparse-opt changes into reviewable patches, separating probability pruning, dense routing-map handling, permute/unpermute skipping, and tuning knobs.
- Continue EP72/NVL72 full-model benchmarking to quantify which microbenchmark wins survive under 1F1B overlap and CUDA graph capture.
- Investigate remaining allgather/device-sync overheads in the full NVL72 run.
- Revisit direct-permute as a conditional optimization for larger H or smaller topk, while avoiding the K-times NVLink write amplification observed for H=512/topk=36.
- Revisit fused dispatch+permute and fused combine+unpermute as future optimization points, focusing on SM allocation, flag polling, and whether a different fused architecture can beat the standalone path.
- Evaluate whether a MegaMoE-style megakernel is applicable to the sparser MoE setup, using the negative fused-kernel and direct-permute findings as design constraints.

## Questions

### Q1: Overall

I am progressing well against the goals identified at the start of the internship. The project started from benchmarking and understanding the Sparser MoE training configuration in MCore/Megatron-LM, then moved into concrete optimization work in Transformer Engine and DeepEP. My key accomplishments so far are the merged TE fused router optimization, continued fused-router forward/backward optimization work, HybridEP sparse-routing optimizations, and a record-stream/CUDA graph memory investigation that clarified which path should not be pursued. In terms of training performance, the target EP72, 2304-expert, topk=36 setup improved from around 92 TFLOP/s before fused router optimization, to about 158 TFLOP/s after the merged fused router optimization, and then to 320+ TFLOP/s with the later HybridEP sparse optimizations and related full-stack improvements.

### Q2: Raise the Bar

The work has been challenging and valuable because it requires understanding performance across multiple layers of the training stack: TE router kernels, DeepEP dispatch/combine/allgather behavior, MCore training overlap, CUDA graph capture, PyTorch CUDA memory semantics, and full NVL72 system behavior. The challenges already met include identifying the correct bottlenecks for a large-E/topk sparse MoE configuration, turning profiling observations into kernel changes, upstreaming a TE performance fix, tuning communication kernels for small H=512 payloads, and using NCU results to reject approaches that looked attractive but were not profitable for the target workload. The future challenges are also valuable for my growth: validating and upstreaming the remaining router and HybridEP changes, understanding which microbenchmark wins survive full training overlap, and exploring deeper kernel restructuring or megakernel-style designs where simple tuning is no longer enough.

### Q3: Development Opportunities

I discussed development opportunities with my mentor, Robin Zhang, through regular 1:1s. The current feedback is that the Sparser MoE optimization work is progressing well and that the next steps have been established. My action plan is to continue the current optimization efforts and move through the next-step plan: finish validating the TE fused router P3 changes, package the HybridEP sparse optimizations into reviewable pieces, continue full-model NVL72 benchmarking, investigate remaining allgather/device-sync overheads, and revisit the negative-result directions as conditional future optimization points under better-suited regimes.

<!-- 
## References

- Google Drive: Harrys Copy of Sparser MoE, <https://docs.google.com/document/d/1iRopu2nZdLAUNSmLzGAjHYTIESVbuQ7uKsXyyYMIFO4>
- Google Drive: TE Fused Router Optimization, <https://docs.google.com/document/d/1oFisyasi469EG_3ExL4LF0ioIru6Hy8UV0JS2UGVruo>
- Confluence: Top 5 Things 2026/03/19, 2026/04/03, 2026/04/20, 2026/05/11.
- Outlook sent mail: Top 5 Things - Devtech Compute - Sparser MoE, including the 2026/05/11 "Sparser MoE Optimizations" update.
- Slack threads: Xiaowei Ren / Nemotron performance context for the merged TE fused router PR, referenced by the user.
- Local notes: `notes/topk_p3_optimizations.md`, `notes/hybrid-ep-sparse-opt-new.md`, `notes/record_stream_removal.md`, `notes/direct_permute_integration_design.md`, and `notes/direct_permute_ncu_analysis.md`.
- Local branches: TE `hhanyu/router_fix_p3R`; DeepEP `hhanyu/hybrid-ep-sparse-opt` and related direct-permute work. 
-->
