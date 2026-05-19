# Top 5 Confluence Pages Transcript

Source folder: <https://nvidia.atlassian.net/wiki/spaces/~712020753b3a7633654afeb025f6cc042701ce/folder/3184283536>

This transcript includes the Top 5 pages found in Harry Zhou's personal Confluence `Top 5 Things` folder.

## Top 5 Things 2026/03/19

Source: <https://nvidia.atlassian.net/wiki/spaces/~712020753b3a7633654afeb025f6cc042701ce/pages/3184531091>

### Last Few Weeks (2026/02/24-2026/03/19)

### Sparser MoE Discoveries

- Benchmarking using Robin Zhang's EP 72 sparser MoE model configuration
- oci-hsg with single NVL72
- TFLOP/s not ideal, EP overhead is bottleneck
- Resolving TE router kernels' performance degradation at large number of experts and topk (in forward pass).
- Fixed TE kernels used in FusedTopkScoreFunction and FusedComputeScoresForMoEAuxLoss
- Used radix selection instead of linear scan; other minor optimizations
- Kernel speedup > 10x at num_expert=2304 and topk=36
- Training TFLOP/s 1.35x gain
- Investigation of MoE router-dispatch-combine implementation
- Possible optimization points including routing map
- Some additional details:
- <https://docs.google.com/document/d/1iRopu2nZdLAUNSmLzGAjHYTIESVbuQ7uKsXyyYMIFO4/edit?usp=sharing>

### Studying Megatron Core MoE

- Reading Megatron Core MoE technical report
- Experimenting with small-scale EP examples

### These Few Weeks

- Continue Sparser MoE investigations
- Optimizing allgather logics that uses dense routing map
- Testing and try optimizing (fused) dispatch/combine (consult Tong Liu (Engrg-Hardware 1) )
- Testing full-iter cuda graph gains
- Testing 1f1b overlapping gains
- Studying Megatron Core
- Getting a full picture of mcore's structure
- Experimenting with more parallelism combinations

## Top 5 Things 2026/04/03

Source: <https://nvidia.atlassian.net/wiki/spaces/~712020753b3a7633654afeb025f6cc042701ce/pages/3222429609>

### Last Few Weeks (2026/03/20-2026/04/03)

### Sparser MoE Discoveries

- Continuing E=2304, topk=36, EP72 on NVL72 optimization
- Integrating 1f1b EP overlapping with full-iteration cuda graph
- Problem: 1f1b combined with full-iter CG causes dramatic VRAM usage
- Observation: the memory growth comes from input tensor discarding guarded with record_stream()
- record_stream() is needed semantically, discussed with Pingtian Li
- CG capture "memory leak" is expected pytorch behavior
- Solution: deferred release registry in 1f1b
- record stream and event info, only release when the allocator stream waits the right event
- results: 1f1b [108.0 GB]; 1f1b+full-iter CG [265+ GB] -> [107.8 GB]
- Using full-iter CG + 1f1b, only 1.1x TFLOP/s gain
- Bottleneck identification:
- hybrid-ep combine & dispatch
- flash_attn backward
- Additional details:
- <https://docs.google.com/document/d/1iRopu2nZdLAUNSmLzGAjHYTIESVbuQ7uKsXyyYMIFO4/edit?usp=sharing>

### Fused Router Integration to Upstream

- fused_router topk optimization required by nemotron
- <https://nvbugspro.nvidia.com/bug/6035274>
- TE: PR under review: <https://github.com/NVIDIA/TransformerEngine/pull/2821>

### These Few Weeks

- Shift optimization focus to hybrid-ep
- Sparse routing map optimization
- hybrid-ep NVL72 SOL investigation
- Consider NCCL EP integration?
- Other things
- 1f1b CG memory fix integration to upstream

## Top 5 Things 2026/04/20

Source: <https://nvidia.atlassian.net/wiki/spaces/~712020753b3a7633654afeb025f6cc042701ce/pages/3291349529>

### Last Few Weeks (2026/04/03-2026/04/20)

### Sparser MoE Discoveries

- Continuing E=2304, topk=36, EP72 on NVL72 optimization
- Optimization of Hybrid-EP with large expert number on NVL72
- Problems:
- Tokens are very small for latent MoE
- Probs are fully dispatched, that are larger than needed
- Routing map allgather -> scan uses boolean map, inefficient for large E
- Combine is latency bound within the warp streamline
- Fixes: Previously (for single-NVL-domain) probs of full expert set are broadcast in dispatch_kernel
- Smaller TMA chunk for probs: only the expert rank's probs are sent for forward dispatch (and backward combine)
- ON 8xB300 NVL 18 machine, dispatch with probs: BW 200GB/s -> 600 GB/s
- Fixes: Can opt to dense (topk_idx) routing map for sparse MoE in hybrid-ep
- Fixes: Combine uses batched free to increase TMA concurrency
- Results:
- 2-4x better for dispatch/combine kernels on 8xB300 NVL 18 machine
- NVL 72 full model: 1.3x TFLOP/s
- Problems:
- Custom NVL allgather performance problem
- hybrid-ep device sync is taking too long, root of problem not identified
- Additional details:
- <https://docs.google.com/document/d/1iRopu2nZdLAUNSmLzGAjHYTIESVbuQ7uKsXyyYMIFO4/edit?usp=sharing>

### Fused Router

- fused_router topk optimization
- <https://nvbugspro.nvidia.com/bug/6035274>
- TE: PR merged: <https://github.com/NVIDIA/TransformerEngine/pull/2821>
- further fused router optimizations:
- Restructuring code, + static score function paths and fused loops
- Add simple double-buffered vectorized global-shared load
- Radix selection using 8-bit histogram
- Result: 1.4x-3x for fprop/bprop for large expert numbers
- <https://docs.google.com/document/d/1oFisyasi469EG_3ExL4LF0ioIru6Hy8UV0JS2UGVruo/edit?usp=sharing>
- Under test

### These Few Weeks

- Further investigate hybrid-ep issues on NVL72
- Why does device sync dominate?
- What is optimal launch parameter (sms, pipeline depth ...) for overlapped/non-overlapped communication kernels?
- New optimization potentials?

## Top 5 Things 2026/05/11

Source: <https://nvidia.atlassian.net/wiki/spaces/~712020753b3a7633654afeb025f6cc042701ce/pages/3374590111>

### Last Few Weeks (2026/04/20-2026/05/11)

### Sparser MoE Discoveries

- Continuing E=2304, topk=36, EP72 on NVL72 optimization
- Optimization of Hybrid-EP with large expert number on NVL72
- Current Optimizations Refined:
- Dispatch & Combine with probs size reduced to only expert-rank-local part
- Tuned with larger queue size with smaller group size in Combine to saturate TMAs
- Permute & Unpermute optimized with ballot skipping (this should be covered better by @Tong Liu (Engrg-Hardware 1)'s dense row_id_map optimization in the future)
- Dense routing map path using topk_idx
- TMA-based NVL allgather implementation
- Other optimizations attempted:
- dispatch_direct to skip permute and write directly into target buffer
- Inspired by @Robin Zhang's suggestions.
- Need an additional d2d copy with CE;
- Current pure direct dispatch part does only achieves half of original dispatch BW
- Need further investigation
- optimizations of fused dispatch+permute
- Issues: cannot directly copy the standalone permute kernel perf due to SM allocations
- Currently cannot beat standalone dispatch+permute (and unpermute+combine)
- Current results:
- 2304 expert EP72 on GB200 NVL72
- With CuTe-DSL grouped gemm from cudnn-frontend, mxfp8
- + paged stash + full-iter CUDA graph + moe 1f1b overlapping
- + optimized TE router kernels
- TFLOP/s: 158 -> 310 using Hybrid-EP sparse optimizations
- Previous investigations:
- device sync domination was caused by cross-rank skews under 1f1b, most-likely due to comp and comm kernels contesting SMs; currently mitigated by switching to cudnn grouped gemm kernels
- Additional details:
- Harrys Copy of Sparser MoE

### These Few Weeks

- Further investigate hybrid-ep issues on EP72+NVL72
- Direct dispatch optimizations
- Does the order and continuity of TMA srcs/dsts matter?
- Discrete dispatch and combine could better SOLs
- Fused dispatch and combine for sparse tokens
- Other opportunities:
- Could sparser MoEs benefit from a megakernel like MegaMoE?
- Get Outlook for Mac
