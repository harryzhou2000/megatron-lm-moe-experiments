# Transformer Engine activation build bottleneck

## Scope

This note measures the Transformer Engine PyTorch build bottleneck on Blackwell
and evaluates whether splitting GELU/QGELU template instantiations across CUDA
translation units improves wall time. The experiment uses a dedicated branch
from upstream `main`; it does not change activation math, API symbols, kernel
launches, or runtime behavior.

## Environment and method

- Container: `dev_2604`
- GPU allocation: one B200; the GPU is not material to host compilation
- CUDA architecture request: `NVTE_CUDA_ARCHS=100`
- Parallelism: `MAX_JOBS=4`, `NVTE_BUILD_THREADS_PER_JOB=4`
- Framework: PyTorch only
- NCCL EP build: disabled
- Build command: editable `pip install --no-build-isolation -e . --verbose`
- Measurement source: CMake/Ninja `.ninja_log`
- Cache policy: a separate empty ccache directory for each timed cold build
- CPU sampling: `vmstat` every 30 seconds

Although `NVTE_CUDA_ARCHS=100` was requested, TE's activation build emitted
architecture-specific `sm_100a` and `sm_103a` code in addition to generic
Blackwell code. The comparison uses identical flags and container state.

## Baseline

The upstream build completed in 2,125 seconds (35m25s). Ninja's final link
completed at 1,834.128 seconds after Ninja start. The 97 object translation
units accounted for 6,430.624 compiler-seconds of work.

Activation sources contributed 4,073.207 compiler-seconds, or 63.3% of all
object work. GELU/QGELU sources alone contributed 1,674.135 compiler-seconds,
or 26.0% of all object work and 41.1% of activation work.

| Baseline translation unit | Compile time |
|---|---:|
| `gelu_grouped.cu` | 883.934 s |
| `swiglu_grouped.cu` | 501.608 s |
| `relu_grouped.cu` | 496.940 s |
| `fused_topk_with_score_function.cu` | 373.474 s |
| `gelu_grouped_dbias.cu` | 318.308 s |
| `gelu.cu` | 293.479 s |
| `swiglu.cu` | 290.616 s |
| `relu.cu` | 238.473 s |
| `gelu_dbias.cu` | 178.414 s |
| `scaled_activation.cu` | 170.529 s |

The ideal four-job lower bound from total object work is 1,607.656 seconds,
about 226 seconds below the observed final-link time. Activation work ended at
1,286.777 seconds, versus an activation-work lower bound of 1,018.302 seconds.
This identifies long-tail scheduling, not aggregate CPU capacity, as the main
opportunity.

CPU samples showed 224 logical CPUs, no swap or I/O wait, and approximately
94--96% aggregate idle time. Individual compiler processes saturated their
cores, but four top-level jobs left most host CPUs idle.

## Diagnostic all-GELU split

The first diagnostic separated forward/backward, GELU/QGELU, grouped, and
grouped-dbias instantiations into 12 translation units. The build later stopped
in an unrelated grouped-GEMM source because that remote test tree lacked the
CUTLASS `bfloat16.h` submodule file. All activation translation units had
already completed, so their individual timings remain useful.

| Diagnostic translation unit | Compile time |
|---|---:|
| `gelu.cu` | 160.134 s |
| `gelu_bwd.cu` | 173.410 s |
| `gelu_dbias.cu` | 151.143 s |
| `gelu_grouped.cu` | 241.072 s |
| `gelu_grouped_bwd.cu` | 159.226 s |
| `gelu_grouped_dbias.cu` | 165.393 s |
| `qgelu.cu` | 160.929 s |
| `qgelu_bwd.cu` | 173.309 s |
| `qgelu_dbias.cu` | 150.041 s |
| `qgelu_grouped.cu` | 330.033 s |
| `qgelu_grouped_bwd.cu` | 162.878 s |
| `qgelu_grouped_dbias.cu` | 168.843 s |

The split GELU family totaled 2,196.411 compiler-seconds, 31.2% more work than
the baseline, but its last translation unit finished 228.345 seconds earlier.
The family-level breakdown explains the tradeoff:

| Family | Baseline | Split | Change |
|---|---:|---:|---:|
| Grouped forward/backward | 883.934 s | 893.209 s | +1.0% |
| Grouped dbias | 318.308 s | 334.236 s | +5.0% |
| Non-grouped forward/backward | 293.479 s | 667.782 s | +127.5% |
| Non-grouped dbias | 178.414 s | 301.184 s | +68.8% |

Splitting grouped instantiations exposes useful parallelism with little parsing
duplication. Splitting the smaller non-grouped files duplicates template/header
work much faster than it improves the schedule.

## Grouped-only candidate

The measured candidate therefore keeps `gelu.cu` and `gelu_dbias.cu` unchanged
and splits only the grouped GELU/QGELU symbols into six translation units:

- grouped GELU forward, backward, and dbias;
- grouped QGELU forward, backward, and dbias.

The six sources are adjacent and early in the architecture-specific CMake list
so Ninja can fill four job slots before later activation families. The same
split is reflected in the optional fast-math source list.

## Grouped-only cold-build result

The grouped-only candidate built successfully, but it did not improve wall
time.

| Metric | Baseline | Grouped-only split | Change |
|---|---:|---:|---:|
| Full editable build | 2,125 s | 2,194 s | +69 s (+3.2%) |
| Common-library Ninja link | 1,834.128 s | 1,912.939 s | +78.811 s (+4.3%) |
| Object translation units | 97 | 101 | +4 |
| Total object work | 6,430.624 s | 6,928.249 s | +497.625 s (+7.7%) |
| Activation work | 4,073.207 s | 4,343.510 s | +270.303 s (+6.6%) |
| GELU/QGELU work | 1,674.135 s | 1,818.579 s | +144.444 s (+8.6%) |
| Last GELU/QGELU completion | 884.008 s | 542.453 s | -341.555 s (-38.6%) |

| Grouped-only candidate translation unit | Compile time |
|---|---:|
| `gelu_grouped.cu` | 260.831 s |
| `gelu_grouped_bwd.cu` | 168.803 s |
| `gelu_grouped_dbias.cu` | 173.663 s |
| `qgelu_grouped.cu` | 356.264 s |
| `qgelu_grouped_bwd.cu` | 177.293 s |
| `qgelu_grouped_dbias.cu` | 183.340 s |

These six objects used 1,320.194 compiler-seconds, 9.8% more than the
1,202.242 seconds used by the two original grouped objects.

The split achieved its local scheduling goal: all grouped GELU/QGELU objects
finished 342 seconds earlier. However, repeating the large activation template
headers increased total compiler work enough to delay later sources. The fused
Top-K router became the final straggler, starting at 1,527.553 seconds and
finishing at 1,909.952 seconds after 382.399 seconds of compilation. The final
link followed at 1,912.939 seconds.

The cold cache recorded zero hits. CPU samples remained approximately 97--99%
idle with no swap or I/O wait, confirming that the result was governed by the
four-job dependency schedule and long single-TU CUDA compilation rather than
machine-wide CPU saturation.

## Correctness validation

The built editable package imported from the candidate source, and
`libtransformer_engine.so` exported all six unchanged grouped activation APIs:

- `nvte_group_gelu` and `nvte_group_qgelu`;
- `nvte_group_dgelu` and `nvte_group_dqgelu`;
- `nvte_group_quantize_dbias_dgelu` and
  `nvte_group_quantize_dbias_dqgelu`.

A minimal CMake harness compiled the existing grouped-MXFP8 C++ test source
against the candidate library. The upstream GELU matrix passed 183 cases with
162 expected shape/method skips. A temporary test-only copy enabled the
existing QGELU cases in the same source and likewise passed 183 cases with 162
expected skips. Neither run had a failure. The candidate also passed the
focused pre-commit checks and `git diff --check` before the build.

## Four-job conclusion

The grouped-only split preserves behavior and shortens the GELU family phase,
but it makes the measured cold build slower. It should not be proposed as-is.
The experimental worktree remains isolated on
`hhanyu/activation-build-split`; its changes are intentionally uncommitted
because the measured result is negative. No runtime source or public API change
depends on it. A future build-only experiment should prioritize the longest
known stragglers earlier in Ninja's ready queue or reduce repeated template
parsing rather than merely adding more translation units.

## Machine-parallel grouped activation split

The follow-up experiment used upstream `main` commit `e6690f47` and expanded
the split to every independently parallel instantiation in five measured
sources:

- `gelu_grouped.cu`: GELU/QGELU forward and backward;
- `gelu_grouped_dbias.cu`: dGELU/dQGELU with dbias;
- `relu_grouped.cu`: ReLU/SReLU forward and backward;
- `relu_grouped_dbias.cu`: dReLU/dSReLU with dbias;
- `swiglu_grouped.cu`: SiLU forward and backward.

`swiglu_grouped_dbias.cu` remained unchanged because it contains only one
dSiLU+dbias instantiation. The five original sources became fourteen sources,
increasing the common-library object count from 97 to 106 without changing any
public symbol or runtime dispatch.

The build ran on a full-node GCP-NRT allocation in `dev_2604`. The allocation
exposed 224 logical CPUs and approximately 3.8 TiB of memory. `MAX_JOBS` and
`CMAKE_BUILD_PARALLEL_LEVEL` were both unset, so CMake invoked plain
`ninja -v`; `NVTE_BUILD_THREADS_PER_JOB=4` remained enabled. Both builds used
`NVTE_CUDA_ARCHS=100`, separate empty ccache directories, the same venv and
submodules, and zero cache hits. The split ran first, making any later system
cache warming favor the baseline.

| Metric | Baseline | Split | Reduction | Speedup |
|---|---:|---:|---:|---:|
| Full editable build | 1,038.390 s | 551.548 s | 46.9% | 1.88x |
| Common-library Ninja link | 935.905 s | 448.261 s | 52.1% | 2.09x |
| Maximum TU | 932.579 s | 445.884 s | 52.2% | 2.09x |

The split increased the object count from 97 to 106 and aggregate compiler work
by 13.7% (11,003.431 to 12,516.317 compiler-seconds). Activation compiler work
increased by 20.5% (4,998.527 to 6,023.510 compiler-seconds).

The baseline maximum was `gelu_grouped.cu` at 932.579 seconds. After the split,
the maximum became the unrelated `fused_topk_with_score_function.cu` at
445.884 seconds. The largest split activation objects were:

| Split translation unit | Compile time |
|---|---:|
| `swiglu_grouped.cu` (SiLU forward) | 411.921 s |
| `qgelu_grouped.cu` | 410.251 s |
| `gelu_grouped.cu` | 327.930 s |
| `srelu_grouped.cu` | 260.692 s |
| `relu_grouped.cu` | 251.699 s |
| `qgelu_grouped_dbias.cu` | 250.840 s |
| `qgelu_grouped_bwd.cu` | 245.843 s |
| `gelu_grouped_dbias.cu` | 243.067 s |
| `swiglu_grouped_bwd.cu` | 240.844 s |
| `gelu_grouped_bwd.cu` | 236.365 s |

The split traded more aggregate compiler work for parallel critical-path
reduction. This was harmful with four top-level jobs but beneficial with
machine parallelism: common-library time improved by 2.09x and full editable
build time by 1.88x. During the machine-parallel builds, `vmstat` observed a
maximum runnable queue of 108 for the split and 86 for the baseline, zero I/O
wait, and peak SLURM-reported RSS of approximately 70 GB and 66 GB,
respectively.

## Machine-parallel correctness validation

The split library exported all fourteen moved GELU/QGELU/ReLU/SReLU/SiLU
forward, backward, and applicable dbias APIs. A test-only expansion of the
existing grouped-MXFP8 C++ parameter matrix enabled all five activation
families. It ran 1,725 cases: 915 passed, 810 incompatible shape/method cases
were skipped by existing test guards, and none failed. Focused pre-commit hooks
and `git diff --check` also passed.

## Final conclusion

The grouped activation split is worthwhile for TE builds that allow Ninja to
use full machine parallelism. The earlier negative result was specific to
`MAX_JOBS=4`, where duplicated template parsing serialized later work. Under
the requested machine-max policy, the same design removes the dominant
monolithic GELU critical path and nearly halves end-to-end build time. The
validated implementation is isolated on
`hhanyu/grouped-activation-tu-split`.
