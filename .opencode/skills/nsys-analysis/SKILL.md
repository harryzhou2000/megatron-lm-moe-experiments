---
name: nsys-analysis
description: Analyze NVIDIA Nsight Systems (.nsys-rep) profiling reports. Converts reports to SQLite via Docker, detects training iterations from kernel timeline density (works with CUDA graph replay), and generates statistics for kernels, NCCL collectives, MoE operations, and cross-rank load imbalance. Use when the user wants to analyze nsys profiles, GPU kernel timelines, or MoE training performance.
license: MIT
compatibility: opencode
---

# Nsys Report Analysis

Analyze NVIDIA Nsight Systems profiling reports locally on macOS using Docker
for `nsys` CLI and native SQLite for querying.

## When to Use This Skill

Activate this skill when:

- The user wants to analyze `.nsys-rep` profiling reports
- The user asks about GPU kernel performance, NCCL collective times, or MoE layer breakdown
- The user wants to compare performance across ranks (load imbalance)
- The user mentions "nsys", "nsight systems", "profiling", "kernel timeline"
- The user wants to focus on a specific training iteration or time window

## Architecture

macOS does not have `nsys` CLI. We run it inside a Docker container
(`nvcr.io/nvidia/pytorch:26.03-py3`) to convert `.nsys-rep` → `.sqlite`,
then analyze the SQLite locally with Python.

### Iteration Detection

CUDA graph replay does NOT replay NVTX markers, so NVTX-based iteration
detection is useless. Instead, iterations are detected from **kernel timeline
density**: we bin kernel GPU-time into 10ms windows, find contiguous "dense
compute" phases (>=8ms GPU time per 10ms bin, lasting >200ms), and identify
these as forward+backward compute iterations.

Typical iteration structure for MoE training:
```
[warmup/NCCL AllGather] → [COMPUTE: attention+MoE layers fwd+bwd] → [optimizer/gradient sync] → [COMPUTE] → ...
```

## Scripts

All scripts are in this skill directory:
`{SKILL_DIR}` = `.opencode/skills/nsys-analysis/`

### 1. Convert: `nsys_convert.py`

Batch-convert `.nsys-rep` files to `.sqlite` via Docker.

```bash
# Convert all ranks (parallel with 4 workers)
python {SKILL_DIR}/nsys_convert.py /path/to/nsys/dir

# Convert a single file
python {SKILL_DIR}/nsys_convert.py /path/to/file.nsys-rep

# Specify Docker image and parallelism
python {SKILL_DIR}/nsys_convert.py /path/to/dir --docker-image nvcr.io/nvidia/pytorch:26.03-py3 --jobs 8

# Force re-conversion (deletes existing .sqlite first)
python {SKILL_DIR}/nsys_convert.py /path/to/dir --force
```

**Docker image selection**: The nsys version in the Docker image must match
or be newer than the version that created the `.nsys-rep` files. The report
header contains the nsys version (e.g., `2026.1.2.63`). Currently:
- `nvcr.io/nvidia/pytorch:26.03-py3` → nsys `2026.1.2.63` (latest)
- `nvcr.io/nvidia/pytorch:26.02-py3` → nsys `2026.1.1.204` (older)

### 2. Analyze: `nsys_analyze.py`

Generate statistics from `.sqlite` files.

```bash
# Single rank — full profile
python {SKILL_DIR}/nsys_analyze.py /path/to/rank0.sqlite

# All ranks in a directory (multi-rank summary + rank-0 detail)
python {SKILL_DIR}/nsys_analyze.py /path/to/nsys/dir

# Restrict to specific ranks
python {SKILL_DIR}/nsys_analyze.py /path/to/dir --ranks 0,1,2,71

# Restrict to iteration N (1-indexed, auto-detected from kernel density)
python {SKILL_DIR}/nsys_analyze.py /path/to/rank0.sqlite --iteration 2

# Restrict to explicit time window (milliseconds)
python {SKILL_DIR}/nsys_analyze.py /path/to/rank0.sqlite --window 2355,2767

# JSON output
python {SKILL_DIR}/nsys_analyze.py /path/to/dir --format json

# Save to file
python {SKILL_DIR}/nsys_analyze.py /path/to/dir --output report.txt --top 20
```

## Report Sections

### Iteration Timeline
Auto-detected dense compute phases with start/end times, duration, GPU time,
and optimizer phase duration.

### Kernel Time by Category
Classified into: MoE ops (combine, dispatch, permute, scan, grouped_gemm, ...),
NCCL collectives, Attention (cuDNN Flash), GEMM (CUTLASS/nvjet),
LayerNorm/RMSNorm, Quantization, Elementwise, Optimizer, etc.

### Top Kernels
Per-kernel stats: invocations, total/avg/min/max time.

### NCCL Collectives
AllGather, AllReduce, ReduceScatter broken down by operation.

### Memory Copies
Device-to-Device, Host-to-Device, Device-to-Host with bandwidth.

### Multi-Rank: Load Imbalance
Coefficient of variation across ranks for each kernel category. High CV
indicates load imbalance (common for NCCL and MoE dispatch).

## SQLite Schema Reference

The key tables in nsys-exported SQLite for ad-hoc queries:

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `CUPTI_ACTIVITY_KIND_KERNEL` | `start`, `end`, `shortName`, `demangledName`, `gridX/Y/Z`, `blockX/Y/Z` | GPU kernel launches |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `start`, `end`, `bytes`, `copyKind` | Memory transfers |
| `CUPTI_ACTIVITY_KIND_MEMSET` | `start`, `end`, `bytes` | Memory sets |
| `NVTX_EVENTS` | `start`, `end`, `text`, `textId` | NVTX annotations (unreliable with CUDA graphs) |
| `StringIds` | `id`, `value` | Shared string table (kernel names reference this) |
| `TARGET_INFO_GPU` | `name`, `smCount`, `totalMemory`, `computeMajor` | GPU hardware info |
| `ENUM_CUDA_MEMCPY_OPER` | `id`, `label` | Memcpy kind enum (Device-to-Device, etc.) |

### Useful Ad-Hoc Queries

```sql
-- Top kernels by time in a window
SELECT s.value, COUNT(*), ROUND(SUM(k.end-k.start)/1e6, 2) AS ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
WHERE k.start >= 2355000000 AND k.start < 2767000000
GROUP BY s.value ORDER BY ms DESC LIMIT 20;

-- Kernel sequence in a time range
SELECT ROUND(start/1e6, 3) AS ms, ROUND((end-start)/1e3, 1) AS us,
       (SELECT value FROM StringIds WHERE id = shortName) AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE start >= 2200000000 AND start < 2210000000
ORDER BY start;

-- GPU utilization histogram (10ms bins)
SELECT CAST(start/10000000 AS INT)*10 AS bin_ms,
       COUNT(*) AS n, ROUND(SUM(end-start)/1e6, 2) AS gpu_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY CAST(start/10000000 AS INT) ORDER BY bin_ms;
```

## Kernel Classification

The analyzer auto-classifies kernels into categories. Key MoE patterns:

| Pattern | Category | Notes |
|---------|----------|-------|
| `combine_kernel` | MoE combine | Token combine after expert compute |
| `dispatch_kernel` | MoE dispatch | Token dispatch to experts |
| `permute_preprocessing_kernel` | MoE permute_preprocess | Sort-based routing prep |
| `scan` | MoE scan | Prefix scan for routing |
| `device_sync_kernel` | MoE device_sync | Inter-SM synchronization |
| `_paged_stash_{pop,copy}_kernel` | MoE paged_stash | Paged memory management |
| `grouped_gemm.*BlockScaled` | MoE grouped_gemm_* | cuDNN block-scaled MoE GEMMs |
| `fused_topk_with_score_function` | MoE topk | Top-K routing |
| `fused_moe_aux_loss` | MoE aux_loss | Auxiliary load-balancing loss |
| `ncclDevKernel_*` | NCCL * | Collective communication |
| `cudnn_generated.*sdpa.*flash` | Attention (cuDNN Flash) | Flash attention fwd/bwd |
| `cutlass3x_sm*`, `nvjet_sm*` | GEMM (CUTLASS/nvjet) | Dense matrix multiply |

## Workflow

1. **Convert**: Run `nsys_convert.py` on the directory of `.nsys-rep` files
2. **Overview**: Run `nsys_analyze.py` on rank 0's `.sqlite` to see iterations + full stats
3. **Focus**: Use `--iteration N` to zoom into a specific training step
4. **Window**: Use `--window start,end` for fine-grained analysis of specific layers
5. **Multi-rank**: Run on the full directory with `--ranks 0-7` (or all) for cross-rank comparison
6. **Ad-hoc**: Open the `.sqlite` in `sqlite3` CLI for custom queries
