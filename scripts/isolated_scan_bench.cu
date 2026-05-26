#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef SCAN_PERMUTE_FUSION
#define SCAN_PERMUTE_FUSION 1
#endif
#if SCAN_PERMUTE_FUSION
#define HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE 1
#endif
#include "hybrid_ep_backend.cuh"

#ifndef SCAN_THREADS
#define SCAN_THREADS 256
#endif
#ifndef SCAN_BLOCKS
#define SCAN_BLOCKS 108
#endif
#ifndef SCAN_LOCAL_EXPERTS
#define SCAN_LOCAL_EXPERTS 32
#endif
#ifndef SCAN_TOPK
#define SCAN_TOPK 36
#endif

#define CHECK_CUDA(expr)                                                        \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kPad = 256;
constexpr int kMaxTokens = 12288;
constexpr int kChunk = 64;
constexpr int kRanks = 72;
constexpr int kNodes = 1;
constexpr int kRows = kMaxTokens * kRanks;
constexpr int kChunksPerRank = (kMaxTokens + kChunk - 1) / kChunk;
constexpr int kTotalChunks = kChunksPerRank * kRanks * kNodes;

template <int LocalExperts, int Topk>
void init_topk(std::vector<int16_t>& topk) {
  constexpr int total_experts = kRanks * LocalExperts;
  topk.resize(static_cast<size_t>(kRows) * Topk);
  for (int row = 0; row < kRows; ++row) {
    for (int k = 0; k < Topk; ++k) {
      int expert = (row * 37 + k * 53 + (row >> 3)) % total_experts;
      bool duplicate = true;
      while (duplicate) {
        duplicate = false;
        for (int prev = 0; prev < k; ++prev) {
          if (topk[static_cast<size_t>(row) * Topk + prev] == expert) {
            duplicate = true;
            expert = (expert + 1) % total_experts;
            break;
          }
        }
      }
      topk[static_cast<size_t>(row) * Topk + k] = static_cast<int16_t>(expert);
    }
  }
}

void launch_scan(const int16_t* input_topk,
                 hybrid_ep::tmp_state_t* tmp,
                 hybrid_ep::tmp_state_t* local_tmp,
                 int32_t* sparse_to_dense,
                 bool* rdma_to_attn,
                 bool* attn_to_rdma,
                 int32_t* num_tokens_for_experts,
                 bool* local_expert_routing,
                 int32_t* dense_chunk_layout,
                 int32_t* dense_to_expert,
                 int32_t* num_local_expert_tokens,
                 int* overflow_flag,
                 cudaStream_t stream) {
  hybrid_ep::scan<SCAN_THREADS, SCAN_BLOCKS, kPad, kMaxTokens, kChunk, kRanks, kNodes,
                  SCAN_LOCAL_EXPERTS, SCAN_TOPK>
      <<<SCAN_BLOCKS, SCAN_THREADS, 0, stream>>>(
          input_topk, tmp, local_tmp, sparse_to_dense, rdma_to_attn,
          attn_to_rdma, num_tokens_for_experts, local_expert_routing,
          dense_chunk_layout, dense_to_expert, num_local_expert_tokens,
          overflow_flag, /*node_rank=*/0, /*local_rank=*/0,
          /*local_experts_tokens_limit=*/0x3fffffff, kMaxTokens);
}

float run_case(int warmups, int tests) {
  std::vector<int16_t> host_topk;
  init_topk<SCAN_LOCAL_EXPERTS, SCAN_TOPK>(host_topk);

  int16_t* input_topk = nullptr;
  hybrid_ep::tmp_state_t* tmp = nullptr;
  hybrid_ep::tmp_state_t* local_tmp = nullptr;
  int32_t* sparse_to_dense = nullptr;
  bool* rdma_to_attn = nullptr;
  bool* attn_to_rdma = nullptr;
  int32_t* num_tokens_for_experts = nullptr;
  bool* local_expert_routing = nullptr;
  int32_t* dense_chunk_layout = nullptr;
  int32_t* dense_to_expert = nullptr;
  int32_t* num_local_expert_tokens = nullptr;
  int* overflow_flag = nullptr;

  CHECK_CUDA(cudaMalloc(&input_topk, host_topk.size() * sizeof(int16_t)));
  CHECK_CUDA(cudaMemcpy(input_topk, host_topk.data(), host_topk.size() * sizeof(int16_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&tmp, SCAN_BLOCKS * kRanks * sizeof(hybrid_ep::tmp_state_t)));
  CHECK_CUDA(cudaMalloc(&local_tmp, SCAN_BLOCKS * SCAN_LOCAL_EXPERTS * sizeof(hybrid_ep::tmp_state_t)));
  CHECK_CUDA(cudaMalloc(&sparse_to_dense, kMaxTokens * kNodes * kRanks * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&rdma_to_attn, kMaxTokens * kNodes * sizeof(bool)));
  CHECK_CUDA(cudaMalloc(&attn_to_rdma, sizeof(bool)));
  CHECK_CUDA(cudaMalloc(&num_tokens_for_experts, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&local_expert_routing,
                        static_cast<size_t>(kRows) * SCAN_LOCAL_EXPERTS * sizeof(bool)));
  CHECK_CUDA(cudaMalloc(&dense_chunk_layout, kTotalChunks * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&dense_to_expert,
                        static_cast<size_t>(kRows) * SCAN_LOCAL_EXPERTS * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&num_local_expert_tokens, SCAN_LOCAL_EXPERTS * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&overflow_flag, sizeof(int)));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  auto reset = [&]() {
    CHECK_CUDA(cudaMemsetAsync(tmp, 0, SCAN_BLOCKS * kRanks * sizeof(hybrid_ep::tmp_state_t), stream));
    CHECK_CUDA(cudaMemsetAsync(local_tmp, 0, SCAN_BLOCKS * SCAN_LOCAL_EXPERTS * sizeof(hybrid_ep::tmp_state_t), stream));
    CHECK_CUDA(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream));
  };

  for (int i = 0; i < warmups; ++i) {
    reset();
    launch_scan(
        input_topk, tmp, local_tmp, sparse_to_dense, rdma_to_attn, attn_to_rdma,
        num_tokens_for_experts, local_expert_routing, dense_chunk_layout, dense_to_expert,
        num_local_expert_tokens, overflow_flag, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  float total_ms = 0.0f;
  for (int i = 0; i < tests; ++i) {
    reset();
    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_scan(
        input_topk, tmp, local_tmp, sparse_to_dense, rdma_to_attn, attn_to_rdma,
        num_tokens_for_experts, local_expert_routing, dense_chunk_layout, dense_to_expert,
        num_local_expert_tokens, overflow_flag, stream);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(input_topk));
  CHECK_CUDA(cudaFree(tmp));
  CHECK_CUDA(cudaFree(local_tmp));
  CHECK_CUDA(cudaFree(sparse_to_dense));
  CHECK_CUDA(cudaFree(rdma_to_attn));
  CHECK_CUDA(cudaFree(attn_to_rdma));
  CHECK_CUDA(cudaFree(num_tokens_for_experts));
  CHECK_CUDA(cudaFree(local_expert_routing));
  CHECK_CUDA(cudaFree(dense_chunk_layout));
  CHECK_CUDA(cudaFree(dense_to_expert));
  CHECK_CUDA(cudaFree(num_local_expert_tokens));
  CHECK_CUDA(cudaFree(overflow_flag));
  return total_ms * 1000.0f / tests;
}

}  // namespace

int main(int argc, char** argv) {
  int warmups = argc > 1 ? std::atoi(argv[1]) : 10;
  int tests = argc > 2 ? std::atoi(argv[2]) : 30;
  float us = run_case(warmups, tests);
  std::printf("isolated scan<%d,%d,256,12288,64,72,1,%d,%d>: %.3f us\n",
              SCAN_THREADS, SCAN_BLOCKS, SCAN_LOCAL_EXPERTS, SCAN_TOPK, us);
  std::printf("permute_fusion=%d\n", SCAN_PERMUTE_FUSION);
  return 0;
}
