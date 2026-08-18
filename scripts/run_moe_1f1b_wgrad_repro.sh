#!/usr/bin/env bash

# Reproduce the CuTe DSL grouped-wgrad stall with the production GPT training
# schedule, while keeping the model small enough for a two-GPU computelab run.

set -euo pipefail

repo_root="${MLM_ROOT:-/home/scratch.hhanyu_gpu/projects/moe/MLM}"
num_gpus="${NUM_GPUS:-2}"
num_layers="${NUM_LAYERS:-4}"
num_experts="${NUM_EXPERTS:-128}"
hidden_size="${HIDDEN_SIZE:-512}"
ffn_hidden_size="${FFN_HIDDEN_SIZE:-1024}"
moe_ffn_hidden_size="${MOE_FFN_HIDDEN_SIZE:-1024}"
num_attention_heads="${NUM_ATTENTION_HEADS:-8}"
kv_channels="${KV_CHANNELS:-64}"
router_topk="${ROUTER_TOPK:-8}"
moe_latent_size="${MOE_LATENT_SIZE:-}"
shared_expert_size="${SHARED_EXPERT_SIZE:-}"
seq_length="${SEQ_LENGTH:-1024}"
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
train_iters="${TRAIN_ITERS:-3}"
run_timeout_seconds="${RUN_TIMEOUT_SECONDS:-360}"
expert_rank_capacity_factor="${EXPERT_RANK_CAPACITY_FACTOR:-16.0}"
stash_buffer_size_factor="${STASH_BUFFER_SIZE_FACTOR:-4.0}"
variant="${VARIANT:-cutedsl_wgrad}"
log_dir="${LOG_DIR:-${repo_root}/logs/onef1b_wgrad_repro}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${LOG_FILE:-${log_dir}/${variant}_${timestamp}.log}"
gpu_log_file="${GPU_LOG_FILE:-${log_dir}/${variant}_${timestamp}_gpu.csv}"

mkdir -p "${log_dir}"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}"
export NVTE_CUTEDSL_FUSED_GROUPED_MLP="${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-1}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-${num_gpus}}"
hybridep_ranks_per_domain="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-${num_gpus}}"
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${hybridep_ranks_per_domain}"
export NUM_SMS_DISPATCH="${NUM_SMS_DISPATCH:-24}"
export NUM_SMS_COMBINE="${NUM_SMS_COMBINE:-24}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}"
export MCORE_DEBUG_DENSE_ROUTING="${MCORE_DEBUG_DENSE_ROUTING:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[repro] variant=${variant}" >"${log_file}"
echo "[repro] log_file=${log_file}" >>"${log_file}"
echo "[repro] gpu_log_file=${gpu_log_file}" >>"${log_file}"
echo "[repro] num_gpus=${num_gpus} layers=${num_layers} experts=${num_experts}" >>"${log_file}"
echo "[repro] hidden=${hidden_size} ffn=${ffn_hidden_size} moe_ffn=${moe_ffn_hidden_size}" >>"${log_file}"
echo "[repro] heads=${num_attention_heads} kv=${kv_channels} topk=${router_topk}" >>"${log_file}"
echo "[repro] latent=${moe_latent_size:-none} shared=${shared_expert_size:-none}" >>"${log_file}"
echo "[repro] mbs=${micro_batch_size} gbs=${global_batch_size} seq=${seq_length}" >>"${log_file}"
echo "[repro] capacity_factor=${expert_rank_capacity_factor}" >>"${log_file}"
echo "[repro] stash_buffer_size_factor=${stash_buffer_size_factor}" >>"${log_file}"
env | sort >>"${log_file}"

nvidia-smi \
    --query-compute-apps=timestamp,gpu_uuid,pid,used_memory \
    --format=csv \
    --loop=2 >"${gpu_log_file}" 2>&1 &
gpu_monitor_pid=$!

cleanup() {
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT

moe_arch_args=()
if [[ -n "${moe_latent_size}" ]]; then
    moe_arch_args+=(--moe-latent-size "${moe_latent_size}")
fi
if [[ -n "${shared_expert_size}" ]]; then
    moe_arch_args+=(
        --moe-shared-expert-intermediate-size "${shared_expert_size}"
        --moe-shared-expert-gate
    )
fi

set +e
timeout --signal=INT --kill-after=30s "${run_timeout_seconds}s" \
    /usr/bin/python3 -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${num_gpus}" \
    pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --enable-experimental \
    --num-layers "${num_layers}" \
    --hidden-size "${hidden_size}" \
    --ffn-hidden-size "${ffn_hidden_size}" \
    --num-attention-heads "${num_attention_heads}" \
    --kv-channels "${kv_channels}" \
    --seq-length "${seq_length}" \
    --max-position-embeddings "${seq_length}" \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-backend unfused \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size "${num_gpus}" \
    --expert-tensor-parallel-size 1 \
    --context-parallel-size 1 \
    --num-experts "${num_experts}" \
    --moe-ffn-hidden-size "${moe_ffn_hidden_size}" \
    --moe-router-topk "${router_topk}" \
    "${moe_arch_args[@]}" \
    --moe-router-load-balancing-type none \
    --moe-router-force-load-balancing \
    --moe-router-dtype fp32 \
    --moe-router-fusion \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --moe-hybridep-num-sms 24 \
    --moe-permute-fusion \
    --moe-router-padding-for-quantization \
    --use-transformer-engine-op-fuser \
    --moe-mlp-glu-interleave-size 32 \
    --moe-paged-stash \
    --moe-expert-rank-capacity-factor "${expert_rank_capacity_factor}" \
    --moe-paged-stash-page-size 64 \
    --moe-paged-stash-buffer-size-factor-cuda "${stash_buffer_size_factor}" \
    --moe-pad-experts-for-cuda-graph-inference \
    --overlap-moe-expert-parallel-comm \
    --delay-wgrad-compute \
    --cuda-graph-impl full_iteration \
    --no-check-for-nan-in-loss-and-grad \
    --bf16 \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --micro-batch-size "${micro_batch_size}" \
    --global-batch-size "${global_batch_size}" \
    --train-iters "${train_iters}" \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters "${train_iters}" \
    --lr-warmup-iters 0 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 1024 \
    --no-create-attention-mask-in-dataloader \
    --num-workers 0 \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 1000 \
    --manual-gc \
    --manual-gc-interval 5 \
    --exit-duration-in-mins 10 \
    >>"${log_file}" 2>&1
status=$?
set -e

echo "[repro] exit_status=${status}" >>"${log_file}"
echo "[repro] completed_at=$(date -u +%Y%m%dT%H%M%SZ)" >>"${log_file}"
echo "${log_file}"
exit "${status}"
