#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen3-Omni server with SHM or CUDA-IPC connector deploy config.
#
# Usage:
#   bash run_server_connector_perf.sh [shm|ipc] [extra vllm serve args...]
#
# Examples:
#   bash run_server_connector_perf.sh shm
#   bash run_server_connector_perf.sh ipc --gpu-memory-utilization 0.9
#   COMMON_SERVER_ARGS="--stage-overrides '{\"2\":{\"devices\":\"1\"}}'" \
#     bash run_server_connector_perf.sh ipc

CONNECTOR_MODE="${1:-ipc}"
shift || true

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENABLE_PROFILE_LOGS="${ENABLE_PROFILE_LOGS:-1}"
# Shell words appended to vllm serve before CLI "$@".
# Use this to guarantee identical server-side args across connector runs.
COMMON_SERVER_ARGS="${COMMON_SERVER_ARGS:-}"

case "${CONNECTOR_MODE}" in
  shm)
    DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_omni_moe.yaml}"
    ;;
  ipc)
    DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_omni_moe_cuda_ipc.yaml}"
    ;;
  *)
    echo "Unsupported connector mode: ${CONNECTOR_MODE}"
    echo "Usage: $0 [shm|ipc] [extra vllm serve args...]"
    exit 1
    ;;
esac

if [[ "${ENABLE_PROFILE_LOGS}" == "1" ]]; then
  # Critical-path IPC profiling (used only by CudaIPCConnector).
  export VLLM_OMNI_CUDA_IPC_PROFILE_LOG="${VLLM_OMNI_CUDA_IPC_PROFILE_LOG:-1}"
  export VLLM_OMNI_CUDA_IPC_PROFILE_LOG_THRESHOLD_MS="${VLLM_OMNI_CUDA_IPC_PROFILE_LOG_THRESHOLD_MS:-2.0}"
  export VLLM_OMNI_CUDA_IPC_PROFILE_LOG_EVERY_N="${VLLM_OMNI_CUDA_IPC_PROFILE_LOG_EVERY_N:-64}"
  # Msgpack tensor encode profiling for hidden D2H bottlenecks.
  export VLLM_OMNI_MSGPACK_PROFILE_TENSORS="${VLLM_OMNI_MSGPACK_PROFILE_TENSORS:-1}"
  export VLLM_OMNI_MSGPACK_PROFILE_TENSORS_THRESHOLD_MS="${VLLM_OMNI_MSGPACK_PROFILE_TENSORS_THRESHOLD_MS:-3.0}"
  export VLLM_OMNI_MSGPACK_PROFILE_TENSORS_THRESHOLD_BYTES="${VLLM_OMNI_MSGPACK_PROFILE_TENSORS_THRESHOLD_BYTES:-1048576}"
  export VLLM_OMNI_MSGPACK_PROFILE_TENSORS_EVERY_N="${VLLM_OMNI_MSGPACK_PROFILE_TENSORS_EVERY_N:-32}"
  # Save-loop profiling (queue wait + put elapsed) in model runner mixin.
  export VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG="${VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG:-1}"
  export VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG_THRESHOLD_MS="${VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG_THRESHOLD_MS:-2.0}"
  export VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG_EVERY_N="${VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG_EVERY_N:-64}"
  # Optional deep split of receiver wait (disabled by default to avoid perturbing timings).
  export VLLM_OMNI_CUDA_IPC_PROFILE_WAIT_SPLIT="${VLLM_OMNI_CUDA_IPC_PROFILE_WAIT_SPLIT:-0}"
fi

if [[ "${CONNECTOR_MODE}" == "ipc" ]]; then
  # Benchmark-oriented default: disable shm-compat probe on ring miss to
  # remove extra /dev/shm miss overhead in dedicated IPC deployments.
  # Set to 1 if you still want receiver-side fallback probing.
  export VLLM_OMNI_CUDA_IPC_SHM_COMPAT_ON_RING_MISS="${VLLM_OMNI_CUDA_IPC_SHM_COMPAT_ON_RING_MISS:-0}"
  # Benchmark-oriented default: disable sender blocking sync in pool put path.
  # This keeps IPC event ordering and source-lifetime tracking, while avoiding
  # hard sender-side waits in put_pool.
  export VLLM_OMNI_CUDA_IPC_PUT_POOL_BLOCKING_SYNC="${VLLM_OMNI_CUDA_IPC_PUT_POOL_BLOCKING_SYNC:-0}"
  # Benchmark-oriented default: do not make pool-get copy stream wait the
  # receiver current stream before D2D decode (improves overlap).
  export VLLM_OMNI_CUDA_IPC_GET_POOL_WAIT_CURRENT_STREAM="${VLLM_OMNI_CUDA_IPC_GET_POOL_WAIT_CURRENT_STREAM:-0}"
  # Sender-side async put_pool tuning: parallel pack streams + bounded
  # inflight window to avoid excessive descriptor lead over copy completion.
  export VLLM_OMNI_CUDA_IPC_PUT_POOL_COPY_STREAMS="${VLLM_OMNI_CUDA_IPC_PUT_POOL_COPY_STREAMS:-4}"
  export VLLM_OMNI_CUDA_IPC_PUT_POOL_ASYNC_INFLIGHT_LIMIT="${VLLM_OMNI_CUDA_IPC_PUT_POOL_ASYNC_INFLIGHT_LIMIT:-16}"
fi

cmd=(
  vllm
  serve
  "${MODEL}"
  --omni
  --host "${HOST}"
  --port "${PORT}"
  --deploy-config "${DEPLOY_CONFIG}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ -n "${COMMON_SERVER_ARGS}" ]]; then
  # shellcheck disable=SC2206
  common_server_args_arr=( ${COMMON_SERVER_ARGS} )
  cmd+=("${common_server_args_arr[@]}")
fi

echo "============================================================"
echo "Starting Qwen3-Omni server"
echo "  connector_mode : ${CONNECTOR_MODE}"
echo "  model          : ${MODEL}"
echo "  host:port      : ${HOST}:${PORT}"
echo "  deploy_config  : ${DEPLOY_CONFIG}"
echo "  profile_logs   : ${ENABLE_PROFILE_LOGS}"
echo "  shm_compat_on_miss : ${VLLM_OMNI_CUDA_IPC_SHM_COMPAT_ON_RING_MISS:-<default>}"
echo "  put_pool_blocking_sync : ${VLLM_OMNI_CUDA_IPC_PUT_POOL_BLOCKING_SYNC:-<default>}"
echo "  get_pool_wait_current_stream : ${VLLM_OMNI_CUDA_IPC_GET_POOL_WAIT_CURRENT_STREAM:-<default>}"
echo "  put_pool_copy_streams : ${VLLM_OMNI_CUDA_IPC_PUT_POOL_COPY_STREAMS:-<default>}"
echo "  put_pool_async_inflight_limit : ${VLLM_OMNI_CUDA_IPC_PUT_POOL_ASYNC_INFLIGHT_LIMIT:-<default>}"
echo "  save_profile_log : ${VLLM_OMNI_CONNECTOR_SAVE_PROFILE_LOG:-<default>}"
echo "  profile_wait_split : ${VLLM_OMNI_CUDA_IPC_PROFILE_WAIT_SPLIT:-<default>}"
echo "  common_args    : ${COMMON_SERVER_ARGS:-<none>}"
echo "============================================================"
echo "Command: ${cmd[*]} $*"
echo "============================================================"

exec "${cmd[@]}" "$@"
