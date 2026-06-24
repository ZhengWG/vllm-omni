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

CONNECTOR_MODE="${1:-ipc}"
shift || true

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENABLE_PROFILE_LOGS="${ENABLE_PROFILE_LOGS:-1}"

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

echo "============================================================"
echo "Starting Qwen3-Omni server"
echo "  connector_mode : ${CONNECTOR_MODE}"
echo "  model          : ${MODEL}"
echo "  host:port      : ${HOST}:${PORT}"
echo "  deploy_config  : ${DEPLOY_CONFIG}"
echo "  profile_logs   : ${ENABLE_PROFILE_LOGS}"
echo "============================================================"
echo "Command: ${cmd[*]} $*"
echo "============================================================"

exec "${cmd[@]}" "$@"
