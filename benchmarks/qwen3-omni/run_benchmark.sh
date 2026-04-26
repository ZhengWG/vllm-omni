#!/bin/bash
# Qwen3-Omni Benchmark Runner
#
# Compares vllm-omni online serving vs HuggingFace transformers offline
# inference on the multi-stage Qwen3-Omni pipeline (Thinker -> Talker ->
# Code2Wav). Produces JSON results and comparison plots.
#
# Usage:
#   # Full comparison (vllm-omni + HF):
#   bash run_benchmark.sh
#
#   # Only vllm-omni:
#   bash run_benchmark.sh --vllm-only
#
#   # Only HuggingFace baseline:
#   bash run_benchmark.sh --hf-only
#
#   # Custom settings:
#   GPU_DEVICES=0,1 NUM_PROMPTS=20 CONCURRENCY="1 4" QUERY_TYPE=text \
#       bash run_benchmark.sh
#
#   # Multimodal input (audio question):
#   QUERY_TYPE=use_audio bash run_benchmark.sh --vllm-only
#
# Environment variables:
#   GPU_DEVICES      - GPUs visible to the server / HF runner
#                      (default: "0,1" because the bundled deploy YAML
#                      uses 2x GPUs: stage 0 on cuda:0, stages 1+2 on cuda:1).
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 20)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   MODEL            - Model name (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   PORT             - Server port (default: 8091)
#   QUERY_TYPE       - text | use_audio | use_image | use_video (default: text)
#   MODALITIES       - Comma-separated output modalities (default: "audio")
#   SPEAKER          - Optional TTS speaker (e.g. "chelsie"); empty -> server default
#   DEPLOY_CONFIG    - Path to deploy YAML (default: vllm_omni/deploy/qwen3_omni_moe.yaml)
#   STAGE_OVERRIDES  - Optional JSON for --stage-overrides
#   HF_NUM_PROMPTS   - HF prompts (defaults to NUM_PROMPTS, capped at 10 because
#                      HF transformers offline path is much slower)
#   HF_NUM_WARMUPS   - HF warmup runs (default: 1)
#   HF_GPU_DEVICE    - GPU index for the HF runner (default: first of GPU_DEVICES)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GPU_DEVICES="${GPU_DEVICES:-0,1}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-8091}"
NUM_WARMUPS="${NUM_WARMUPS:-2}"
QUERY_TYPE="${QUERY_TYPE:-text}"
MODALITIES="${MODALITIES:-audio}"
SPEAKER="${SPEAKER:-}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/qwen3_omni_moe.yaml}"
STAGE_OVERRIDES="${STAGE_OVERRIDES:-}"
HF_NUM_PROMPTS="${HF_NUM_PROMPTS:-$([ "${NUM_PROMPTS}" -gt 10 ] && echo 10 || echo "${NUM_PROMPTS}")}"
HF_NUM_WARMUPS="${HF_NUM_WARMUPS:-1}"
HF_GPU_DEVICE="${HF_GPU_DEVICE:-${GPU_DEVICES%%,*}}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Parse args
RUN_VLLM=true
RUN_HF=true
for arg in "$@"; do
    case "$arg" in
        --vllm-only) RUN_HF=false ;;
        --hf-only) RUN_VLLM=false ;;
        --skip-hf) RUN_HF=false ;;
    esac
done

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-Omni Benchmark"
echo "============================================================"
echo " GPUs:           ${GPU_DEVICES}"
echo " Model:          ${MODEL}"
echo " Prompts:        ${NUM_PROMPTS}"
echo " Concurrency:    ${CONCURRENCY}"
echo " Port:           ${PORT}"
echo " Query type:     ${QUERY_TYPE}"
echo " Modalities:     ${MODALITIES}"
echo " Deploy config:  ${DEPLOY_CONFIG}"
echo " Stage overrides:${STAGE_OVERRIDES:-<none>}"
echo " Speaker:        ${SPEAKER:-<server-default>}"
echo " Results:        ${RESULT_DIR}"
echo "============================================================"

start_server() {
    local config_name="$1"
    local log_file="${RESULT_DIR}/server_${config_name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting server with config: ${config_name}"
    echo "  Deploy config: ${DEPLOY_CONFIG}"
    if [ -n "${STAGE_OVERRIDES}" ]; then
        echo "  Stage overrides: ${STAGE_OVERRIDES}"
    fi
    echo "  Log file: ${log_file}"

    local stage_override_args=()
    if [ -n "${STAGE_OVERRIDES}" ]; then
        stage_override_args=(--stage-overrides "${STAGE_OVERRIDES}")
    fi

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
        --omni \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --deploy-config "${DEPLOY_CONFIG}" \
        --stage-init-timeout 300 \
        --trust-remote-code \
        --disable-log-stats \
        "${stage_override_args[@]}" \
        > "${log_file}" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    echo "  Waiting for server to be ready..."
    local max_wait=600
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "  ERROR: Server process died. Check log: ${log_file}"
            tail -20 "${log_file}"
            return 1
        fi
        sleep 3
        waited=$((waited + 3))
    done

    echo "  ERROR: Server did not start within ${max_wait}s. Check log: ${log_file}"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "  Stopping server (PID: ${SERVER_PID})..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        local pids
        pids=$(lsof -ti:${PORT} 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo "  Cleaning up remaining processes on port ${PORT}..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
        fi
        echo "  Server stopped."
        SERVER_PID=""
    fi
}

trap 'stop_server' EXIT

run_vllm_bench() {
    local config_name="async_chunk"

    echo ""
    echo "============================================================"
    echo " Benchmarking: vllm-omni (${config_name})"
    echo "============================================================"

    start_server "${config_name}"

    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    local speaker_args=()
    if [ -n "${SPEAKER}" ]; then
        speaker_args=(--speaker "${SPEAKER}")
    fi

    cd "${PROJECT_ROOT}"
    python "${SCRIPT_DIR}/vllm_omni/bench_omni_serve.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --query-type "${QUERY_TYPE}" \
        --modalities "${MODALITIES}" \
        --config-name "${config_name}" \
        --result-dir "${RESULT_DIR}" \
        "${speaker_args[@]}"

    stop_server
    sleep 5
}

if [ "${RUN_VLLM}" = true ]; then
    run_vllm_bench
fi

if [ "${RUN_HF}" = true ]; then
    echo ""
    echo "============================================================"
    echo " Benchmarking: HuggingFace transformers (offline, single GPU)"
    echo "============================================================"

    cd "${PROJECT_ROOT}"
    CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    python "${SCRIPT_DIR}/transformers/bench_omni_hf.py" \
        --model "${MODEL}" \
        --num-prompts "${HF_NUM_PROMPTS}" \
        --num-warmups "${HF_NUM_WARMUPS}" \
        --gpu-device "${HF_GPU_DEVICE}" \
        --query-type "${QUERY_TYPE}" \
        --modalities "${MODALITIES}" \
        --config-name "hf_transformers" \
        --result-dir "${RESULT_DIR}"
    sleep 3
fi

echo ""
echo "============================================================"
echo " Generating plots..."
echo "============================================================"

RESULT_FILES=""
LABELS=""

if [ "${RUN_VLLM}" = true ]; then
    VLLM_FILE=$(ls -t "${RESULT_DIR}"/bench_async_chunk_*.json 2>/dev/null | head -1)
    if [ -n "${VLLM_FILE}" ]; then
        RESULT_FILES="${VLLM_FILE}"
        LABELS="vllm-omni"
    fi
fi

if [ "${RUN_HF}" = true ]; then
    HF_FILE=$(ls -t "${RESULT_DIR}"/bench_hf_transformers_*.json 2>/dev/null | head -1)
    if [ -n "${HF_FILE}" ]; then
        if [ -n "${RESULT_FILES}" ]; then
            RESULT_FILES="${RESULT_FILES} ${HF_FILE}"
            LABELS="${LABELS} hf_transformers"
        else
            RESULT_FILES="${HF_FILE}"
            LABELS="hf_transformers"
        fi
    fi
fi

if [ -n "${RESULT_FILES}" ]; then
    python "${SCRIPT_DIR}/plot_results.py" \
        --results ${RESULT_FILES} \
        --labels ${LABELS} \
        --output "${RESULT_DIR}/qwen3_omni_benchmark_${TIMESTAMP}.png"
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
echo "============================================================"
