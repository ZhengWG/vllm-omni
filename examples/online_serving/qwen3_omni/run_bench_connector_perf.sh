#!/usr/bin/env bash
set -euo pipefail

# Run serving benchmarks for Qwen3-Omni against a running server.
#
# Usage:
#   bash run_bench_connector_perf.sh [shm|ipc] [extra vllm bench serve args...]
#
# Examples:
#   bash run_bench_connector_perf.sh ipc
#   CONCURRENCY_LIST="1 4 8" NUM_PROMPTS=64 bash run_bench_connector_perf.sh shm
#   COMMON_BENCH_ARGS="--metric-percentiles 50,90,99 --trust-remote-code" \
#     bash run_bench_connector_perf.sh ipc

CONNECTOR_MODE="${1:-ipc}"
shift || true

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
RESULT_DIR="${RESULT_DIR:-bench_results/qwen3_omni_connector_perf}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 4}"
NUM_PROMPTS="${NUM_PROMPTS:-40}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-4000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-900}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-{\"modalities\":[\"text\",\"audio\"]}}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration}"
# Shell words appended to every bench invocation before CLI "$@".
# Use this to keep bench args strictly identical across connector runs.
COMMON_BENCH_ARGS="${COMMON_BENCH_ARGS:-}"

mkdir -p "${RESULT_DIR}"

common_bench_args_arr=()
if [[ -n "${COMMON_BENCH_ARGS}" ]]; then
  # shellcheck disable=SC2206
  common_bench_args_arr=( ${COMMON_BENCH_ARGS} )
fi

for c in ${CONCURRENCY_LIST}; do
  result_file="qwen3_omni_${CONNECTOR_MODE}_c${c}_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}.json"
  echo "------------------------------------------------------------"
  echo "Running benchmark"
  echo "  connector_mode : ${CONNECTOR_MODE}"
  echo "  host:port      : ${HOST}:${PORT}"
  echo "  concurrency    : ${c}"
  echo "  num_prompts    : ${NUM_PROMPTS}"
  echo "  input/output   : ${RANDOM_INPUT_LEN}/${RANDOM_OUTPUT_LEN}"
  echo "  result_file    : ${RESULT_DIR}/${result_file}"
  echo "  common_args    : ${COMMON_BENCH_ARGS:-<none>}"
  echo "------------------------------------------------------------"

  vllm bench serve \
    --omni \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL}" \
    --endpoint /v1/chat/completions \
    --backend openai-chat-omni \
    --dataset-name random \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${c}" \
    --request-rate inf \
    --random-input-len "${RANDOM_INPUT_LEN}" \
    --random-output-len "${RANDOM_OUTPUT_LEN}" \
    --ignore-eos \
    --extra-body "${EXTRA_BODY_JSON}" \
    --percentile-metrics "${PERCENTILE_METRICS}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${result_file}" \
    --print-stage \
    "${common_bench_args_arr[@]}" \
    "$@"
done

echo "Done. Benchmark results are under: ${RESULT_DIR}"
