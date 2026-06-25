#!/usr/bin/env bash
set -euo pipefail

# One-shot analyzer for IPC profiling logs.
# Uses grep-based filters (per request) to quickly inspect:
#  1) IPC connector profiling lines
#  2) put/get hot phases
#  3) msgpack tensor encode profiling
#  4) fallback / error signals
#  5) top-N slow profile rows by elapsed_ms
#
# Usage:
#   bash analyze_ipc_profile_log.sh [log_path] [tail_lines]
#
# Example:
#   bash analyze_ipc_profile_log.sh /tmp/ipc_server.log
#   bash analyze_ipc_profile_log.sh /tmp/ipc_server.log 3000

LOG_PATH="${1:-/tmp/ipc_server.log}"
TAIL_LINES="${2:-4000}"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "ERROR: log file not found: ${LOG_PATH}" >&2
  exit 1
fi

if [[ ! "${TAIL_LINES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: tail_lines must be a non-negative integer, got: ${TAIL_LINES}" >&2
  exit 1
fi

print_section() {
  echo
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

print_section "0) Source log"
echo "${LOG_PATH}"
if [[ "${TAIL_LINES}" -gt 0 ]]; then
  echo "Tail mode: ON (last ${TAIL_LINES} lines)"
  WORK_LOG="$(mktemp)"
  tail -n "${TAIL_LINES}" "${LOG_PATH}" > "${WORK_LOG}"
else
  echo "Tail mode: OFF (full file)"
  WORK_LOG="${LOG_PATH}"
fi

print_section "1) IPC connector profiling lines"
grep -nE "CudaIPCConnector profile" "${WORK_LOG}" || true

print_section "2) IPC put/get critical phases"
grep -nE "phase=put_pool|phase=get_control_plane|phase=put_control_plane|phase=put_inline" "${WORK_LOG}" || true

print_section "3) Msgpack tensor encode profiling (implicit D2H clue)"
grep -nE "tensor_encode_profile" "${WORK_LOG}" || true

print_section "4) Fallback / anomaly / error signals"
grep -nE "fallback|ring_full|credits_exhausted|slot_overflow|descriptor_too_big|control-plane get failed|control-plane put failed|EngineDeadError|ValidationError|shm_compat decode failed" "${WORK_LOG}" || true

print_section "5) Top 30 slow IPC profile rows (by elapsed_ms)"
# Parse lines with:
#   CudaIPCConnector profile ... elapsed_ms=12.345 ...
# Sort numerically and print top rows.
grep -E "CudaIPCConnector profile" "${WORK_LOG}" \
  | sed -nE 's/.*elapsed_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 30 \
  | cut -f2- || true

print_section "6) Top 20 slow msgpack tensor encode rows (by elapsed_ms)"
grep -E "tensor_encode_profile" "${WORK_LOG}" \
  | sed -nE 's/.*elapsed_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 20 \
  | cut -f2- || true

print_section "Done"

if [[ "${TAIL_LINES}" -gt 0 ]] && [[ -n "${WORK_LOG:-}" ]] && [[ -f "${WORK_LOG}" ]] && [[ "${WORK_LOG}" != "${LOG_PATH}" ]]; then
  rm -f "${WORK_LOG}"
fi
