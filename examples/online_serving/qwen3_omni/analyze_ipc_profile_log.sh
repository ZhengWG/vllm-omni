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
#   bash analyze_ipc_profile_log.sh [log_path]
#
# Example:
#   bash analyze_ipc_profile_log.sh /tmp/ipc_server.log

LOG_PATH="${1:-/tmp/ipc_server.log}"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "ERROR: log file not found: ${LOG_PATH}" >&2
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

print_section "1) IPC connector profiling lines"
grep -nE "CudaIPCConnector profile" "${LOG_PATH}" || true

print_section "2) IPC put/get critical phases"
grep -nE "phase=put_pool|phase=get_control_plane|phase=put_control_plane|phase=put_inline" "${LOG_PATH}" || true

print_section "3) Msgpack tensor encode profiling (implicit D2H clue)"
grep -nE "tensor_encode_profile" "${LOG_PATH}" || true

print_section "4) Fallback / anomaly / error signals"
grep -nE "fallback|ring_full|credits_exhausted|slot_overflow|descriptor_too_big|control-plane get failed|control-plane put failed|EngineDeadError|ValidationError|shm_compat decode failed" "${LOG_PATH}" || true

print_section "5) Top 30 slow IPC profile rows (by elapsed_ms)"
# Parse lines with:
#   CudaIPCConnector profile ... elapsed_ms=12.345 ...
# Sort numerically and print top rows.
grep -E "CudaIPCConnector profile" "${LOG_PATH}" \
  | sed -nE 's/.*elapsed_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 30 \
  | cut -f2- || true

print_section "6) Top 20 slow msgpack tensor encode rows (by elapsed_ms)"
grep -E "tensor_encode_profile" "${LOG_PATH}" \
  | sed -nE 's/.*elapsed_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 20 \
  | cut -f2- || true

print_section "Done"
