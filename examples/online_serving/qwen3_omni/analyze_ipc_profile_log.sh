#!/usr/bin/env bash
set -euo pipefail

# One-shot analyzer for IPC profiling logs.
# Uses grep-based filters (per request) to quickly inspect:
#  1) IPC connector profiling lines
#  2) put/get hot phases
#  3) msgpack tensor encode profiling
#  4) fallback / error signals
#  5) top-N slow profile rows by elapsed_ms
#  6) quantiles + sync-ratio summaries for key IPC stages
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

print_section "3.5) Connector save-loop profiling"
grep -nE "OmniConnector save_profile" "${WORK_LOG}" || true

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

print_section "7) Quantiles & ratio summary for IPC hot paths"
python3 - "${WORK_LOG}" <<'PY'
import math
import re
import statistics as st
import sys

if len(sys.argv) < 2:
    print("ERROR: missing work log path")
    sys.exit(1)

path = sys.argv[1]
pat = re.compile(r"(\w+)=([^\s]+)")


def pctl(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def show(name, xs):
    if not xs:
        print(f"{name}: n=0")
        return
    print(
        f"{name}: n={len(xs)} "
        f"mean={st.mean(xs):.3f} p50={pctl(xs, 0.5):.3f} "
        f"p90={pctl(xs, 0.9):.3f} p99={pctl(xs, 0.99):.3f}"
    )


s0_inline = []
s0_pool_cp = []
s0_put_pool = []
s0_pack = []
s0_desc = []
s0_credit = []
s0_credit_poll_iters = []
s0_producer_order_wait = []
s0_event_record = []
s0_ring_publish = []
s1_get_pool = []
s1_copy = []
s1_copy_wait_current = []
s1_copy_finish = []
s1_decode_enqueue = []
s1_desc_decode = []
s1_open_pool = []
s1_event_wait_enqueue = []
s1_board_release = []
s1_recv_ingress = []
save_send_task = []
save_queue_wait = []
save_total_age = []

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "CudaIPCConnector profile" not in line:
            continue
        kv = dict(pat.findall(line))
        phase = kv.get("phase")
        stage = kv.get("stage")
        try:
            elapsed = float(kv.get("elapsed_ms", "nan"))
        except ValueError:
            continue

        if phase == "put_control_plane" and stage == "0":
            route = kv.get("route")
            if route == "inline":
                s0_inline.append(elapsed)
            elif route == "pool":
                s0_pool_cp.append(elapsed)

        if phase == "put_pool" and stage == "0" and kv.get("outcome") == "ring_publish":
            s0_put_pool.append(elapsed)
            if "pack_sync_ms" in kv:
                s0_pack.append(float(kv["pack_sync_ms"]))
            if "descriptor_ser_ms" in kv:
                s0_desc.append(float(kv["descriptor_ser_ms"]))
            if "credit_wait_ms" in kv:
                s0_credit.append(float(kv["credit_wait_ms"]))
            if "credit_poll_iters" in kv:
                s0_credit_poll_iters.append(float(kv["credit_poll_iters"]))
            if "producer_order_wait_ms" in kv:
                s0_producer_order_wait.append(float(kv["producer_order_wait_ms"]))
            if "event_record_ms" in kv:
                s0_event_record.append(float(kv["event_record_ms"]))
            if "ring_publish_ms" in kv:
                s0_ring_publish.append(float(kv["ring_publish_ms"]))

        if phase == "get_control_plane" and stage == "1" and kv.get("pclass") == "pool":
            s1_get_pool.append(elapsed)
            if "copy_sync_ms" in kv:
                s1_copy.append(float(kv["copy_sync_ms"]))
            if "copy_wait_current_stream_ms" in kv:
                s1_copy_wait_current.append(float(kv["copy_wait_current_stream_ms"]))
            if "copy_finish_sync_ms" in kv:
                s1_copy_finish.append(float(kv["copy_finish_sync_ms"]))
            if "decode_enqueue_ms" in kv:
                s1_decode_enqueue.append(float(kv["decode_enqueue_ms"]))
            if "descriptor_decode_ms" in kv:
                s1_desc_decode.append(float(kv["descriptor_decode_ms"]))
            if "open_pool_ms" in kv:
                s1_open_pool.append(float(kv["open_pool_ms"]))
            if "event_wait_enqueue_ms" in kv:
                s1_event_wait_enqueue.append(float(kv["event_wait_enqueue_ms"]))
            if "board_release_ms" in kv:
                s1_board_release.append(float(kv["board_release_ms"]))
            if "recv_ingress_ms" in kv:
                s1_recv_ingress.append(float(kv["recv_ingress_ms"]))

        if "OmniConnector save_profile" in line and kv.get("phase") == "send_task":
            save_send_task.append(elapsed)
            if "queue_wait_ms" in kv:
                save_queue_wait.append(float(kv["queue_wait_ms"]))
            if "total_age_ms" in kv:
                save_total_age.append(float(kv["total_age_ms"]))

show("stage0 put_control_plane inline elapsed_ms", s0_inline)
show("stage0 put_control_plane pool elapsed_ms", s0_pool_cp)
show("stage0 put_pool(ring_publish) elapsed_ms", s0_put_pool)
show("stage0 put_pool pack_sync_ms", s0_pack)
show("stage0 put_pool descriptor_ser_ms", s0_desc)
show("stage0 put_pool credit_wait_ms", s0_credit)
show("stage0 put_pool credit_poll_iters", s0_credit_poll_iters)
show("stage0 put_pool producer_order_wait_ms", s0_producer_order_wait)
show("stage0 put_pool event_record_ms", s0_event_record)
show("stage0 put_pool ring_publish_ms", s0_ring_publish)
show("stage1 get_control_plane pool elapsed_ms", s1_get_pool)
show("stage1 get_control_plane pool copy_sync_ms", s1_copy)
show("stage1 get_control_plane pool copy_wait_current_stream_ms", s1_copy_wait_current)
show("stage1 get_control_plane pool copy_finish_sync_ms", s1_copy_finish)
show("stage1 get_control_plane pool decode_enqueue_ms", s1_decode_enqueue)
show("stage1 get_control_plane pool descriptor_decode_ms", s1_desc_decode)
show("stage1 get_control_plane pool open_pool_ms", s1_open_pool)
show("stage1 get_control_plane pool event_wait_enqueue_ms", s1_event_wait_enqueue)
show("stage1 get_control_plane pool board_release_ms", s1_board_release)
show("stage1 get_control_plane pool recv_ingress_ms", s1_recv_ingress)
show("save_loop send_task elapsed_ms", save_send_task)
show("save_loop send_task queue_wait_ms", save_queue_wait)
show("save_loop send_task total_age_ms", save_total_age)

ratio = [c / e for c, e in zip(s1_copy, s1_get_pool) if e > 0]
if ratio:
    print(
        "stage1 copy_sync/elapsed ratio: "
        f"mean={st.mean(ratio) * 100:.1f}% "
        f"p50={pctl(ratio, 0.5) * 100:.1f}% "
        f"p90={pctl(ratio, 0.9) * 100:.1f}%"
    )
PY

print_section "8) Top 20 slow stage0 put_pool rows (pack_sync_ms)"
grep -E "CudaIPCConnector profile phase=put_pool .*stage=0 .*outcome=ring_publish" "${WORK_LOG}" \
  | sed -nE 's/.*pack_sync_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 20 \
  | cut -f2- || true

print_section "9) Top 20 slow stage1 pool-get rows (copy_sync_ms)"
grep -E "CudaIPCConnector profile phase=get_control_plane .*stage=1 .*pclass=pool" "${WORK_LOG}" \
  | sed -nE 's/.*copy_sync_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 20 \
  | cut -f2- || true

print_section "10) Top 20 slow save-loop send tasks (queue_wait_ms)"
grep -E "OmniConnector save_profile phase=send_task" "${WORK_LOG}" \
  | sed -nE 's/.*queue_wait_ms=([0-9]+(\.[0-9]+)?).*/\1\t&/p' \
  | sort -t$'\t' -k1,1nr \
  | head -n 20 \
  | cut -f2- || true

print_section "Done"

if [[ "${TAIL_LINES}" -gt 0 ]] && [[ -n "${WORK_LOG:-}" ]] && [[ -f "${WORK_LOG}" ]] && [[ "${WORK_LOG}" != "${LOG_PATH}" ]]; then
  rm -f "${WORK_LOG}"
fi
