#!/usr/bin/env bash
set -euo pipefail

# Extract the first CUDA assert-like line with context.
#
# Usage:
#   bash extract_first_cuda_assert_context.sh [log_path] [context_lines] [custom_regex]
#
# Examples:
#   bash extract_first_cuda_assert_context.sh /tmp/ipc_server.log
#   bash extract_first_cuda_assert_context.sh /tmp/ipc_server.log 120
#   bash extract_first_cuda_assert_context.sh /tmp/ipc_server.log 80 "cudaStreamWaitEvent failed"
#
# Notes:
# - Handles ANSI-colored logs by stripping escape sequences.
# - Supports .gz logs as well.

LOG_PATH="${1:-/tmp/ipc_server.log}"
CONTEXT_LINES="${2:-80}"
CUSTOM_REGEX="${3:-}"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "ERROR: log file not found: ${LOG_PATH}" >&2
  exit 1
fi

if [[ ! "${CONTEXT_LINES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: context_lines must be a non-negative integer, got: ${CONTEXT_LINES}" >&2
  exit 1
fi

python3 - "${LOG_PATH}" "${CONTEXT_LINES}" "${CUSTOM_REGEX}" <<'PY'
import gzip
import re
import sys

if len(sys.argv) < 4:
    print("ERROR: missing arguments", file=sys.stderr)
    sys.exit(1)

log_path = sys.argv[1]
context_lines = int(sys.argv[2])
custom_regex = sys.argv[3].strip()

ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

default_patterns = [
    r"TensorCompare\.cu",
    r"_assert_async_cuda_kernel",
    r"device-side assert triggered",
    r"cudaErrorAssert",
]

if custom_regex:
    search_re = re.compile(custom_regex)
else:
    search_re = re.compile("|".join(f"(?:{p})" for p in default_patterns))


def open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


lines: list[str] = []
with open_log(log_path) as f:
    for raw in f:
        line = raw.rstrip("\n").replace("\r", "")
        line = ansi_re.sub("", line)
        lines.append(line)

first_idx = None
for i, line in enumerate(lines):
    if search_re.search(line):
        first_idx = i
        break

if first_idx is None:
    print(
        "No match found for pattern:"
        f" {search_re.pattern}\n"
        "Tip: pass a custom regex as the 3rd argument.",
        file=sys.stderr,
    )
    sys.exit(2)

start = max(0, first_idx - context_lines)
end = min(len(lines), first_idx + context_lines + 1)

print("================================================================")
print(f"Log file   : {log_path}")
print(f"Pattern    : {search_re.pattern}")
print(f"First match: line {first_idx + 1}")
print("----------------------------------------------------------------")
print(lines[first_idx])
print("================================================================")
print(f"Context lines [{start + 1}, {end}]")
print("================================================================")

for i in range(start, end):
    print(f"{i + 1:7d}|{lines[i]}")
PY
