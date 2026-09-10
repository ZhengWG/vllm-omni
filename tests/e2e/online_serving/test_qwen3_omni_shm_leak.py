# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Live /dev/shm leak check for Qwen3-Omni ``async_chunk`` (stage-0-final).

The default L2 Qwen3-Omni job uses ``ci/qwen3_omni_moe.yaml`` with
``async_chunk: false`` plus ``--no-async-chunk``, so it never exercises the
thinker→talker SharedMemoryConnector path.

A text-only request is tagged ``omni_final_stage_id=0``. Nothing downstream
``get()``s a finished-marker ``put()``, and adapter ``cleanup()`` does not
unlink the POSIX segment. This module starts the production async-chunk
deploy and asserts connector lockfiles do not grow after those requests.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_USE_PD = os.environ.get("VLLM_TEST_PD_MODE", "0") == "1"
_MODEL = os.environ.get("VLLM_OMNI_TEST_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
_SHM_DIR = Path("/dev/shm")
_NUM_TEXT_REQUESTS = 3
_SETTLE_S = 15.0

# CI overlay keeps seq/token limits small; flip async_chunk back on so stage-0
# still uses SharedMemoryConnector. Production yaml is already async_chunk:true.
_ASYNC_CHUNK_DEPLOY = modify_stage_config(
    get_deploy_config_path("ci/qwen3_omni_moe.yaml"),
    updates={"async_chunk": True},
)

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_ASYNC_CHUNK_DEPLOY,
            use_stage_cli=True,
        ),
        id="async_chunk",
    )
]


def _system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def _is_connector_lockfile(name: str) -> bool:
    """SharedMemoryConnector lockfiles, not CUDA/NCCL IPC objects."""
    return name.startswith("shm_") and name.endswith("_lockfile.lock")


def _shm_used_mib() -> str:
    try:
        st = os.statvfs(_SHM_DIR)
        used = (st.f_blocks - st.f_bavail) * st.f_frsize
        return f"{used / (1024 * 1024):.1f}MiB"
    except OSError:
        return "unknown"


def _connector_lockfile_count() -> int:
    if not _SHM_DIR.is_dir():
        pytest.skip("/dev/shm is not available")
    with os.scandir(_SHM_DIR) as entries:
        return sum(1 for entry in entries if _is_connector_lockfile(entry.name))


def _connector_lockfiles_since(since_s: float, limit: int = 32) -> list[str]:
    """Return a sample of connector lockfiles with mtime >= ``since_s``."""
    found: list[str] = []
    with os.scandir(_SHM_DIR) as entries:
        for entry in entries:
            if not _is_connector_lockfile(entry.name):
                continue
            try:
                if entry.stat().st_mtime >= since_s:
                    found.append(entry.name)
                    if len(found) >= limit:
                        break
            except FileNotFoundError:
                continue
    return found


def _wait_no_new_lockfiles(since_s: float, baseline: int, timeout_s: float = _SETTLE_S) -> tuple[int, list[str]]:
    deadline = time.monotonic() + timeout_s
    time.sleep(1.0)
    while True:
        count = _connector_lockfile_count()
        extra = _connector_lockfiles_since(since_s)
        if count <= baseline and not extra:
            return count, extra
        if time.monotonic() >= deadline:
            return count, extra
        time.sleep(2.0)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.skipif(_USE_PD, reason="Temporarily skip PD mode in this test module.")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_only_async_chunk_does_not_leak_shm(omni_server, online_client) -> None:
    """Stage-0-final (text-only) requests must not leave connector SHM behind."""
    # Timestamp first so the scan below does not treat its own runtime as "new".
    since_s = time.time()
    baseline = _connector_lockfile_count()
    print(
        f"[shm-leak] before requests: lockfiles={baseline} /dev/shm_used={_shm_used_mib()}",
        flush=True,
    )

    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in 20 words.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }
    responses = online_client.send_omni_request(request_config, request_num=_NUM_TEXT_REQUESTS)
    produced = [resp.text_content for resp in responses if getattr(resp, "text_content", None)]
    assert len(produced) == _NUM_TEXT_REQUESTS, (
        f"Need {_NUM_TEXT_REQUESTS} text completions to exercise stage-0 finish; "
        f"got {len(produced)} non-empty texts. A preprocess/engine miss would "
        f"leave /dev/shm unchanged and hide a leak."
    )

    after, extra = _wait_no_new_lockfiles(since_s, baseline)
    print(
        f"[shm-leak] after {_NUM_TEXT_REQUESTS} text-only requests: "
        f"lockfiles={after} /dev/shm_used={_shm_used_mib()} new_sample={extra}",
        flush=True,
    )
    assert after <= baseline and not extra, (
        f"SharedMemoryConnector leaked lockfiles after {_NUM_TEXT_REQUESTS} "
        f"stage-0-final request(s): before={baseline} after={after} new_sample={extra}"
    )
