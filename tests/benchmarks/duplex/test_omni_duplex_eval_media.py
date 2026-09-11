# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_media as media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_run_ffmpeg_returns_stdout_when_decode_hangs_after_writing(monkeypatch):
    jpeg = b"\xff\xd8fake-jpeg"

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output=jpeg)

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert media._run_ffmpeg(["-i", "clip.mp4", "pipe:1"], timeout=0.05) == jpeg


def test_run_ffmpeg_times_out_without_output(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output=b"")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    with pytest.raises(TimeoutError, match="without producing output"):
        media._run_ffmpeg(["-i", "clip.mp4", "pipe:1"], timeout=0.05)


def test_iter_jpegs_stops_when_ffmpeg_times_out_empty(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1), output=b"")

    monkeypatch.setattr(media, "video_duration", lambda path: 3.0)
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert list(media.iter_jpegs(clip, fps=1.0, duration=3.0)) == []


def test_run_ffmpeg_passes_timeout_to_subprocess(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout=b"ok")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert media._run_ffmpeg(["-version"], timeout=12.5) == b"ok"
    assert seen["timeout"] == 12.5
    assert seen["capture_output"] is True
    assert seen["check"] is True
