# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remote MAGVITv2 imports ``FLAX_WEIGHTS_NAME`` from ``diffusers.utils``.
Current diffusers no longer exports that Flax constant, so Dynin stage-1
load failed during ``trust_remote_code``.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_stub_diffusers_utils(monkeypatch: pytest.MonkeyPatch, *, with_flax: bool = False):
    pkg = types.ModuleType("diffusers")
    utils = types.ModuleType("diffusers.utils")
    if with_flax:
        utils.FLAX_WEIGHTS_NAME = "already-present"
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", pkg)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    return utils


def test_legacy_flax_weights_name_is_restored_when_missing(monkeypatch):
    from vllm_omni.model_executor.models.dynin_omni.dynin_omni_common import (
        _ensure_legacy_diffusers_utils_exports,
    )

    utils = _install_stub_diffusers_utils(monkeypatch)
    assert not hasattr(utils, "FLAX_WEIGHTS_NAME")

    _ensure_legacy_diffusers_utils_exports()

    assert utils.FLAX_WEIGHTS_NAME == "diffusion_flax_model.msgpack"


def test_legacy_flax_weights_name_is_left_alone_when_present(monkeypatch):
    from vllm_omni.model_executor.models.dynin_omni.dynin_omni_common import (
        _ensure_legacy_diffusers_utils_exports,
    )

    utils = _install_stub_diffusers_utils(monkeypatch, with_flax=True)

    _ensure_legacy_diffusers_utils_exports()

    assert utils.FLAX_WEIGHTS_NAME == "already-present"


def test_load_remote_module_can_import_flax_weights_name(tmp_path, monkeypatch):
    from vllm_omni.model_executor.models.dynin_omni import dynin_omni_common as dynin_common

    _install_stub_diffusers_utils(monkeypatch)
    (tmp_path / "modeling_utils.py").write_text(
        "from diffusers.utils import FLAX_WEIGHTS_NAME\nVALUE = FLAX_WEIGHTS_NAME\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dynin_common, "_resolve_remote_snapshot_dir", lambda **kwargs: str(tmp_path))

    loaded = dynin_common._load_remote_module(
        module_name="modeling_utils",
        source="snu-aidas/magvitv2",
        revision=None,
        local_files_only=True,
    )

    assert loaded.VALUE == "diffusion_flax_model.msgpack"
