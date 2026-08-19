"""Tests for merging a LoRA adapter into full-precision base weights (spec 12 sections
43, 44). Torch/peft/transformers are not installed here, so ``_import_backend`` is
replaced with a fake backend that records calls -- these tests exercise this module's own
orchestration (reload unquantized, merge, save, never touch ``adapter/``), not real
weight math.
"""

from __future__ import annotations

import python_dpo.packaging.merge as merge_module
import pytest

from python_dpo.packaging.errors import MergeUnsupportedError
from python_dpo.packaging.package import ModelPackage


def make_package(tmp_path) -> ModelPackage:
    root = tmp_path / "package"
    (root / "adapter").mkdir(parents=True)
    return ModelPackage(root=root, base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct", training_run_id="dpo_x")


class _FakeTokenizer:
    def __init__(self):
        self.saved_to: str | None = None

    def save_pretrained(self, path):
        self.saved_to = path


class _FakeMerged:
    def __init__(self):
        self.saved_to: str | None = None

    def save_pretrained(self, path):
        self.saved_to = path


class _FakeModel:
    def __init__(self, merged):
        self._merged = merged

    def merge_and_unload(self):
        return self._merged


def _install_fake_backend(monkeypatch, *, merge_and_unload_raises: bool = False):
    tokenizer = _FakeTokenizer()
    merged = _FakeMerged()
    calls: dict[str, object] = {"quantization_config": "unset"}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name, revision=None, trust_remote_code=False):
            return tokenizer

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, revision=None, quantization_config=None, dtype=None, device_map=None, trust_remote_code=False):
            calls["quantization_config"] = quantization_config
            return object()

    class _PeftModel:
        @staticmethod
        def from_pretrained(base, adapter_dir):
            if merge_and_unload_raises:
                class _Explodes:
                    def merge_and_unload(self):
                        raise RuntimeError("bitsandbytes cannot merge a quantized base")

                return _Explodes()
            return _FakeModel(merged)

    class _Transformers:
        AutoTokenizer = _AutoTokenizer
        AutoModelForCausalLM = _AutoModelForCausalLM

    class _Peft:
        PeftModel = _PeftModel

    class _Torch:
        bfloat16 = "bfloat16-dtype"
        float16 = "float16-dtype"
        float32 = "float32-dtype"

    def _fake_import_backend():
        return {"torch": _Torch(), "transformers": _Transformers(), "peft": _Peft()}

    monkeypatch.setattr(merge_module, "_import_backend", _fake_import_backend)
    return tokenizer, merged, calls


def test_merge_adapter_reloads_the_base_unquantized(tmp_path, monkeypatch):
    tokenizer, merged, calls = _install_fake_backend(monkeypatch)
    package = make_package(tmp_path)
    dest = tmp_path / "merged"

    result = merge_module.merge_adapter(package, dest, compute_dtype="bfloat16")

    assert result == dest
    # The merge always reloads the base with no quantization_config, regardless of what
    # the package was trained with (spec section 43): bitsandbytes cannot merge in place.
    assert calls["quantization_config"] is None
    assert merged.saved_to == str(dest)
    assert tokenizer.saved_to == str(dest)


def test_merge_adapter_never_deletes_the_original_adapter_directory(tmp_path, monkeypatch):
    _install_fake_backend(monkeypatch)
    package = make_package(tmp_path)
    dest = tmp_path / "merged"

    merge_module.merge_adapter(package, dest)

    assert package.adapter_dir.is_dir()


def test_merge_adapter_wraps_a_backend_failure(tmp_path, monkeypatch):
    _install_fake_backend(monkeypatch, merge_and_unload_raises=True)
    package = make_package(tmp_path)

    with pytest.raises(MergeUnsupportedError, match="could not merge"):
        merge_module.merge_adapter(package, tmp_path / "merged")


def test_merge_adapter_raises_when_the_backend_is_unavailable(tmp_path, monkeypatch):
    def _raise_import_error():
        raise MergeUnsupportedError("No module named 'torch'; install the training backend")

    monkeypatch.setattr(merge_module, "_import_backend", _raise_import_error)
    package = make_package(tmp_path)

    with pytest.raises(MergeUnsupportedError, match="install the training backend"):
        merge_module.merge_adapter(package, tmp_path / "merged")
