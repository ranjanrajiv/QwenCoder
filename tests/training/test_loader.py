"""Tests for the LoRA safety checks (spec 09 sections 16, 19, 20, 50).

These are the stage's real safety net, so they are tested against fake models rather than
only through a GPU run: a regression here would silently produce either a no-op adapter or
a full fine-tune, and neither announces itself.

The fakes expose only ``named_modules()`` and ``parameters()``, which is all the functions
under test touch — so no torch is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from python_dpo.training.config import FALLBACK_OPTIMIZER, LoraSettings
from python_dpo.training.errors import FullFineTuneError, TargetModuleError
from python_dpo.training.loader import (
    ParameterCounts,
    count_parameters,
    resolve_optimizer,
    validate_target_modules,
)


@dataclass
class FakeParameter:
    numel_value: int
    requires_grad: bool

    def numel(self) -> int:
        return self.numel_value


class FakeModel:
    def __init__(self, module_names: list[str], params: list[FakeParameter]) -> None:
        self._module_names = module_names
        self._params = params

    def named_modules(self):
        return [(name, object()) for name in self._module_names]

    def parameters(self):
        return list(self._params)


def qwen_like_modules() -> list[str]:
    """A module tree shaped like the real Qwen2 attention stack."""
    names = [""]
    for layer in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            names.append(f"model.layers.{layer}.self_attn.{proj}")
    return names


# --------------------------------------------------------------- section 16 (targets)


def test_all_configured_targets_present():
    model = FakeModel(qwen_like_modules(), [])
    resolved = validate_target_modules(model, ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert resolved == ("k_proj", "o_proj", "q_proj", "v_proj")


def test_partial_match_keeps_the_ones_that_exist(caplog):
    model = FakeModel(qwen_like_modules(), [])
    resolved = validate_target_modules(model, ("q_proj", "nonexistent_proj"))
    assert resolved == ("q_proj",)
    assert any("do not exist" in r.message for r in caplog.records)


def test_no_matching_target_raises():
    """Training with zero LoRA targets would produce an adapter that changes nothing."""
    model = FakeModel(qwen_like_modules(), [])
    with pytest.raises(TargetModuleError, match="none of the configured"):
        validate_target_modules(model, ("wrong_a", "wrong_b"))


def test_module_matching_uses_the_leaf_name():
    # Real module names are dotted paths; the configuration names only the leaf.
    model = FakeModel(["model.layers.0.self_attn.q_proj"], [])
    assert validate_target_modules(model, ("q_proj",)) == ("q_proj",)


# ------------------------------------------------------ sections 19, 20 (parameters)


def test_counts_total_and_trainable():
    model = FakeModel(
        [],
        [FakeParameter(1_000_000, False), FakeParameter(1_000, True)],
    )
    counts = count_parameters(model)
    assert counts.total == 1_001_000
    assert counts.trainable == 1_000
    assert counts.percentage == pytest.approx(0.0999, rel=1e-2)


def test_full_fine_tune_is_refused():
    """Spec section 20: the safety trip. Every parameter trainable means LoRA did not
    apply, and this would be a full fine-tune."""
    model = FakeModel([], [FakeParameter(1_000, True), FakeParameter(2_000, True)])
    with pytest.raises(FullFineTuneError, match="not frozen"):
        count_parameters(model)


def test_zero_trainable_parameters_is_refused():
    # The opposite failure: LoRA never attached, so training would be a silent no-op.
    model = FakeModel([], [FakeParameter(1_000, False)])
    with pytest.raises(FullFineTuneError, match="no parameters are trainable"):
        count_parameters(model)


def test_zero_parameters_is_refused():
    with pytest.raises(FullFineTuneError, match="zero parameters"):
        count_parameters(FakeModel([], []))


def test_realistic_qlora_ratio_passes():
    # The real shape: 7.37M trainable LoRA parameters against a 1.7B quantized base.
    model = FakeModel(
        [], [FakeParameter(1_706_045_440 - 7_372_800, False), FakeParameter(7_372_800, True)]
    )
    counts = count_parameters(model)
    assert counts.trainable == 7_372_800
    assert counts.percentage < 1.0


def test_parameter_counts_serialize():
    counts = ParameterCounts(total=1000, trainable=10)
    assert counts.to_dict() == {
        "total_parameters": 1000,
        "trainable_parameters": 10,
        "trainable_percentage": 1.0,
    }


# ------------------------------------------------------------- section 50 (optimizer)


def test_non_paged_optimizer_is_returned_unchanged():
    assert resolve_optimizer("adamw_torch") == ("adamw_torch", False)


def test_paged_optimizer_is_kept_when_bitsandbytes_is_available():
    pytest.importorskip("bitsandbytes")
    assert resolve_optimizer("paged_adamw_8bit") == ("paged_adamw_8bit", False)


def test_paged_optimizer_falls_back_without_bitsandbytes(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bitsandbytes":
            raise ImportError("no bitsandbytes")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    name, fell_back = resolve_optimizer("paged_adamw_8bit")
    assert name == FALLBACK_OPTIMIZER
    assert fell_back is True


# ------------------------------------------------------------------------ lora config


def test_lora_settings_reject_empty_targets():
    from python_dpo.training.errors import TrainingConfigError

    with pytest.raises(TrainingConfigError):
        LoraSettings(target_modules=())
