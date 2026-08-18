"""Tests for the hardware check (spec 09 sections 10, 12, 13, 51).

Every test uses an injected fake probe, so the whole module — including the BF16/FP16
fallback and the report shapes — is verified on a machine with no GPU.
"""

from __future__ import annotations

from typing import Any

from python_dpo.training.errors import TrainingDependencyError
from python_dpo.training.hardware import (
    MIN_FREE_VRAM_BYTES,
    HardwareInfo,
    check_hardware,
    format_hardware_report,
    resolve_compute_dtype,
)

GIB = 1024**3


def make_info(**overrides: Any) -> HardwareInfo:
    fields: dict[str, Any] = {
        "cuda_available": True,
        "device_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 3060",
        "total_vram_bytes": 12 * GIB,
        "free_vram_bytes": 11 * GIB,
        "compute_capability": "8.6",
        "cuda_version": "13.0",
        "torch_version": "2.13.0+cu130",
        "bf16_supported": True,
        "bitsandbytes_available": True,
    }
    fields.update(overrides)
    return HardwareInfo(**fields)


class FakeProbe:
    def __init__(self, info: HardwareInfo | Exception) -> None:
        self._info = info

    def collect(self) -> HardwareInfo:
        if isinstance(self._info, Exception):
            raise self._info
        return self._info


# ------------------------------------------------------------------------- happy path


def test_a_capable_gpu_passes_every_check():
    report = check_hardware(FakeProbe(make_info()))
    assert report.passed
    names = [check.name for check in report.checks]
    assert names == ["CUDA", "GPU", "VRAM", "BF16", "4-bit quantization"]


def test_report_records_the_collected_info():
    info = make_info()
    report = check_hardware(FakeProbe(info))
    assert report.info == info


# ----------------------------------------------------------------------------- CUDA


def test_no_cuda_fails_and_short_circuits():
    report = check_hardware(FakeProbe(make_info(cuda_available=False)))
    assert not report.passed
    # Short-circuits: no point restating the same problem four more times.
    assert len(report.checks) == 1
    assert report.checks[0].name == "CUDA"
    assert "nvidia-smi" in report.checks[0].detail


def test_a_missing_backend_is_reported_not_raised():
    probe = FakeProbe(TrainingDependencyError("No module named 'torch'; install ..."))
    report = check_hardware(probe)
    assert not report.passed
    assert report.checks[0].name == "Training backend"


# ----------------------------------------------------------------------------- VRAM


def test_insufficient_free_vram_fails():
    report = check_hardware(FakeProbe(make_info(free_vram_bytes=2 * GIB)))
    assert not report.passed
    vram = next(c for c in report.failures if c.name == "VRAM")
    assert "2.0 GiB free" in vram.detail
    assert "close other GPU processes" in vram.detail


def test_free_vram_not_total_is_what_is_checked():
    # A 12 GiB card with almost all of it held by something else must fail, even though
    # its *total* is ample. This is the desktop-session case on the development box.
    report = check_hardware(
        FakeProbe(make_info(total_vram_bytes=12 * GIB, free_vram_bytes=1 * GIB))
    )
    assert not report.passed


def test_the_vram_floor_is_configurable():
    info = make_info(free_vram_bytes=4 * GIB)
    assert not check_hardware(FakeProbe(info)).passed
    assert check_hardware(FakeProbe(info), min_free_vram_bytes=2 * GIB).passed


def test_default_floor_is_documented_value():
    assert MIN_FREE_VRAM_BYTES == 6 * GIB


# ----------------------------------------------------------------------------- BF16


def test_bf16_unsupported_still_passes_but_says_so():
    report = check_hardware(FakeProbe(make_info(bf16_supported=False)))
    assert report.passed
    bf16 = next(c for c in report.checks if c.name == "BF16")
    assert "fall back to fp16" in bf16.detail


# ---------------------------------------------------------------------- quantization


def test_missing_bitsandbytes_fails_when_quantization_is_required():
    report = check_hardware(FakeProbe(make_info(bitsandbytes_available=False)))
    assert not report.passed
    assert any(c.name == "4-bit quantization" for c in report.failures)


def test_missing_bitsandbytes_passes_when_quantization_is_not_required():
    report = check_hardware(
        FakeProbe(make_info(bitsandbytes_available=False)), require_quantization=False
    )
    assert report.passed


# ------------------------------------------------------------- compute dtype (10, 51)


def test_auto_picks_bfloat16_when_supported():
    assert resolve_compute_dtype("auto", make_info(bf16_supported=True)) == "bfloat16"


def test_auto_falls_back_to_float16():
    assert resolve_compute_dtype("auto", make_info(bf16_supported=False)) == "float16"


def test_explicit_bfloat16_is_downgraded_on_unsupported_hardware():
    # Spec section 10: never blindly request BF16.
    assert resolve_compute_dtype("bfloat16", make_info(bf16_supported=False)) == "float16"


def test_explicit_bfloat16_is_kept_when_supported():
    assert resolve_compute_dtype("bfloat16", make_info(bf16_supported=True)) == "bfloat16"


def test_explicit_float16_is_never_upgraded():
    assert resolve_compute_dtype("float16", make_info(bf16_supported=True)) == "float16"


# ------------------------------------------------------------------------ formatting


def test_passing_report_lists_every_check():
    text = format_hardware_report(check_hardware(FakeProbe(make_info())))
    assert text.startswith("Hardware check passed.")
    assert "RTX 3060" in text
    assert text.endswith("\n")


def test_failing_report_lists_only_failures():
    text = format_hardware_report(check_hardware(FakeProbe(make_info(cuda_available=False))))
    assert text.startswith("Hardware check failed:")
    assert "CUDA" in text


def test_hardware_info_round_trips():
    info = make_info()
    assert HardwareInfo.from_dict(info.to_dict()) == info
