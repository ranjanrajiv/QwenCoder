"""Hardware detection and the preflight capability check (spec 09 sections 12, 13, 85).

Mirrors :mod:`python_dpo.sandbox.health` deliberately: frozen ``HardwareCheck`` /
``HardwareReport`` records, checks that short-circuit at the first failure, every failure
detail ending in something the user can act on, and a separate formatter. A user must not
be handed a raw traceback where a sentence would do.

The torch import is deferred into :class:`TorchHardwareProbe`, and every function here
takes an **injectable probe**, so the whole module — including the report shapes and the
BF16/FP16 fallback logic — is unit-testable on a machine with no GPU and no torch.

**Free VRAM, not total, is what matters.** This project's development box runs a desktop
session that permanently holds several hundred MB of the RTX 3060's 12 GB. Reporting total
VRAM would overstate the budget by exactly the amount most likely to cause an OOM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .errors import TrainingDependencyError

# A 4-bit 3B model plus LoRA, optimizer state, and activations does not fit in much less
# than this. Deliberately a floor for a *usable* run, not a hard architectural limit —
# check_hardware reports it as a failure with the numbers, and the caller decides.
MIN_FREE_VRAM_BYTES = 6 * 1024**3

_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class HardwareInfo:
    """What the machine actually offers (spec section 12), persisted as hardware.json."""

    cuda_available: bool
    device_count: int = 0
    gpu_name: str | None = None
    total_vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    compute_capability: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    bf16_supported: bool = False
    bitsandbytes_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "gpu_name": self.gpu_name,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "compute_capability": self.compute_capability,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "bf16_supported": self.bf16_supported,
            "bitsandbytes_available": self.bitsandbytes_available,
        }

    @classmethod
    def from_dict(cls, data: Any) -> HardwareInfo:
        if not isinstance(data, dict):
            raise ValueError("hardware info: expected a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class HardwareCheck:
    """One step of the hardware check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class HardwareReport:
    checks: tuple[HardwareCheck, ...]
    info: HardwareInfo

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[HardwareCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


class HardwareProbe(Protocol):
    """The seam that keeps this module testable without a GPU."""

    def collect(self) -> HardwareInfo: ...


class TorchHardwareProbe:
    """The real probe. Imports torch lazily, on call rather than on construction."""

    def collect(self) -> HardwareInfo:
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError(
                f"{exc}; install the training backend with: pip install -e '.[training]'"
            ) from exc

        bitsandbytes_available = _bitsandbytes_available()
        torch_version = getattr(torch, "__version__", None)

        if not torch.cuda.is_available():
            return HardwareInfo(
                cuda_available=False,
                torch_version=torch_version,
                bitsandbytes_available=bitsandbytes_available,
            )

        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        # mem_get_info reports (free, total) for the *device*, so it accounts for memory
        # held by other processes — a desktop session, another training job — which
        # total_memory alone would hide.
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)

        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001 - an old torch may not expose it at all
            bf16 = properties.major >= 8

        return HardwareInfo(
            cuda_available=True,
            device_count=torch.cuda.device_count(),
            gpu_name=properties.name,
            total_vram_bytes=int(total_bytes),
            free_vram_bytes=int(free_bytes),
            compute_capability=f"{properties.major}.{properties.minor}",
            cuda_version=getattr(torch.version, "cuda", None),
            torch_version=torch_version,
            bf16_supported=bf16,
            bitsandbytes_available=bitsandbytes_available,
        )


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken CUDA build raises more than ImportError
        return False
    return True


def check_hardware(
    probe: HardwareProbe | None = None,
    *,
    min_free_vram_bytes: int = MIN_FREE_VRAM_BYTES,
    require_quantization: bool = True,
) -> HardwareReport:
    """Run the spec section 12/13 checks, stopping at the first that fails.

    Stopping early is deliberate, as in the sandbox health check: if CUDA is unavailable
    there is no value in four more failures that all restate it.
    """
    probe = probe if probe is not None else TorchHardwareProbe()
    checks: list[HardwareCheck] = []

    try:
        info = probe.collect()
    except TrainingDependencyError as exc:
        return HardwareReport(
            (HardwareCheck("Training backend", False, str(exc)),),
            HardwareInfo(cuda_available=False),
        )

    if not info.cuda_available:
        checks.append(
            HardwareCheck(
                "CUDA",
                False,
                "no CUDA device is available; DPO/QLoRA training requires a CUDA GPU. "
                "Check `nvidia-smi` and that torch was installed with CUDA support",
            )
        )
        return HardwareReport(tuple(checks), info)
    checks.append(
        HardwareCheck(
            "CUDA",
            True,
            f"CUDA {info.cuda_version} via torch {info.torch_version}, "
            f"{info.device_count} device(s)",
        )
    )

    checks.append(
        HardwareCheck(
            "GPU",
            True,
            f"{info.gpu_name} (compute capability {info.compute_capability})",
        )
    )

    free_gib = (info.free_vram_bytes or 0) / _BYTES_PER_GIB
    total_gib = (info.total_vram_bytes or 0) / _BYTES_PER_GIB
    if (info.free_vram_bytes or 0) < min_free_vram_bytes:
        checks.append(
            HardwareCheck(
                "VRAM",
                False,
                f"{free_gib:.1f} GiB free of {total_gib:.1f} GiB total, below the "
                f"{min_free_vram_bytes / _BYTES_PER_GIB:.1f} GiB needed; close other GPU "
                "processes or reduce max_length / batch size",
            )
        )
        return HardwareReport(tuple(checks), info)
    checks.append(
        HardwareCheck("VRAM", True, f"{free_gib:.1f} GiB free of {total_gib:.1f} GiB total")
    )

    checks.append(
        HardwareCheck(
            "BF16",
            True,
            "supported" if info.bf16_supported else "not supported; will fall back to fp16",
        )
    )

    if require_quantization and not info.bitsandbytes_available:
        checks.append(
            HardwareCheck(
                "4-bit quantization",
                False,
                "bitsandbytes is not importable, so 4-bit loading is unavailable; "
                "install it with: pip install -e '.[training]'",
            )
        )
        return HardwareReport(tuple(checks), info)
    checks.append(
        HardwareCheck(
            "4-bit quantization",
            True,
            "bitsandbytes is available" if info.bitsandbytes_available else "not required",
        )
    )

    return HardwareReport(tuple(checks), info)


def resolve_compute_dtype(configured: str, info: HardwareInfo) -> str:
    """Pick the actual compute dtype (spec sections 10, 51).

    ``bfloat16`` is downgraded to ``float16`` on hardware that cannot do it, rather than
    being requested blindly. ``auto`` picks bfloat16 where supported. The two are never
    enabled together — the caller derives mutually exclusive ``bf16``/``fp16`` flags from
    this single answer.
    """
    if configured == "auto":
        return "bfloat16" if info.bf16_supported else "float16"
    if configured == "bfloat16" and not info.bf16_supported:
        return "float16"
    return configured


def format_hardware_report(report: HardwareReport) -> str:
    """Render the spec section 13 report."""
    if report.passed:
        lines = ["Hardware check passed."]
        lines += [f"  {check.name}: {check.detail}" for check in report.checks]
        return "\n".join(lines) + "\n"

    lines = ["Hardware check failed:"]
    lines += [f"  {check.name}: {check.detail}" for check in report.failures]
    return "\n".join(lines) + "\n"


__all__ = [
    "MIN_FREE_VRAM_BYTES",
    "HardwareCheck",
    "HardwareInfo",
    "HardwareProbe",
    "HardwareReport",
    "TorchHardwareProbe",
    "check_hardware",
    "format_hardware_report",
    "resolve_compute_dtype",
]
