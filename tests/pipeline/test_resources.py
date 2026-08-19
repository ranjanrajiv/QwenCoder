"""Tests for stage durations and the best-effort host resource snapshot (spec 12
section 51)."""

from __future__ import annotations

from python_dpo.pipeline.manifest import StageManifest
from python_dpo.pipeline.resources import capture_resource_snapshot, stage_durations


def make_stage(name: str, *, start: str | None, end: str | None, status: str = "COMPLETED") -> StageManifest:
    return StageManifest(
        stage_name=name, stage_run_id=f"{name}_run", status=status, code_version="0.12.0",
        start_time=start, end_time=end,
    )


def test_stage_durations_computes_elapsed_seconds():
    stages = {
        "problem_dataset": make_stage(
            "problem_dataset", start="2026-08-19T10:00:00Z", end="2026-08-19T10:00:30Z"
        ),
    }
    assert stage_durations(stages) == {"problem_dataset": 30.0}


def test_stage_durations_omits_stages_with_no_timestamps():
    stages = {
        "problem_dataset": make_stage("problem_dataset", start=None, end=None, status="SKIPPED"),
        "candidate_generation": make_stage(
            "candidate_generation", start="2026-08-19T10:00:00Z", end="2026-08-19T10:01:00Z"
        ),
    }
    assert stage_durations(stages) == {"candidate_generation": 60.0}


def test_stage_durations_handles_multiple_stages():
    stages = {
        "a": make_stage("problem_dataset", start="2026-08-19T10:00:00Z", end="2026-08-19T10:00:10Z"),
        "b": make_stage(
            "candidate_generation", start="2026-08-19T10:00:10Z", end="2026-08-19T10:02:10Z"
        ),
    }
    result = stage_durations(stages)
    assert result["a"] == 10.0
    assert result["b"] == 120.0


def test_capture_resource_snapshot_never_raises_and_fields_are_the_right_shape():
    """Whatever this host actually has (or lacks), the call must not raise and every
    field must be present, typed correctly or None -- never a fabricated value."""
    snapshot = capture_resource_snapshot()
    data = snapshot.to_dict()

    assert set(data) == {
        "cpu_percent", "ram_used_bytes", "ram_total_bytes",
        "gpu_name", "gpu_utilization_percent", "gpu_memory_used_bytes", "gpu_memory_total_bytes",
    }
    for key in ("cpu_percent", "ram_used_bytes", "ram_total_bytes"):
        assert data[key] is None or isinstance(data[key], (int, float))
    for key in ("gpu_utilization_percent", "gpu_memory_used_bytes", "gpu_memory_total_bytes"):
        assert data[key] is None or isinstance(data[key], (int, float))
    assert data["gpu_name"] is None or isinstance(data["gpu_name"], str)
