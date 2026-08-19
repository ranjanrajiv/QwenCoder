"""Tests for single-prompt and batch generation (spec 12 sections 39, 40).

Torch/transformers/peft are not installed in this environment (by design --
``tests/test_no_heavy_imports.py`` asserts the whole package stays importable without
them), so these tests replace ``AdapterModelRunner`` itself with a fake that records calls
and returns scripted text, exercising only this module's own orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import python_dpo.packaging.inference as inference
from python_dpo.atomic_io import append_jsonl
from python_dpo.model_evaluation.config import GenerationSettings, QuantizationSettings
from python_dpo.packaging.errors import PackagingError
from python_dpo.packaging.package import ModelPackage


@dataclass
class _FakeGeneration:
    text: str
    generation_time_ms: int = 5
    generated_tokens: int = 3


class _FakeAdapterRunner:
    instances: list["_FakeAdapterRunner"] = []

    def __init__(self, *, model_name, model_revision, adapter_dir, quantization, generation):
        self.model_name = model_name
        self.adapter_dir = adapter_dir
        self.loaded = False
        self.unloaded = False
        self.prompts: list[tuple[str, int]] = []
        _FakeAdapterRunner.instances.append(self)

    def ensure_loaded(self):
        self.loaded = True

    def generate(self, prompt, *, seed):
        self.prompts.append((prompt, seed))
        return _FakeGeneration(text=f"response to: {prompt}")

    def unload(self):
        self.unloaded = True


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch):
    _FakeAdapterRunner.instances = []
    monkeypatch.setattr(inference, "AdapterModelRunner", _FakeAdapterRunner)
    return _FakeAdapterRunner


def make_package(tmp_path) -> ModelPackage:
    root = tmp_path / "package"
    (root / "adapter").mkdir(parents=True)
    return ModelPackage(root=root, base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct", training_run_id="dpo_x")


def test_generate_loads_generates_and_unloads(tmp_path, fake_runner):
    package = make_package(tmp_path)

    text = inference.generate(
        package, "reverse a string",
        quantization=QuantizationSettings(), generation=GenerationSettings(), seed=7,
    )

    assert text == "response to: reverse a string"
    [runner] = fake_runner.instances
    assert runner.loaded is True
    assert runner.unloaded is True
    assert runner.prompts == [("reverse a string", 7)]


def test_generate_unloads_even_if_generation_raises(tmp_path, fake_runner):
    package = make_package(tmp_path)

    class _RaisingRunner(fake_runner):
        def generate(self, prompt, *, seed):
            raise RuntimeError("boom")

    import python_dpo.packaging.inference as inference_module

    inference_module.AdapterModelRunner = _RaisingRunner
    with pytest.raises(RuntimeError):
        inference.generate(
            package, "x", quantization=QuantizationSettings(), generation=GenerationSettings()
        )
    [runner] = _RaisingRunner.instances
    assert runner.unloaded is True


def test_generate_batch_loads_the_model_once_for_every_prompt(tmp_path, fake_runner):
    package = make_package(tmp_path)
    input_path = tmp_path / "prompts.jsonl"
    append_jsonl(input_path, {"id": "a", "prompt": "one"})
    append_jsonl(input_path, {"id": "b", "prompt": "two"})
    output_path = tmp_path / "out.jsonl"

    count = inference.generate_batch(
        package, input_path, output_path,
        quantization=QuantizationSettings(), generation=GenerationSettings(),
    )

    assert count == 2
    [runner] = fake_runner.instances
    assert runner.prompts == [("one", 42), ("two", 42)]
    from python_dpo.atomic_io import iter_jsonl

    records = [record for _, record in iter_jsonl(output_path)]
    assert records[0]["response"] == "response to: one"
    assert records[1]["id"] == "b"


def test_generate_batch_requires_an_existing_input_file(tmp_path):
    package = make_package(tmp_path)
    with pytest.raises(PackagingError, match="no input JSONL"):
        inference.generate_batch(
            package, tmp_path / "missing.jsonl", tmp_path / "out.jsonl",
            quantization=QuantizationSettings(), generation=GenerationSettings(),
        )


def test_generate_batch_rejects_a_record_with_no_prompt(tmp_path, fake_runner):
    package = make_package(tmp_path)
    input_path = tmp_path / "prompts.jsonl"
    append_jsonl(input_path, {"id": "a"})

    with pytest.raises(PackagingError, match="missing or empty 'prompt'"):
        inference.generate_batch(
            package, input_path, tmp_path / "out.jsonl",
            quantization=QuantizationSettings(), generation=GenerationSettings(),
        )
    [runner] = fake_runner.instances
    assert runner.unloaded is True
