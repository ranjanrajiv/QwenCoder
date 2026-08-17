"""Tests for the model abstraction, the mock client, and the Qwen client's pure parts.

Nothing here loads a model, needs a GPU, or touches the network.
"""

from __future__ import annotations

import sys

import pytest

from python_dpo.models import (
    PROVIDER_MOCK,
    PROVIDER_TRANSFORMERS,
    GenerationConfig,
    InferenceError,
    MockModelClient,
    ModelClient,
    ModelConfig,
    ModelError,
    ModelLoadError,
    QwenModelClient,
    RawGeneration,
)
from python_dpo.models.qwen import build_generation_kwargs, resolve_device, resolve_dtype


# --------------------------------------------------------------------------- config


def test_generation_config_defaults_are_valid():
    config = GenerationConfig()
    assert config.temperature == pytest.approx(0.8)
    assert config.max_new_tokens == 512
    assert config.do_sample is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"temperature": -0.5},
        {"repetition_penalty": 0.0},
        {"seed": -1},
        {"temperature": "hot"},
        {"do_sample": "yes"},
        # bool is an int subclass; accepting it here would silently mean 1 token.
        {"max_new_tokens": True},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs):
    with pytest.raises(ModelError):
        GenerationConfig(**kwargs)


def test_generation_config_rejects_zero_temperature_while_sampling():
    with pytest.raises(ModelError, match="do_sample"):
        GenerationConfig(do_sample=True, temperature=0.0)


def test_generation_config_allows_zero_temperature_when_greedy():
    assert GenerationConfig(do_sample=False, temperature=0.0).temperature == 0.0


def test_generation_config_round_trips_through_dict():
    config = GenerationConfig(temperature=0.5, seed=7)
    assert GenerationConfig.from_dict(config.to_dict()) == config


def test_generation_config_from_dict_rejects_unknown_field():
    with pytest.raises(ModelError, match="unknown field"):
        GenerationConfig.from_dict({"temperature": 0.5, "top_k": 40})


def test_model_config_accepts_indexed_cuda_device():
    assert ModelConfig(provider=PROVIDER_TRANSFORMERS, name="x", device="cuda:1").device == "cuda:1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "openai", "name": "x"},
        {"provider": PROVIDER_TRANSFORMERS, "name": ""},
        {"provider": PROVIDER_TRANSFORMERS, "name": "x", "device": "tpu"},
        {"provider": PROVIDER_TRANSFORMERS, "name": "x", "device": "cuda:abc"},
        {"provider": PROVIDER_TRANSFORMERS, "name": "x", "dtype": "float8"},
        {"provider": PROVIDER_TRANSFORMERS, "name": "x", "trust_remote_code": "yes"},
        {"provider": PROVIDER_TRANSFORMERS, "name": "x", "revision": ""},
    ],
)
def test_model_config_rejects_invalid_values(kwargs):
    with pytest.raises(ModelError):
        ModelConfig(**kwargs)


def test_model_config_rejects_quantization_until_it_is_implemented():
    with pytest.raises(ModelError, match="quantization"):
        ModelConfig(provider=PROVIDER_TRANSFORMERS, name="x", quantization="4bit")


def test_model_config_defaults_trust_remote_code_to_false():
    assert ModelConfig(provider=PROVIDER_TRANSFORMERS, name="x").trust_remote_code is False


def test_model_config_from_mapping_applies_defaults():
    config = ModelConfig.from_mapping({"provider": PROVIDER_TRANSFORMERS, "name": "Qwen/x"})
    assert config.device == "auto"
    assert config.dtype == "auto"
    assert config.revision is None


@pytest.mark.parametrize(
    "mapping, match",
    [
        ({"provider": PROVIDER_TRANSFORMERS}, "missing required"),
        ({"provider": PROVIDER_TRANSFORMERS, "name": "x", "cache": "y"}, "unknown field"),
        ("not a mapping", "expected a mapping"),
    ],
)
def test_model_config_from_mapping_rejects_bad_input(mapping, match):
    with pytest.raises(ModelError, match=match):
        ModelConfig.from_mapping(mapping)


# ----------------------------------------------------------------------- mock client


def test_mock_client_satisfies_the_protocol():
    assert isinstance(MockModelClient(), ModelClient)
    assert MockModelClient().provider == PROVIDER_MOCK


def test_mock_client_is_deterministic_across_instances():
    first = MockModelClient().generate("a prompt", GenerationConfig())
    second = MockModelClient().generate("a prompt", GenerationConfig())
    assert first.text == second.text
    assert isinstance(first, RawGeneration)


def test_mock_client_varies_output_by_prompt():
    config = GenerationConfig()
    outputs = {
        MockModelClient().generate(f"prompt variant {n}", config).text for n in range(5)
    }
    assert len(outputs) == 5


def test_mock_client_uses_the_signature_from_the_prompt():
    prompt = "Required function signature:\ndef first_unique(text):"
    assert "def first_unique(" in MockModelClient().generate(prompt, GenerationConfig()).text


def test_mock_client_follows_its_script_in_order():
    client = MockModelClient(script=["first", "second"])
    config = GenerationConfig()
    assert client.generate("p", config).text == "first"
    assert client.generate("p", config).text == "second"
    # Past the end of the script it falls back to synthesized output.
    assert client.generate("p", config).text not in {"first", "second"}
    assert client.call_count == 3


def test_mock_client_raises_scripted_exceptions():
    client = MockModelClient(script=[InferenceError("boom")])
    with pytest.raises(InferenceError, match="boom"):
        client.generate("p", GenerationConfig())


# ----------------------------------------------------------------------- qwen client


def test_qwen_client_satisfies_the_protocol_without_loading():
    client = QwenModelClient(ModelConfig(provider=PROVIDER_TRANSFORMERS, name="Qwen/x"))
    assert isinstance(client, ModelClient)
    assert client.name == "Qwen/x"
    assert client.revision is None
    assert client.provider == PROVIDER_TRANSFORMERS
    assert client.loaded is False


def test_qwen_client_rejects_a_non_transformers_provider():
    with pytest.raises(ModelLoadError, match="provider"):
        QwenModelClient(ModelConfig(provider=PROVIDER_MOCK, name="mock/x"))


def test_missing_backend_produces_an_install_hint(monkeypatch):
    # A None entry in sys.modules makes `import torch` raise ImportError, simulating an
    # environment without the optional [model] extra installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    client = QwenModelClient(ModelConfig(provider=PROVIDER_TRANSFORMERS, name="Qwen/x"))
    with pytest.raises(ModelLoadError, match=r"\[model\]"):
        client.generate("prompt", GenerationConfig())


@pytest.mark.parametrize(
    "requested, cuda_available, expected",
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda:1", True, "cuda:1"),
    ],
)
def test_resolve_device(requested, cuda_available, expected):
    assert resolve_device(requested, cuda_available=cuda_available) == expected


def test_resolve_device_refuses_to_silently_downgrade_to_cpu():
    with pytest.raises(ModelLoadError, match="CUDA is not available"):
        resolve_device("cuda", cuda_available=False)


@pytest.mark.parametrize(
    "requested, device, expected",
    [
        ("auto", "cuda", "bfloat16"),
        ("auto", "cuda:0", "bfloat16"),
        ("auto", "cpu", "float32"),
        ("float16", "cuda", "float16"),
        ("float32", "cuda", "float32"),
    ],
)
def test_resolve_dtype(requested, device, expected):
    assert resolve_dtype(requested, device) == expected


def test_generation_kwargs_include_sampling_parameters_only_when_sampling():
    sampled = build_generation_kwargs(GenerationConfig(do_sample=True))
    assert sampled["temperature"] == pytest.approx(0.8)
    assert sampled["top_p"] == pytest.approx(0.95)

    greedy = build_generation_kwargs(GenerationConfig(do_sample=False, temperature=0.0))
    assert "temperature" not in greedy
    assert "top_p" not in greedy
    assert greedy["max_new_tokens"] == 512
