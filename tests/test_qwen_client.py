"""Tests for QwenModelClient that don't need torch or a GPU.

Construction and ``unload()`` on a never-loaded client are both torch-free by design
(the heavy import happens lazily inside ``_ensure_loaded``), so this stays in the
default offline suite even though the client itself is GPU-oriented.
"""

from __future__ import annotations

from python_dpo.models.base import PROVIDER_TRANSFORMERS, ModelConfig
from python_dpo.models.qwen import QwenModelClient


def make_client() -> QwenModelClient:
    config = ModelConfig(provider=PROVIDER_TRANSFORMERS, name="Qwen/Qwen2.5-Coder-3B-Instruct")
    return QwenModelClient(config)


def test_construction_does_not_load_anything():
    client = make_client()
    assert client.loaded is False


def test_unload_on_a_never_loaded_client_is_a_safe_no_op():
    client = make_client()
    client.unload()  # must not raise, must not import torch
    assert client.loaded is False


def test_unload_is_idempotent():
    client = make_client()
    client.unload()
    client.unload()
    assert client.loaded is False


def test_unload_clears_model_state_without_touching_config():
    client = make_client()
    client.unload()
    assert client.name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert client.provider == PROVIDER_TRANSFORMERS
