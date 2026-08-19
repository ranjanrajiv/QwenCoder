"""Tests for building and reading a model package (spec 12 sections 35-37)."""

from __future__ import annotations

import pytest

from python_dpo.packaging.errors import PackagingError
from python_dpo.packaging.package import ModelPackage, build_package

from .conftest import make_training_run


def test_build_package_copies_adapter_and_tokenizer_but_never_the_base_model(tmp_path):
    repo, training_run_id = make_training_run(tmp_path)
    dest = tmp_path / "packages" / "p1"

    package = build_package(
        dest_dir=dest,
        training_run_id=training_run_id,
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision=None,
        adapter_source=repo.adapter_dir(training_run_id),
        tokenizer_source=repo.tokenizer_dir(training_run_id),
        quantization={"enabled": True, "bits": 4},
        created_at="2026-08-19T10:00:00Z",
    )

    assert (dest / "adapter" / "adapter_config.json").is_file()
    assert (dest / "tokenizer" / "tokenizer_config.json").is_file()
    assert (dest / "manifest.json").is_file()
    # No file or directory named after the base model was created -- it is referenced by
    # name only (spec section 36).
    assert not (dest / "Qwen").exists()
    assert package.base_model_name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert package.training_run_id == training_run_id


def test_build_package_excludes_trls_frozen_reference_adapter_and_duplicate_tokenizer(tmp_path):
    """TRL's DPOTrainer saves a frozen "ref/" copy of the adapter (never loaded by
    anything -- AdapterModelRunner fetches the tokenizer from the base model by name, not
    from the adapter directory) plus its own tokenizer.json duplicate, alongside the real
    adapter. Copying those into a package would roughly quadruple its size for nothing
    (spec section 42's "usable standalone" is about the adapter, not TRL's bookkeeping)."""
    repo, training_run_id = make_training_run(tmp_path)
    adapter_dir = repo.adapter_dir(training_run_id)
    (adapter_dir / "ref").mkdir()
    (adapter_dir / "ref" / "adapter_model.safetensors").write_bytes(b"frozen-reference-weights")
    (adapter_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "packages" / "p1"

    build_package(
        dest_dir=dest,
        training_run_id=training_run_id,
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision=None,
        adapter_source=adapter_dir,
        tokenizer_source=repo.tokenizer_dir(training_run_id),
        quantization={},
        created_at="2026-08-19T10:00:00Z",
    )

    assert not (dest / "adapter" / "ref").exists()
    assert not (dest / "adapter" / "tokenizer.json").exists()
    assert (dest / "adapter" / "adapter_config.json").is_file()


def test_build_package_rejects_a_training_run_with_no_adapter(tmp_path):
    repo, training_run_id = make_training_run(tmp_path, with_adapter=False)

    with pytest.raises(PackagingError, match="no adapter"):
        build_package(
            dest_dir=tmp_path / "packages" / "p1",
            training_run_id=training_run_id,
            base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            base_model_revision=None,
            adapter_source=repo.adapter_dir(training_run_id),
            tokenizer_source=repo.tokenizer_dir(training_run_id),
            quantization={},
            created_at="2026-08-19T10:00:00Z",
        )


def test_build_package_tolerates_a_missing_tokenizer_directory(tmp_path):
    repo, training_run_id = make_training_run(tmp_path, with_tokenizer=False)

    package = build_package(
        dest_dir=tmp_path / "packages" / "p1",
        training_run_id=training_run_id,
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision=None,
        adapter_source=repo.adapter_dir(training_run_id),
        tokenizer_source=repo.tokenizer_dir(training_run_id),
        quantization={},
        created_at="2026-08-19T10:00:00Z",
    )

    assert package.tokenizer_dir.is_dir()
    assert not any(package.tokenizer_dir.iterdir())


def test_load_round_trips_a_built_package(tmp_path):
    repo, training_run_id = make_training_run(tmp_path)
    dest = tmp_path / "packages" / "p1"
    build_package(
        dest_dir=dest,
        training_run_id=training_run_id,
        base_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        base_model_revision="main",
        adapter_source=repo.adapter_dir(training_run_id),
        tokenizer_source=repo.tokenizer_dir(training_run_id),
        quantization={"enabled": True, "bits": 4},
        created_at="2026-08-19T10:00:00Z",
        experiment_run_id="exp_x",
    )

    loaded = ModelPackage.load(dest)

    assert loaded.base_model_name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert loaded.base_model_revision == "main"
    assert loaded.training_run_id == training_run_id
    assert loaded.experiment_run_id == "exp_x"
    assert loaded.quantization == {"enabled": True, "bits": 4}
    assert loaded.adapter_dir == dest / "adapter"


def test_load_rejects_a_directory_with_no_manifest(tmp_path):
    empty = tmp_path / "not-a-package"
    empty.mkdir()

    with pytest.raises(PackagingError, match="no model package"):
        ModelPackage.load(empty)


def test_load_rejects_a_manifest_whose_adapter_directory_is_missing(tmp_path):
    root = tmp_path / "broken-package"
    root.mkdir()
    package = ModelPackage(
        root=root, base_model_name="m", training_run_id="dpo_x", created_at="now",
    )
    import json as _json

    (root / "manifest.json").write_text(_json.dumps(package.to_dict()), encoding="utf-8")

    with pytest.raises(PackagingError, match="adapter directory"):
        ModelPackage.load(root)


def test_manifest_rejects_unknown_fields(tmp_path):
    with pytest.raises(PackagingError, match="unknown manifest field"):
        ModelPackage.from_dict(tmp_path, {"base_model_name": "m", "training_run_id": "t", "bogus": 1})


def test_manifest_requires_base_model_name_and_training_run_id(tmp_path):
    with pytest.raises(PackagingError, match="missing required field"):
        ModelPackage.from_dict(tmp_path, {})
