import os
import subprocess
import sys
from pathlib import Path

import pytest

import python_dpo
from python_dpo.cli import _STAGE_NAMES, build_parser
from python_dpo.config import Config, ConfigError, Paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_import_succeeds():
    assert python_dpo is not None


def test_version_is_a_nonempty_dotted_string():
    version = python_dpo.__version__
    assert isinstance(version, str)
    assert version
    parts = version.split(".")
    assert len(parts) >= 2
    for part in parts:
        assert part.isdigit()


def test_config_loads_real_config_yaml():
    config = Config.load()
    assert config.project_name == "python-dpo"
    assert config.project_root == PROJECT_ROOT
    for path in (
        config.paths.raw,
        config.paths.problems,
        config.paths.candidates,
        config.paths.evaluations,
        config.paths.preferences,
        config.paths.reports,
    ):
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents or path == PROJECT_ROOT


def test_config_load_raises_on_malformed_yaml(tmp_path):
    bad_config = tmp_path / "config.yaml"
    bad_config.write_text("project:\n  name: incomplete\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="paths"):
        Config.load(path=bad_config)


def test_config_load_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="not found"):
        Config.load(path=missing)


def test_paths_ensure_exists_creates_all_directories(tmp_path):
    paths = Paths(
        raw=tmp_path / "raw",
        problems=tmp_path / "problems",
        candidates=tmp_path / "candidates",
        evaluations=tmp_path / "evaluations",
        preferences=tmp_path / "preferences",
        reports=tmp_path / "reports",
    )
    paths.ensure_exists()
    for path in (
        paths.raw,
        paths.problems,
        paths.candidates,
        paths.evaluations,
        paths.preferences,
        paths.reports,
    ):
        assert path.is_dir()


def test_real_data_directories_exist():
    for name in (
        "raw",
        "problems",
        "candidates",
        "evaluations",
        "preferences",
        "reports",
    ):
        assert (PROJECT_ROOT / "data" / name).is_dir()


def _run_module(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "python_dpo", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_cli_help_exits_zero():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_cli_version_exits_zero_and_prints_version():
    result = _run_module("--version")
    assert result.returncode == 0
    assert python_dpo.__version__ in result.stdout


@pytest.mark.parametrize("command", sorted(_STAGE_NAMES))
def test_placeholder_subcommands_parse_and_return_nonzero(command):
    parser = build_parser()
    args = parser.parse_args([command])
    assert args.func(args) != 0


def test_no_subcommand_prints_help_and_returns_nonzero():
    result = _run_module()
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()
