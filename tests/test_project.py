import os
import subprocess
import sys
from pathlib import Path

import pytest

import python_dpo
from python_dpo.cli import _PLACEHOLDER_STAGES, build_parser
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
        config.paths.rankings,
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
        rankings=tmp_path / "rankings",
        preferences=tmp_path / "preferences",
        reports=tmp_path / "reports",
    )
    paths.ensure_exists()
    for path in (
        paths.raw,
        paths.problems,
        paths.candidates,
        paths.evaluations,
        paths.rankings,
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
        "rankings",
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


@pytest.mark.parametrize("command", sorted(_PLACEHOLDER_STAGES))
def test_placeholder_subcommands_parse_and_return_nonzero(command):
    parser = build_parser()
    args = parser.parse_args([command])
    assert args.func(args, Config.load()) != 0


def test_problems_is_no_longer_a_placeholder():
    assert "problems" not in _PLACEHOLDER_STAGES


@pytest.mark.parametrize("subcommand", ["build", "validate"])
def test_problems_subcommands_are_wired(subcommand):
    parser = build_parser()
    args = parser.parse_args(["problems", subcommand])
    assert args.command == "problems"
    assert args.problems_command == subcommand
    assert callable(args.func)


def test_bare_problems_prints_help_and_returns_nonzero():
    result = _run_module("problems")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_generate_is_no_longer_a_placeholder():
    assert "generate" not in _PLACEHOLDER_STAGES


def test_generate_flags_parse():
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--problem-id",
            "p001",
            "--num-candidates",
            "3",
            "--strategy",
            "normal",
            "--strategy",
            "optimized",
            "--force",
            "--dry-run",
            "--mock-model",
        ]
    )
    assert args.command == "generate"
    assert args.problem_id == "p001"
    assert args.num_candidates == 3
    assert args.strategies == ["normal", "optimized"]
    assert args.force is True
    assert args.dry_run is True
    assert args.mock_model is True
    assert callable(args.func)


def test_generate_defaults_are_unset_so_config_supplies_them():
    args = build_parser().parse_args(["generate"])
    assert args.problem_id is None
    assert args.limit is None
    assert args.num_candidates is None
    assert args.strategies is None
    assert args.force is False
    assert args.dry_run is False


def test_generate_limit_parses():
    assert build_parser().parse_args(["generate", "--limit", "2"]).limit == 2


def test_generate_resume_flag_parses():
    args = build_parser().parse_args(["generate", "--resume", "run_20260817_133700_a81f"])
    assert args.resume == "run_20260817_133700_a81f"


def test_generate_resume_defaults_to_none():
    assert build_parser().parse_args(["generate"]).resume is None


def test_generate_resume_and_dry_run_together_is_rejected():
    result = _run_module("generate", "--resume", "run_does_not_exist", "--dry-run")
    assert result.returncode == 1
    assert "--dry-run" in result.stderr and "--resume" in result.stderr


def test_generate_resume_rejects_conflicting_selection_flags():
    result = _run_module(
        "generate", "--resume", "run_does_not_exist", "--problem-id", "p001"
    )
    assert result.returncode == 1
    assert "--problem-id" in result.stderr


def test_generate_resume_reports_an_unknown_run_id():
    result = _run_module("generate", "--resume", "run_does_not_exist", "--mock-model")
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


def test_generate_rejects_an_unknown_strategy():
    result = _run_module("generate", "--strategy", "creative", "--dry-run")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr.lower()


def test_dry_run_prints_a_prompt_and_writes_nothing():
    candidates_dir = PROJECT_ROOT / "data" / "candidates"
    before = sorted(path.name for path in candidates_dir.iterdir())

    result = _run_module("generate", "--problem-id", "p001", "--dry-run")

    assert result.returncode == 0
    assert "You are an expert Python programmer." in result.stdout
    assert "def sum_even(numbers):" in result.stdout
    assert "prompt_version=v1" in result.stdout
    assert sorted(path.name for path in candidates_dir.iterdir()) == before


def test_generate_reports_an_unknown_problem_id():
    result = _run_module("generate", "--problem-id", "p999", "--dry-run")
    assert result.returncode == 1
    assert "p999" in result.stderr


def test_no_subcommand_prints_help_and_returns_nonzero():
    result = _run_module()
    assert result.returncode == 1


# --------------------------------------------------------------------------------- runs


def test_runs_list_show_validate_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["runs", "list"])
    assert args.command == "runs" and args.runs_command == "list"
    assert callable(args.func)

    args = parser.parse_args(["runs", "show", "run_20260817_133700_a81f"])
    assert args.run_id == "run_20260817_133700_a81f"
    assert callable(args.func)

    args = parser.parse_args(["runs", "validate", "run_20260817_133700_a81f", "--repair"])
    assert args.run_id == "run_20260817_133700_a81f"
    assert args.repair is True
    assert callable(args.func)


def test_bare_runs_prints_help_and_returns_nonzero():
    result = _run_module("runs")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_runs_list_exits_zero_regardless_of_run_count():
    # Whether the real data/candidates/runs/ tree is empty or already holds runs, listing
    # it must succeed — an empty store is not an error.
    result = _run_module("runs", "list")
    assert result.returncode == 0
    assert "usage" not in result.stdout.lower()


def test_runs_show_reports_an_unknown_run_id():
    result = _run_module("runs", "show", "run_does_not_exist")
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


def test_runs_validate_reports_an_unknown_run_id():
    result = _run_module("runs", "validate", "run_does_not_exist")
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


# --------------------------------------------------------------------------- candidates


def test_candidates_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(
        ["candidates", "list", "run_1", "--problem-id", "p001", "--strategy", "normal"]
    )
    assert args.run_id == "run_1"
    assert args.problem_id == "p001"
    assert args.strategy == "normal"
    assert callable(args.func)

    args = parser.parse_args(
        ["candidates", "show", "run_1", "p001_c001", "--show-code", "--show-raw"]
    )
    assert args.candidate_id == "p001_c001"
    assert args.show_code is True
    assert args.show_raw is True
    assert callable(args.func)

    args = parser.parse_args(["candidates", "stats", "run_1"])
    assert args.run_id == "run_1"
    assert callable(args.func)

    args = parser.parse_args(["candidates", "migrate", "--source", "x.jsonl", "--force"])
    assert args.source == "x.jsonl"
    assert args.force is True
    assert callable(args.func)


def test_bare_candidates_prints_help_and_returns_nonzero():
    result = _run_module("candidates")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_candidates_list_reports_an_unknown_run_id():
    result = _run_module("candidates", "list", "run_does_not_exist")
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


def test_candidates_migrate_reports_a_missing_source_file():
    result = _run_module("candidates", "migrate", "--source", "/no/such/file.jsonl")
    assert result.returncode == 1
    assert "/no/such/file.jsonl" in result.stderr


# ------------------------------------------------------------------------------ sandbox


def test_sandbox_is_not_a_placeholder():
    assert "sandbox" not in _PLACEHOLDER_STAGES


def test_sandbox_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["sandbox", "health"])
    assert args.command == "sandbox" and args.sandbox_command == "health"
    assert callable(args.func)

    args = parser.parse_args(
        ["sandbox", "run", "--file", "examples/hello.py", "--timeout", "9", "--show-stderr"]
    )
    assert args.file == "examples/hello.py"
    assert args.timeout == 9
    assert args.show_stderr is True
    assert callable(args.func)


def test_sandbox_run_defaults_are_unset_so_config_supplies_them():
    args = build_parser().parse_args(["sandbox", "run", "--file", "x.py"])
    assert args.timeout is None
    assert args.show_stderr is False


def test_sandbox_run_requires_a_file():
    result = _run_module("sandbox", "run")
    assert result.returncode == 2
    assert "--file" in result.stderr


def test_bare_sandbox_prints_help_and_returns_nonzero():
    result = _run_module("sandbox")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_sandbox_run_reports_a_missing_file_without_touching_docker():
    # Must fail on the missing path before any container work begins, so this stays a
    # fast, Docker-free check.
    result = _run_module("sandbox", "run", "--file", "/no/such/candidate.py")
    assert result.returncode == 1
    assert "/no/such/candidate.py" in result.stderr


def test_example_file_exists_for_the_documented_smoke_command():
    # The README and spec section 87 both tell users to run this exact file.
    assert (PROJECT_ROOT / "examples" / "hello.py").is_file()


# ------------------------------------------------------------------------------ evaluate


def test_evaluate_is_not_a_placeholder():
    assert "evaluate" not in _PLACEHOLDER_STAGES


def test_evaluate_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(
        ["evaluate", "candidate", "--run-id", "run_x", "--candidate-id", "p001_c001"]
    )
    assert args.command == "evaluate" and args.evaluate_command == "candidate"
    assert args.run_id == "run_x" and args.candidate_id == "p001_c001"
    assert args.force is False
    assert callable(args.func)

    args = parser.parse_args(
        ["evaluate", "run", "--run-id", "run_x", "--problem-id", "p001", "--limit", "3", "--force"]
    )
    assert args.command == "evaluate" and args.evaluate_command == "run"
    assert args.problem_id == "p001" and args.limit == 3 and args.force is True
    assert callable(args.func)


def test_evaluate_candidate_requires_run_id_and_candidate_id():
    result = _run_module("evaluate", "candidate")
    assert result.returncode == 2
    assert "--run-id" in result.stderr


def test_bare_evaluate_prints_help_and_returns_nonzero():
    result = _run_module("evaluate")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_evaluate_candidate_reports_an_unknown_run_id():
    # Must fail before any Docker work begins, so this stays a fast, Docker-free check.
    result = _run_module(
        "evaluate", "candidate", "--run-id", "run_does_not_exist", "--candidate-id", "p001_c001"
    )
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


def test_evaluate_run_reports_an_unknown_run_id():
    result = _run_module("evaluate", "run", "--run-id", "run_does_not_exist")
    assert result.returncode == 1
    assert "run_does_not_exist" in result.stderr


# ---------------------------------------------------------------------------- evaluations


def test_evaluations_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["evaluations", "list", "eval_x"])
    assert args.command == "evaluations" and args.evaluations_command == "list"
    assert args.eval_id == "eval_x"
    assert callable(args.func)

    args = parser.parse_args(["evaluations", "show", "eval_x", "p001_c001"])
    assert args.eval_id == "eval_x" and args.candidate_id == "p001_c001"
    assert callable(args.func)

    args = parser.parse_args(["evaluations", "stats", "eval_x"])
    assert args.eval_id == "eval_x"
    assert callable(args.func)


def test_bare_evaluations_prints_help_and_returns_nonzero():
    result = _run_module("evaluations")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_evaluations_list_reports_an_unknown_eval_id():
    result = _run_module("evaluations", "list", "eval_does_not_exist")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


def test_evaluations_show_reports_an_unknown_eval_id():
    result = _run_module("evaluations", "show", "eval_does_not_exist", "p001_c001")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


def test_evaluations_stats_reports_an_unknown_eval_id():
    result = _run_module("evaluations", "stats", "eval_does_not_exist")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


def test_evaluations_list_with_no_argument_lists_evaluation_runs():
    result = _run_module("evaluations", "list")
    assert result.returncode == 0


# ----------------------------------------------------------------------------------- rank


def test_rank_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["rank", "run", "--evaluation-run-id", "eval_x"])
    assert args.command == "rank" and args.rank_command == "run"
    assert args.evaluation_run_id == "eval_x"
    assert args.problem_id is None and args.limit is None
    assert args.resume is None and args.force is False
    assert callable(args.func)

    args = parser.parse_args(
        [
            "rank", "run", "--evaluation-run-id", "eval_x",
            "--problem-id", "p001", "--limit", "3", "--resume", "rank_x", "--force",
        ]
    )
    assert args.problem_id == "p001" and args.limit == 3
    assert args.resume == "rank_x" and args.force is True


def test_rank_run_requires_evaluation_run_id():
    result = _run_module("rank", "run")
    assert result.returncode == 2
    assert "--evaluation-run-id" in result.stderr


def test_bare_rank_prints_help_and_returns_nonzero():
    result = _run_module("rank")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_rank_run_reports_an_unknown_evaluation_run_id():
    result = _run_module("rank", "run", "--evaluation-run-id", "eval_does_not_exist")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


# ------------------------------------------------------------------------------- rankings


def test_rankings_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["rankings", "list", "rank_x"])
    assert args.command == "rankings" and args.rankings_command == "list"
    assert args.ranking_run_id == "rank_x"
    assert callable(args.func)

    args = parser.parse_args(["rankings", "show", "rank_x", "p001"])
    assert args.ranking_run_id == "rank_x" and args.problem_id == "p001"
    assert callable(args.func)

    args = parser.parse_args(["rankings", "stats", "rank_x"])
    assert args.ranking_run_id == "rank_x"
    assert callable(args.func)

    args = parser.parse_args(["rankings", "validate", "rank_x"])
    assert args.ranking_run_id == "rank_x"
    assert callable(args.func)


def test_bare_rankings_prints_help_and_returns_nonzero():
    result = _run_module("rankings")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_rankings_list_reports_an_unknown_ranking_run_id():
    result = _run_module("rankings", "list", "rank_does_not_exist")
    assert result.returncode == 1
    assert "rank_does_not_exist" in result.stderr


def test_rankings_show_reports_an_unknown_ranking_run_id():
    result = _run_module("rankings", "show", "rank_does_not_exist", "p001")
    assert result.returncode == 1
    assert "rank_does_not_exist" in result.stderr


def test_rankings_stats_reports_an_unknown_ranking_run_id():
    result = _run_module("rankings", "stats", "rank_does_not_exist")
    assert result.returncode == 1
    assert "rank_does_not_exist" in result.stderr


def test_rankings_validate_reports_an_unknown_ranking_run_id():
    result = _run_module("rankings", "validate", "rank_does_not_exist")
    assert result.returncode == 1
    assert "rank_does_not_exist" in result.stderr
