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
        config.paths.training,
        config.paths.model_evaluations,
        config.paths.experiments,
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
        training=tmp_path / "training",
        model_evaluations=tmp_path / "model_evaluations",
        experiments=tmp_path / "experiments",
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
        paths.training,
        paths.model_evaluations,
        paths.experiments,
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
        "training",
        "model_evaluations",
        "experiments",
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


def test_no_placeholder_stages_remain():
    # Stage 12 was the last stage with a placeholder ("run", a full pipeline run, now
    # implemented as `experiment run`); nothing should ever populate this dict again.
    assert _PLACEHOLDER_STAGES == {}


def test_problems_is_no_longer_a_placeholder():
    assert "problems" not in _PLACEHOLDER_STAGES


def test_run_is_no_longer_a_placeholder():
    assert "run" not in _PLACEHOLDER_STAGES


@pytest.mark.parametrize(
    "args",
    [
        ["experiment", "preflight"],
        ["experiment", "graph"],
        ["experiment", "run"],
        ["experiment", "resume", "--experiment-run-id", "exp_x"],
        ["experiment", "retry", "--experiment-run-id", "exp_x", "--stage", "dpo_training"],
        ["experiment", "status", "--experiment-run-id", "exp_x"],
        ["experiment", "list"],
    ],
)
def test_experiment_subcommands_are_wired(args):
    parser = build_parser()
    parsed = parser.parse_args(args)
    assert callable(parsed.func)


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


# ---------------------------------------------------------------------------- preferences


def test_preferences_is_not_a_placeholder():
    assert "preferences" not in _PLACEHOLDER_STAGES


def test_preferences_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["preferences", "generate", "--ranking-run-id", "rank_x"])
    assert args.command == "preferences" and args.preferences_command == "generate"
    assert args.ranking_run_id == "rank_x"
    assert args.policy is None and args.margin is None
    assert args.max_pairs_per_problem is None and args.split_seed is None
    assert args.resume is None and args.force is False
    assert callable(args.func)

    args = parser.parse_args(
        [
            "preferences", "generate", "--ranking-run-id", "rank_x",
            "--policy", "margin", "--margin", "0.3", "--max-pairs-per-problem", "5",
            "--split-seed", "7", "--resume", "pref_x", "--force",
        ]
    )
    assert args.policy == "margin" and args.margin == 0.3
    assert args.max_pairs_per_problem == 5 and args.split_seed == 7
    assert args.resume == "pref_x" and args.force is True

    args = parser.parse_args(["preferences", "list"])
    assert args.preferences_command == "list"
    assert callable(args.func)

    args = parser.parse_args(
        ["preferences", "show", "--preference-run-id", "pref_x", "--preference-id", "pref_y"]
    )
    assert args.preference_run_id == "pref_x" and args.preference_id == "pref_y"
    assert args.show_code is False
    assert callable(args.func)

    args = parser.parse_args(["preferences", "stats", "--preference-run-id", "pref_x"])
    assert args.preference_run_id == "pref_x"
    assert callable(args.func)

    args = parser.parse_args(["preferences", "validate", "--preference-run-id", "pref_x"])
    assert args.preference_run_id == "pref_x"
    assert callable(args.func)


def test_bare_preferences_prints_help_and_returns_nonzero():
    result = _run_module("preferences")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_preferences_generate_requires_ranking_run_id():
    result = _run_module("preferences", "generate")
    assert result.returncode == 2
    assert "--ranking-run-id" in result.stderr


def test_preferences_generate_reports_an_unknown_ranking_run_id():
    result = _run_module("preferences", "generate", "--ranking-run-id", "rank_does_not_exist")
    assert result.returncode == 1
    assert "rank_does_not_exist" in result.stderr


def test_preferences_show_reports_an_unknown_preference_run_id():
    result = _run_module(
        "preferences", "show", "--preference-run-id", "pref_does_not_exist", "--preference-id", "x"
    )
    assert result.returncode == 1
    assert "pref_does_not_exist" in result.stderr


def test_preferences_stats_reports_an_unknown_preference_run_id():
    result = _run_module("preferences", "stats", "--preference-run-id", "pref_does_not_exist")
    assert result.returncode == 1
    assert "pref_does_not_exist" in result.stderr


def test_preferences_validate_reports_an_unknown_preference_run_id():
    result = _run_module("preferences", "validate", "--preference-run-id", "pref_does_not_exist")
    assert result.returncode == 1
    assert "pref_does_not_exist" in result.stderr


# ------------------------------------------------------------------------------- train


def test_train_is_not_a_placeholder():
    assert "train" not in _PLACEHOLDER_STAGES


def test_train_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["train", "hardware-check"])
    assert args.command == "train" and args.train_command == "hardware-check"
    assert callable(args.func)

    args = parser.parse_args(["train", "dpo", "--preference-run-id", "pref_x"])
    assert args.preference_run_id == "pref_x"
    assert args.config is None
    assert args.dry_run is False and args.smoke_test is False
    assert args.allow_small_dataset is False
    assert args.resume_from_checkpoint is None and args.force_resume is False

    args = parser.parse_args(
        [
            "train", "dpo", "--config", "configs/training/dpo_qlora.yaml",
            "--preference-run-id", "pref_x", "--dry-run", "--smoke-test",
            "--allow-small-dataset", "--override-truncation",
            "--resume-from-checkpoint", "dpo_x", "--force-resume",
            "--experiment-name", "exp", "--learning-rate", "2e-5", "--beta", "0.5",
            "--epochs", "2", "--max-steps", "10", "--seed", "7", "--lora-r", "8",
        ]
    )
    assert args.dry_run is True and args.smoke_test is True
    assert args.allow_small_dataset is True and args.override_truncation is True
    assert args.resume_from_checkpoint == "dpo_x" and args.force_resume is True
    assert args.experiment_name == "exp"
    assert args.learning_rate == 2e-5 and args.beta == 0.5
    assert args.epochs == 2 and args.max_steps == 10
    assert args.seed == 7 and args.lora_r == 8

    args = parser.parse_args(["train", "verify", "--training-run-id", "dpo_x"])
    assert args.training_run_id == "dpo_x"

    args = parser.parse_args(
        ["train", "inference", "--training-run-id", "dpo_x", "--prompt", "hi"]
    )
    assert args.prompt == "hi" and args.max_new_tokens == 256

    assert callable(parser.parse_args(["train", "list"]).func)
    assert parser.parse_args(["train", "show", "--training-run-id", "dpo_x"]).training_run_id


def test_bare_train_prints_help_and_returns_nonzero():
    result = _run_module("train")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_train_verify_requires_a_training_run_id():
    result = _run_module("train", "verify")
    assert result.returncode == 2
    assert "--training-run-id" in result.stderr


def test_train_verify_reports_an_unknown_training_run_id():
    result = _run_module("train", "verify", "--training-run-id", "dpo_does_not_exist")
    assert result.returncode == 1
    assert "dpo_does_not_exist" in result.stderr


def test_train_show_reports_an_unknown_training_run_id():
    result = _run_module("train", "show", "--training-run-id", "dpo_does_not_exist")
    assert result.returncode == 1
    assert "dpo_does_not_exist" in result.stderr


def test_train_dpo_reports_an_unknown_preference_run_id():
    result = _run_module(
        "train", "dpo", "--preference-run-id", "pref_does_not_exist", "--dry-run"
    )
    assert result.returncode == 1
    assert "pref_does_not_exist" in result.stderr


def test_preferences_generate_accepts_split_ratios():
    args = build_parser().parse_args(
        ["preferences", "generate", "--ranking-run-id", "rank_x",
         "--split-ratios", "0.5,0.25,0.25"]
    )
    assert args.split_ratios == "0.5,0.25,0.25"


def test_split_ratios_default_to_none_so_config_supplies_them():
    args = build_parser().parse_args(
        ["preferences", "generate", "--ranking-run-id", "rank_x"]
    )
    assert args.split_ratios is None


def test_bad_split_ratios_are_rejected():
    # A real ranking run id, so the invocation gets past the upstream-run lookup and
    # actually reaches the ratio validation. It still writes nothing: the ratios are
    # parsed before any preference run is created.
    result = _run_module(
        "preferences", "generate",
        "--ranking-run-id", "rank_20260817_161726_a84d",
        "--split-ratios", "0.5,0.5,0.5",
    )
    assert result.returncode == 1
    assert "sum to 1.0" in result.stderr


# ---------------------------------------------------------------------------- benchmark


def test_benchmark_is_not_a_placeholder():
    assert "benchmark" not in _PLACEHOLDER_STAGES


def test_benchmark_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["benchmark", "build", "--name", "python_eval_v1"])
    assert args.name == "python_eval_v1"
    assert args.problem_id is None
    assert args.exclude_preference_run_id is None

    args = parser.parse_args(
        [
            "benchmark", "build", "--name", "python_eval_v1",
            "--problem-id", "p001", "--problem-id", "p002",
        ]
    )
    assert args.problem_id == ["p001", "p002"]

    args = parser.parse_args(["benchmark", "validate", "--benchmark", "python_eval_v1"])
    assert args.benchmark == "python_eval_v1"

    args = parser.parse_args(
        [
            "benchmark", "check-leakage",
            "--benchmark", "python_eval_v1", "--preference-run-id", "pref_x",
        ]
    )
    assert args.preference_run_id == "pref_x"


def test_bare_benchmark_prints_help_and_returns_nonzero():
    result = _run_module("benchmark")
    assert result.returncode == 1
    assert "usage" in result.stdout.lower()


def test_benchmark_validate_reports_unknown_benchmark():
    result = _run_module("benchmark", "validate", "--benchmark", "does_not_exist_v1")
    assert result.returncode == 1
    assert "does_not_exist_v1" in result.stderr


def test_benchmark_check_leakage_reports_unknown_preference_run():
    result = _run_module(
        "benchmark", "check-leakage",
        "--benchmark", "python_eval_v1", "--preference-run-id", "pref_does_not_exist",
    )
    assert result.returncode == 1
    assert "pref_does_not_exist" in result.stderr


# ------------------------------------------------------------------------ evaluate-model


def test_evaluate_model_is_not_a_placeholder():
    assert "evaluate-model" not in _PLACEHOLDER_STAGES


def test_evaluate_model_bare_form_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["evaluate-model", "--benchmark", "python_eval_v1", "--training-run-id", "dpo_x"]
    )
    assert args.benchmark == "python_eval_v1"
    assert args.training_run_id == "dpo_x"
    assert args.model == "both"
    assert args.num_samples is None
    assert args.limit is None
    assert args.smoke_test is False
    assert callable(args.func)


def test_evaluate_model_subcommands_parse():
    parser = build_parser()

    args = parser.parse_args(["evaluate-model", "validate", "--evaluation-run-id", "eval_x"])
    assert args.evaluation_run_id == "eval_x"

    args = parser.parse_args(["evaluate-model", "report", "--evaluation-run-id", "eval_x"])
    assert args.evaluation_run_id == "eval_x"

    args = parser.parse_args(["evaluate-model", "stats", "--evaluation-run-id", "eval_x"])
    assert args.evaluation_run_id == "eval_x"

    args = parser.parse_args(["evaluate-model", "compare", "--runs", "eval_a,eval_b"])
    assert args.runs == "eval_a,eval_b"

    assert callable(parser.parse_args(["evaluate-model", "list"]).func)


def test_evaluate_model_bare_requires_training_run_id():
    result = _run_module("evaluate-model", "--benchmark", "python_eval_v1")
    assert result.returncode == 1
    assert "--training-run-id" in result.stderr


def test_evaluate_model_validate_reports_unknown_run():
    result = _run_module("evaluate-model", "validate", "--evaluation-run-id", "eval_does_not_exist")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


def test_evaluate_model_report_reports_unknown_run():
    result = _run_module("evaluate-model", "report", "--evaluation-run-id", "eval_does_not_exist")
    assert result.returncode == 1
    assert "eval_does_not_exist" in result.stderr


def test_evaluate_model_run_reports_unknown_training_run_id():
    result = _run_module(
        "evaluate-model",
        "--benchmark", "python_eval_v1",
        "--training-run-id", "dpo_does_not_exist",
        "--smoke-test",
    )
    assert result.returncode == 1
    assert "dpo_does_not_exist" in result.stderr
