from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .evaluation.config import EvaluationConfig
from .evaluation.errors import EvaluationConfigError
from .generation.strategies import STRATEGIES, StrategyError, instruction_for
from .models.base import GenerationConfig, ModelConfig, ModelError
from .preferences.config import PreferenceConfig
from .preferences.errors import PreferenceConfigError
from .sandbox.config import SandboxConfig
from .sandbox.errors import SandboxConfigError

_REQUIRED_PATH_KEYS = (
    "raw_data",
    "problems",
    "candidates",
    "evaluations",
    "rankings",
    "preferences",
    "training",
    "model_evaluations",
    "reports",
)

# Keys accepted under `generation:`. The decoding parameters build a GenerationConfig;
# candidates_per_problem is a pipeline setting rather than a decoding one.
_GENERATION_CONFIG_KEYS = (
    "temperature",
    "top_p",
    "max_new_tokens",
    "do_sample",
    "repetition_penalty",
    "seed",
)
_GENERATION_KEYS = frozenset({"candidates_per_problem", "retry", *_GENERATION_CONFIG_KEYS})

DEFAULT_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class RetrySettings:
    """Retry policy for infrastructure failures (spec 04 sections 28, 29).

    ``max_attempts=1`` means "no retry" — the first failure is terminal, matching
    candidate failures (empty output, extraction failure), which are never retried
    regardless of this setting.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ConfigError(
                "config.yaml: generation.retry.max_attempts must be an integer of 1 or greater"
            )

    def to_dict(self) -> dict[str, int]:
        return {"max_attempts": self.max_attempts}

    @classmethod
    def from_mapping(cls, data: Any) -> RetrySettings:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigError("config.yaml: generation.retry must be a mapping")
        unknown = sorted(set(data) - {"max_attempts"})
        if unknown:
            raise ConfigError(
                f"config.yaml: generation.retry: unknown key(s): {', '.join(unknown)}"
            )
        return cls(max_attempts=data.get("max_attempts", DEFAULT_MAX_ATTEMPTS))


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed, or incomplete."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: CWD) looking for pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw: Path
    problems: Path
    candidates: Path
    evaluations: Path
    rankings: Path
    preferences: Path
    training: Path
    model_evaluations: Path
    reports: Path

    def ensure_exists(self) -> None:
        for path in (
            self.raw,
            self.problems,
            self.candidates,
            self.evaluations,
            self.rankings,
            self.preferences,
            self.training,
            self.model_evaluations,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GenerationSettings:
    """The `generation:` and `generation_strategies:` sections, validated."""

    candidates_per_problem: int
    config: GenerationConfig
    strategies: tuple[str, ...]
    retry: RetrySettings


def _parse_model(raw: Any) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml: missing required key 'model'")
    try:
        return ModelConfig.from_mapping(raw)
    except ModelError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc


def _parse_sandbox(raw: Any) -> SandboxConfig:
    """Parse the optional `sandbox:` section, defaulting when it is absent.

    `sandbox/` raises its own SandboxConfigError rather than ConfigError, so that package
    never has to import this one; translating it here keeps the dependency one-way, exactly
    as with ModelError above.
    """
    try:
        return SandboxConfig.from_mapping(raw)
    except SandboxConfigError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc


def _parse_evaluation(raw: Any) -> EvaluationConfig:
    """Parse the optional `evaluation:` section, defaulting when it is absent.

    `evaluation/` raises its own EvaluationConfigError rather than ConfigError, so that
    package never has to import this one; translating it here keeps the dependency
    one-way, exactly as with SandboxConfigError above.
    """
    try:
        return EvaluationConfig.from_mapping(raw)
    except EvaluationConfigError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc


def _parse_preferences(raw: Any) -> PreferenceConfig:
    """Parse the optional `preferences:` section, defaulting when it is absent.

    `preferences/` raises its own PreferenceConfigError rather than ConfigError, so that
    package never has to import this one; translating it here keeps the dependency
    one-way, exactly as with EvaluationConfigError above.
    """
    try:
        return PreferenceConfig.from_mapping(raw)
    except PreferenceConfigError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc


def _parse_generation(raw: Any, strategies_raw: Any) -> GenerationSettings:
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml: missing required key 'generation'")

    unknown = sorted(set(raw) - _GENERATION_KEYS)
    if unknown:
        raise ConfigError(f"config.yaml: generation: unknown key(s): {', '.join(unknown)}")

    candidates_per_problem = raw.get("candidates_per_problem", 5)
    if (
        isinstance(candidates_per_problem, bool)
        or not isinstance(candidates_per_problem, int)
        or candidates_per_problem < 1
    ):
        raise ConfigError(
            "config.yaml: generation.candidates_per_problem must be an integer of 1 or greater"
        )

    try:
        config = GenerationConfig(
            **{key: raw[key] for key in _GENERATION_CONFIG_KEYS if key in raw}
        )
    except ModelError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc

    retry = RetrySettings.from_mapping(raw.get("retry"))

    if strategies_raw is None:
        strategies = STRATEGIES
    else:
        if not isinstance(strategies_raw, list) or not strategies_raw:
            raise ConfigError(
                "config.yaml: generation_strategies must be a non-empty list of strategy names"
            )
        seen: set[str] = set()
        for name in strategies_raw:
            if not isinstance(name, str):
                raise ConfigError(
                    "config.yaml: generation_strategies entries must be strings"
                )
            try:
                instruction_for(name)
            except StrategyError as exc:
                raise ConfigError(f"config.yaml: generation_strategies: {exc}") from exc
            if name in seen:
                raise ConfigError(
                    f"config.yaml: generation_strategies: duplicate entry {name!r}"
                )
            seen.add(name)
        strategies = tuple(strategies_raw)

    return GenerationSettings(
        candidates_per_problem=candidates_per_problem,
        config=config,
        strategies=strategies,
        retry=retry,
    )


@dataclass(frozen=True)
class Config:
    project_name: str
    paths: Paths
    log_level: str
    project_root: Path
    model: ModelConfig
    generation: GenerationSettings
    sandbox: SandboxConfig
    evaluation: EvaluationConfig
    preferences: PreferenceConfig

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        project_root = find_project_root()
        config_path = path if path is not None else project_root / "config.yaml"

        if not config_path.is_file():
            raise ConfigError(f"config.yaml not found at {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            try:
                raw: Any = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ConfigError(f"config.yaml: invalid YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError("config.yaml: root must be a mapping")

        project = raw.get("project")
        if not isinstance(project, dict) or not isinstance(project.get("name"), str):
            raise ConfigError("config.yaml: missing required key 'project.name'")
        project_name = project["name"]

        paths_section = raw.get("paths")
        if not isinstance(paths_section, dict):
            raise ConfigError("config.yaml: missing required key 'paths'")

        resolved_paths: dict[str, Path] = {}
        for key in _REQUIRED_PATH_KEYS:
            value = paths_section.get(key)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"config.yaml: missing required key 'paths.{key}'")
            resolved_paths[key] = (project_root / value).resolve()

        logging_section = raw.get("logging")
        if not isinstance(logging_section, dict) or not isinstance(
            logging_section.get("level"), str
        ):
            raise ConfigError("config.yaml: missing required key 'logging.level'")
        log_level = logging_section["level"]

        paths = Paths(
            raw=resolved_paths["raw_data"],
            problems=resolved_paths["problems"],
            candidates=resolved_paths["candidates"],
            evaluations=resolved_paths["evaluations"],
            rankings=resolved_paths["rankings"],
            preferences=resolved_paths["preferences"],
            training=resolved_paths["training"],
            model_evaluations=resolved_paths["model_evaluations"],
            reports=resolved_paths["reports"],
        )

        model = _parse_model(raw.get("model"))
        generation = _parse_generation(
            raw.get("generation"), raw.get("generation_strategies")
        )
        sandbox = _parse_sandbox(raw.get("sandbox"))
        evaluation = _parse_evaluation(raw.get("evaluation"))
        preferences = _parse_preferences(raw.get("preferences"))

        return cls(
            project_name=project_name,
            paths=paths,
            log_level=log_level,
            project_root=project_root,
            model=model,
            generation=generation,
            sandbox=sandbox,
            evaluation=evaluation,
            preferences=preferences,
        )
