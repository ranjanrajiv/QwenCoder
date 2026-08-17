from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .generation.strategies import STRATEGIES, StrategyError, instruction_for
from .models.base import GenerationConfig, ModelConfig, ModelError

_REQUIRED_PATH_KEYS = (
    "raw_data",
    "problems",
    "candidates",
    "evaluations",
    "preferences",
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
_GENERATION_KEYS = frozenset({"candidates_per_problem", *_GENERATION_CONFIG_KEYS})


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
    preferences: Path
    reports: Path

    def ensure_exists(self) -> None:
        for path in (
            self.raw,
            self.problems,
            self.candidates,
            self.evaluations,
            self.preferences,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GenerationSettings:
    """The `generation:` and `generation_strategies:` sections, validated."""

    candidates_per_problem: int
    config: GenerationConfig
    strategies: tuple[str, ...]


def _parse_model(raw: Any) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml: missing required key 'model'")
    try:
        return ModelConfig.from_mapping(raw)
    except ModelError as exc:
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
    )


@dataclass(frozen=True)
class Config:
    project_name: str
    paths: Paths
    log_level: str
    project_root: Path
    model: ModelConfig
    generation: GenerationSettings

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
            preferences=resolved_paths["preferences"],
            reports=resolved_paths["reports"],
        )

        model = _parse_model(raw.get("model"))
        generation = _parse_generation(
            raw.get("generation"), raw.get("generation_strategies")
        )

        return cls(
            project_name=project_name,
            paths=paths,
            log_level=log_level,
            project_root=project_root,
            model=model,
            generation=generation,
        )
