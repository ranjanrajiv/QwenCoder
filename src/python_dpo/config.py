from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_PATH_KEYS = (
    "raw_data",
    "problems",
    "candidates",
    "evaluations",
    "preferences",
    "reports",
)


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
class Config:
    project_name: str
    paths: Paths
    log_level: str
    project_root: Path

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

        return cls(
            project_name=project_name,
            paths=paths,
            log_level=log_level,
            project_root=project_root,
        )
