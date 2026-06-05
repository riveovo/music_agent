"""Skill discovery and parsing for standard SKILL.md bundles."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from ..errors import MusicAgentError
from ..paths import PROJECT_ROOT
from ..tools import ToolRegistry


SKILLS_PATH_ENV = "MUSIC_AGENT_SKILLS_PATH"


@dataclass(frozen=True)
class Skill:
    """A standard SKILL.md bundle."""

    name: str
    description: str
    allowed_tools: tuple[str, ...]
    required_inputs: tuple[str, ...]
    body: str
    path: Path
    source: str
    frontmatter: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_tools": list(self.allowed_tools),
            "required_inputs": list(self.required_inputs),
            "source": self.source,
            "path": str(self.path),
        }

    def loaded_document(self, tool_registry: ToolRegistry) -> dict[str, Any]:
        return {
            **self.summary(),
            "instructions": self.body.strip(),
            "available_tools": tool_registry.summaries(self.allowed_tools),
        }


class SkillRegistry:
    """Registry for discovered skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            existing = self._skills[skill.name]
            raise MusicAgentError(f"Duplicate skill name '{skill.name}' in {skill.path} and {existing.path}.")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise MusicAgentError(f"Unknown skill '{name}'. Available skills: {available}.") from exc

    def names(self) -> list[str]:
        return sorted(self._skills)

    def summaries(self) -> list[dict[str, Any]]:
        return [self._skills[name].summary() for name in self.names()]

    def validate_allowed_tools(self, tool_registry: ToolRegistry) -> None:
        missing: list[str] = []
        for skill in self._skills.values():
            for tool_name in skill.allowed_tools:
                if not tool_registry.has(tool_name):
                    missing.append(f"{skill.name}: {tool_name}")
        if missing:
            joined = ", ".join(missing)
            raise MusicAgentError(f"Skill(s) reference missing external tool(s): {joined}")


def build_skill_registry(
    tool_registry: ToolRegistry,
    *,
    skills_path: str | Path | Iterable[str | Path] | None = None,
) -> SkillRegistry:
    registry = SkillRegistry()
    for skill_file in _builtin_skill_files():
        registry.register(parse_skill_file(skill_file, source="builtin"))
    for skill_file in _discover_external_skill_files(skills_path):
        registry.register(parse_skill_file(skill_file, source="external"))
    registry.validate_allowed_tools(tool_registry)
    return registry


def parse_skill_file(path: str | Path, *, source: str = "external") -> Skill:
    skill_path = Path(path).expanduser()
    text = skill_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text, skill_path)
    name = _require_string(frontmatter, "name", skill_path)
    description = _require_string(frontmatter, "description", skill_path)
    allowed_tools = tuple(_as_string_list(frontmatter.get("allowed_tools"), "allowed_tools", skill_path))
    required_inputs = tuple(_as_string_list(frontmatter.get("required_inputs", []), "required_inputs", skill_path))
    return Skill(
        name=name,
        description=description,
        allowed_tools=allowed_tools,
        required_inputs=required_inputs,
        body=body,
        path=skill_path,
        source=source,
        frontmatter=frontmatter,
    )


def _builtin_skill_files() -> list[Path]:
    builtin_root = Path(__file__).with_name("builtin")
    return sorted(builtin_root.glob("*/SKILL.md"))


def _discover_external_skill_files(skills_path: str | Path | Iterable[str | Path] | None) -> list[Path]:
    files: list[Path] = []
    for directory in _configured_paths(skills_path, env_var=SKILLS_PATH_ENV, default=PROJECT_ROOT / ".agents" / "skills"):
        if directory.exists():
            if not directory.is_dir():
                raise MusicAgentError(f"Skill path is not a directory: {directory}")
            files.extend(sorted(directory.glob("*/SKILL.md")))
    return files


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise MusicAgentError(f"Skill file must start with YAML frontmatter: {path}")
    try:
        end_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise MusicAgentError(f"Skill file has unterminated YAML frontmatter: {path}") from exc
    frontmatter = _parse_simple_yaml(lines[1:end_index], path)
    body = "\n".join(lines[end_index + 1 :]).strip() + "\n"
    return frontmatter, body


def _parse_simple_yaml(lines: list[str], path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  - ") or raw_line.startswith("- "):
            if current_list_key is None:
                raise MusicAgentError(f"Unexpected list item in skill frontmatter: {path}")
            result.setdefault(current_list_key, []).append(_strip_quotes(raw_line.split("-", 1)[1].strip()))
            continue
        current_list_key = None
        if ":" not in raw_line:
            raise MusicAgentError(f"Invalid skill frontmatter line in {path}: {raw_line}")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not value:
            result[key] = []
            current_list_key = key
        elif value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise MusicAgentError(f"Invalid list value for '{key}' in {path}.") from exc
            result[key] = parsed
        else:
            result[key] = _strip_quotes(value)
    return result


def _configured_paths(
    explicit: str | Path | Iterable[str | Path] | None,
    *,
    env_var: str,
    default: Path,
) -> list[Path]:
    raw: list[str | Path] = []
    if explicit is not None:
        if isinstance(explicit, (str, Path)):
            raw.extend(str(explicit).split(os.pathsep))
        else:
            raw.extend(explicit)
    env_value = os.getenv(env_var)
    if env_value:
        raw.extend(env_value.split(os.pathsep))
    if not raw:
        raw.append(default)
    return [Path(item).expanduser() for item in raw if str(item)]


def _require_string(payload: Mapping[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MusicAgentError(f"Skill file '{path}' must include a non-empty '{key}' in frontmatter.")
    return value.strip()


def _as_string_list(value: Any, key: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MusicAgentError(f"Skill file '{path}' frontmatter '{key}' must be a list.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MusicAgentError(f"Skill file '{path}' frontmatter '{key}' must contain only strings.")
        result.append(item.strip())
    return result


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
