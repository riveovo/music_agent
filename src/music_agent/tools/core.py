"""External tool registry used by the ReAct agent."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ..errors import MusicAgentError
from ..paths import PROJECT_ROOT


TOOLS_PATH_ENV = "MUSIC_AGENT_TOOLS_PATH"
SUPPORTED_TOOL_PROVIDERS = ("python", "mcp", "http", "cli")

ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolProvider:
    """Declared provider for an external tool."""

    name: str
    config: Mapping[str, Any]

    def validate(self) -> None:
        if self.name not in SUPPORTED_TOOL_PROVIDERS:
            available = ", ".join(SUPPORTED_TOOL_PROVIDERS)
            raise MusicAgentError(f"Unknown tool provider '{self.name}'. Available providers: {available}.")


@dataclass(frozen=True)
class ToolSpec:
    """Executable tool metadata and handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    source: str = "builtin"
    timeout_seconds: float | None = None
    danger_level: str = "low"
    provider: str = "python"

    @property
    def openai_name(self) -> str:
        """OpenAI function names cannot rely on dotted names, so expose a stable alias."""
        alias = re.sub(r"[^A-Za-z0-9_-]", "_", self.name)
        return alias[:64]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.openai_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function_name": self.openai_name,
            "description": self.description,
            "source": self.source,
            "provider": self.provider,
            "danger_level": self.danger_level,
        }


class ToolRegistry:
    """Registry for executable external tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise MusicAgentError(f"Duplicate tool name: {spec.name}")
        if spec.openai_name in self._aliases:
            other = self._aliases[spec.openai_name]
            raise MusicAgentError(f"Tool OpenAI alias collision: {spec.name} and {other} both map to {spec.openai_name}")
        self._tools[spec.name] = spec
        self._aliases[spec.openai_name] = spec.name

    def get(self, name_or_alias: str) -> ToolSpec:
        name = self._aliases.get(name_or_alias, name_or_alias)
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MusicAgentError(f"Unknown external tool: {name_or_alias}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def summaries(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        selected = self.names() if names is None else list(names)
        return [self.get(name).summary() for name in selected]

    def to_openai_tools(self, names: Iterable[str]) -> list[dict[str, Any]]:
        return [self.get(name).to_openai_tool() for name in names]

    def call(self, name_or_alias: str, arguments: Mapping[str, Any]) -> Any:
        spec = self.get(name_or_alias)
        return spec.handler(dict(arguments))


def build_tool_registry(
    runtime_options: Mapping[str, Any] | None = None,
    *,
    tools_path: str | Path | Iterable[str | Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ToolRegistry:
    """Build a registry with built-in music tools plus configured external tools."""
    from .music import register_music_tools

    registry = ToolRegistry()
    register_music_tools(registry, runtime_options or {}, progress=progress)
    load_external_tool_configs(registry, tools_path=tools_path)
    return registry


def load_external_tool_configs(
    registry: ToolRegistry,
    *,
    tools_path: str | Path | Iterable[str | Path] | None = None,
) -> None:
    for config_path in _discover_tool_config_files(tools_path):
        spec = _load_tool_config(config_path)
        registry.register(spec)


def _discover_tool_config_files(tools_path: str | Path | Iterable[str | Path] | None) -> list[Path]:
    directories = _configured_paths(tools_path, env_var=TOOLS_PATH_ENV, default=PROJECT_ROOT / ".agents" / "tools")
    files: list[Path] = []
    for directory in directories:
        if directory.exists():
            if not directory.is_dir():
                raise MusicAgentError(f"Tool path is not a directory: {directory}")
            files.extend(sorted(directory.glob("*.json")))
    return files


def _load_tool_config(path: Path) -> ToolSpec:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MusicAgentError(f"Could not parse tool config JSON: {path}") from exc

    name = _require_string(payload, "name", path)
    description = _require_string(payload, "description", path)
    provider_name = _require_string(payload, "provider", path)
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise MusicAgentError(f"Tool config '{path}' must include an object 'parameters' JSON Schema.")

    provider = ToolProvider(provider_name, payload)
    provider.validate()
    handler = _handler_for_provider(provider, path)
    return ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
        source=str(path),
        danger_level=str(payload.get("danger_level", "medium")),
        provider=provider_name,
        timeout_seconds=_optional_float(payload.get("timeout_seconds")),
    )


def _handler_for_provider(provider: ToolProvider, path: Path) -> ToolHandler:
    if provider.name != "python":
        def unsupported(_: dict[str, Any]) -> Any:
            raise MusicAgentError(
                f"Tool provider '{provider.name}' is declared in {path}, but this v1 registry only executes python providers."
            )

        return unsupported

    module_name = _first_string(provider.config, ("module", "python_module"))
    callable_name = _first_string(provider.config, ("callable", "python_callable", "function"))
    python_config = provider.config.get("python")
    if isinstance(python_config, dict):
        module_name = module_name or _first_string(python_config, ("module",))
        callable_name = callable_name or _first_string(python_config, ("callable", "function"))
    if not module_name or not callable_name:
        raise MusicAgentError(f"Python tool config '{path}' must include module and callable.")

    def call_python(arguments: dict[str, Any]) -> Any:
        module = importlib.import_module(module_name)
        function = getattr(module, callable_name)
        return function(**arguments)

    return call_python


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
        raise MusicAgentError(f"Tool config '{path}' must include a non-empty string '{key}'.")
    return value.strip()


def _first_string(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MusicAgentError(f"Invalid timeout_seconds value: {value!r}") from exc
