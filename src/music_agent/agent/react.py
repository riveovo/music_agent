"""OpenAI Responses API ReAct agent runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
from typing import Any

from ..errors import MusicAgentError
from ..skills import Skill, build_skill_registry
from ..tools import ToolRegistry, build_tool_registry


OPENAI_MODEL_ENV = "MUSIC_AGENT_OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class OpenAIReActUnavailable(MusicAgentError):
    """Raised when the OpenAI ReAct engine is not configured locally."""


def run_react_agent(
    request: str,
    runtime_options: Mapping[str, Any],
    *,
    model: str | None = None,
    skills_path: str | Path | list[str | Path] | None = None,
    tools_path: str | Path | list[str | Path] | None = None,
    max_steps: int = 8,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a ReAct loop using the OpenAI Responses API and external tools."""
    session = ReActSession(
        runtime_options,
        model=model,
        skills_path=skills_path,
        tools_path=tools_path,
        progress=progress,
    )
    return session.ask(request, max_steps=max_steps)


class ReActSession:
    """Reusable OpenAI ReAct conversation session."""

    def __init__(
        self,
        runtime_options: Mapping[str, Any],
        *,
        model: str | None = None,
        skills_path: str | Path | list[str | Path] | None = None,
        tools_path: str | Path | list[str | Path] | None = None,
        progress: Callable[[str], None] | None = None,
        events: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = _openai_client()
        self.model_name = model or os.getenv(OPENAI_MODEL_ENV) or DEFAULT_OPENAI_MODEL
        self.runtime_options = dict(runtime_options)
        self.progress = progress
        self.events = events
        self.tool_registry = build_tool_registry(self.runtime_options, tools_path=tools_path, progress=progress)
        self.skill_registry = build_skill_registry(self.tool_registry, skills_path=skills_path)
        self.input_items: list[Any] = []
        self.turn_count = 0

    def clear(self) -> None:
        """Clear conversation history while keeping registries and defaults."""
        self.input_items.clear()
        self.turn_count = 0

    def ask(self, request: str, *, max_steps: int = 8) -> dict[str, Any]:
        """Ask one user turn while preserving previous session context."""
        if max_steps <= 0:
            raise MusicAgentError("max_steps must be greater than 0.")
        if not request.strip():
            raise MusicAgentError("Agent request cannot be empty.")

        self.turn_count += 1
        self.runtime_options["request"] = request
        loaded_skill: Skill | None = None
        tool_calls: list[dict[str, Any]] = []
        last_external_result: Any = None
        self.input_items.append(
            {
                "role": "user",
                "content": _build_user_content(request, self.runtime_options),
            }
        )

        for step in range(1, max_steps + 1):
            active_tools = _core_openai_tools()
            if loaded_skill is not None:
                active_tools.extend(self.tool_registry.to_openai_tools(loaded_skill.allowed_tools))
            _report(self.progress, f"Agent ReAct: model step {step}/{max_steps}")
            _emit(
                self.events,
                {
                    "type": "agent_step",
                    "step": step,
                    "max_steps": max_steps,
                    "message": f"Agent ReAct: model step {step}/{max_steps}",
                },
            )
            response = self.client.responses.create(
                model=self.model_name,
                instructions=_agent_instructions(),
                input=self.input_items,
                tools=active_tools,
            )
            output_items = list(_get(response, "output") or [])
            self.input_items.extend(output_items)

            function_calls = [item for item in output_items if _get(item, "type") == "function_call"]
            if not function_calls:
                return {
                    "capability": "agent",
                    "engine": "openai_react",
                    "request": request,
                    "turn": self.turn_count,
                    "model": self.model_name,
                    "skill_used": loaded_skill.name if loaded_skill else None,
                    "tool_calls": tool_calls,
                    "final_answer": _response_text(response),
                    "result": last_external_result,
                }

            for call in function_calls:
                call_name = str(_get(call, "name") or "")
                call_id = str(_get(call, "call_id") or _get(call, "id") or "")
                arguments = _parse_arguments(_get(call, "arguments"))
                observation: dict[str, Any]
                is_external = False
                display_name = _display_tool_name(call_name, self.tool_registry)
                _emit(
                    self.events,
                    {
                        "type": "tool_call",
                        "step": step,
                        "tool": display_name,
                        "arguments": arguments,
                    },
                )
                try:
                    if call_name == "list_skills":
                        observation = {"ok": True, "skills": self.skill_registry.summaries()}
                    elif call_name == "load_skill":
                        skill_name = _required_argument(arguments, "name")
                        loaded_skill = self.skill_registry.get(skill_name)
                        observation = {
                            "ok": True,
                            "skill": loaded_skill.loaded_document(self.tool_registry),
                            "callable_function_names": {
                                tool_name: self.tool_registry.get(tool_name).openai_name
                                for tool_name in loaded_skill.allowed_tools
                            },
                        }
                    elif call_name == "list_tools":
                        skill_name = arguments.get("skill_name")
                        if skill_name:
                            skill = self.skill_registry.get(str(skill_name))
                            summaries = self.tool_registry.summaries(skill.allowed_tools)
                        else:
                            summaries = self.tool_registry.summaries()
                        observation = {"ok": True, "tools": summaries}
                    else:
                        if loaded_skill is None:
                            raise MusicAgentError("Load a skill before calling external music tools.")
                        spec = self.tool_registry.get(call_name)
                        if spec.name not in loaded_skill.allowed_tools:
                            raise MusicAgentError(f"Tool '{spec.name}' is not allowed by loaded skill '{loaded_skill.name}'.")
                        is_external = True
                        output = self.tool_registry.call(call_name, arguments)
                        last_external_result = output
                        observation = {"ok": True, "tool": spec.name, "output": output}
                except Exception as exc:  # The model needs an observation it can recover from.
                    observation = {
                        "ok": False,
                        "tool": call_name,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }

                tool_calls.append(
                    {
                        "step": step,
                        "tool": display_name,
                        "arguments": arguments,
                        "ok": bool(observation.get("ok")),
                        "external": is_external,
                        "observation": observation,
                    }
                )
                _emit(
                    self.events,
                    {
                        "type": "tool_result",
                        "step": step,
                        "tool": display_name,
                        "ok": bool(observation.get("ok")),
                        "external": is_external,
                        "observation": observation,
                    },
                )
                self.input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _json_dumps(observation),
                    }
                )

        raise MusicAgentError(f"OpenAI ReAct agent reached max_steps={max_steps} without a final answer.")


def _openai_client() -> Any:
    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIReActUnavailable("OPENAI_API_KEY is not set; falling back to keyword routing is allowed in auto mode.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIReActUnavailable("OpenAI SDK is not installed. Install with `python -m pip install -e '.[agent-openai]'`.") from exc
    return OpenAI()


def _core_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "list_skills",
            "description": "List available SKILL.md workflows and their allowed external tools.",
            "parameters": _schema({}),
        },
        {
            "type": "function",
            "name": "load_skill",
            "description": "Load one SKILL.md workflow by name before using its external tools.",
            "parameters": _schema(
                {
                    "name": {"type": "string", "description": "Skill name from list_skills."},
                },
                required=("name",),
            ),
        },
        {
            "type": "function",
            "name": "list_tools",
            "description": "List safe external tool summaries. Pass skill_name to limit to one skill's allowed tools.",
            "parameters": _schema(
                {
                    "skill_name": {"type": "string", "description": "Optional skill name."},
                }
            ),
        },
    ]


def _agent_instructions() -> str:
    return (
        "You are a music production agent. Use ReAct-style tool use: inspect available skills, "
        "load exactly one relevant skill, then call only external tools allowed by that skill. "
        "Skills are instructions and never execute code by themselves. External tools do the work. "
        "Use runtime defaults from the user context when arguments are omitted. Keep final answers concise "
        "and include generated artifact paths from tool outputs."
    )


def _build_user_content(request: str, runtime_options: Mapping[str, Any]) -> str:
    context = {
        key: value
        for key, value in runtime_options.items()
        if key != "progress" and value is not None and value is not False
    }
    return "User request:\n" + request + "\n\nRuntime defaults JSON:\n" + _json_dumps(context)


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments in (None, ""):
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        parsed = json.loads(str(raw_arguments))
    except json.JSONDecodeError as exc:
        raise MusicAgentError(f"Could not parse tool arguments JSON: {raw_arguments}") from exc
    if not isinstance(parsed, dict):
        raise MusicAgentError("Tool arguments must decode to a JSON object.")
    return parsed


def _required_argument(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise MusicAgentError(f"Tool argument '{name}' is required.")
    return value.strip()


def _response_text(response: Any) -> str:
    text = _get(response, "output_text")
    if isinstance(text, str):
        return text
    fragments: list[str] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content") or []:
            value = _get(content, "text")
            if isinstance(value, str):
                fragments.append(value)
    return "\n".join(fragments)


def _display_tool_name(call_name: str, tool_registry: ToolRegistry) -> str:
    if call_name in {"list_skills", "load_skill", "list_tools"}:
        return call_name
    try:
        return tool_registry.get(call_name).name
    except MusicAgentError:
        return call_name


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _emit(events: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if events is not None:
        events(event)
