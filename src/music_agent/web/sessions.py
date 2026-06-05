"""Agent session management for the web service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from ..agent import route_request
from ..agent.react import OpenAIReActUnavailable, ReActSession
from ..errors import MusicAgentError
from ..paths import OUTPUT_ROOT


EventSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class WebConfig:
    """Runtime defaults for the local web service."""

    agent_engine: str = "auto"
    openai_model: str | None = None
    skills_path: str | Path | list[str | Path] | None = None
    tools_path: str | Path | list[str | Path] | None = None
    max_steps: int = 8
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    output_root: str | Path = OUTPUT_ROOT


@dataclass
class AgentSession:
    """One browser conversation mapped to one Agent context."""

    id: str
    requested_engine: str
    actual_engine: str
    openai_model: str | None
    skills_path: str | Path | list[str | Path] | None
    tools_path: str | Path | list[str | Path] | None
    max_steps: int
    runtime_options: dict[str, Any]
    fallback_reason: str | None = None
    react_session: ReActSession | None = None
    lock: Lock = field(default_factory=Lock)
    _event_sink: EventSink | None = None

    def ask(
        self,
        request: str,
        *,
        runtime_updates: Mapping[str, Any] | None = None,
        events: EventSink | None = None,
    ) -> dict[str, Any]:
        if not request.strip():
            raise MusicAgentError("Agent request cannot be empty.")
        with self.lock:
            self._event_sink = events
            try:
                if runtime_updates:
                    self.runtime_options.update({key: value for key, value in runtime_updates.items() if value is not None})
                if self.react_session is not None:
                    self.react_session.runtime_options.update(self.runtime_options)
                    self.react_session.progress = self.emit_progress
                    self.react_session.events = self.emit_event
                    return self.react_session.ask(request, max_steps=self.max_steps)
                return route_request(
                    request=request,
                    **_route_runtime_options(self.runtime_options),
                    agent_engine="keyword",
                    openai_model=self.openai_model,
                    skills_path=self.skills_path,
                    tools_path=self.tools_path,
                    max_steps=self.max_steps,
                    progress=self.emit_progress,
                )
            finally:
                self._event_sink = None

    def clear(self) -> None:
        with self.lock:
            if self.react_session is not None:
                self.react_session.clear()

    def emit_progress(self, message: str) -> None:
        self.emit_event({"type": "agent_step", "message": message})

    def emit_event(self, event: dict[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(event)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "requested_engine": self.requested_engine,
            "engine": self.actual_engine,
            "openai_model": self.openai_model,
            "max_steps": self.max_steps,
            "fallback_reason": self.fallback_reason,
        }


class SessionManager:
    """In-memory session store for the first local single-user web version."""

    def __init__(self, config: WebConfig | None = None) -> None:
        self.config = config or WebConfig()
        self._sessions: dict[str, AgentSession] = {}

    def create_session(
        self,
        *,
        agent_engine: str | None = None,
        openai_model: str | None = None,
        skills_path: str | Path | list[str | Path] | None = None,
        tools_path: str | Path | list[str | Path] | None = None,
        max_steps: int | None = None,
        runtime_options: Mapping[str, Any] | None = None,
    ) -> AgentSession:
        engine = (agent_engine or self.config.agent_engine).strip().lower()
        if engine not in {"auto", "openai", "keyword"}:
            raise MusicAgentError("agent_engine must be one of: auto, openai, keyword.")
        resolved_max_steps = max_steps if max_steps is not None else self.config.max_steps
        if resolved_max_steps <= 0:
            raise MusicAgentError("max_steps must be greater than 0.")

        session = AgentSession(
            id=uuid4().hex,
            requested_engine=engine,
            actual_engine="keyword",
            openai_model=openai_model or self.config.openai_model,
            skills_path=skills_path if skills_path is not None else self.config.skills_path,
            tools_path=tools_path if tools_path is not None else self.config.tools_path,
            max_steps=resolved_max_steps,
            runtime_options=_merged_runtime_options(self.config.runtime_options, runtime_options),
        )

        if engine != "keyword":
            try:
                session.react_session = ReActSession(
                    session.runtime_options,
                    model=session.openai_model,
                    skills_path=session.skills_path,
                    tools_path=session.tools_path,
                    progress=session.emit_progress,
                    events=session.emit_event,
                )
                session.actual_engine = "openai_react"
            except OpenAIReActUnavailable as exc:
                if engine == "openai":
                    raise
                session.fallback_reason = str(exc)

        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AgentSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise MusicAgentError("Unknown session id.") from exc


def _merged_runtime_options(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {key: value for key, value in dict(base).items() if value is not None}
    if overrides:
        merged.update({key: value for key, value in dict(overrides).items() if value is not None})
    return merged


_ROUTE_OPTION_NAMES = {
    name
    for name in inspect.signature(route_request).parameters
    if name not in {"request", "agent_engine", "openai_model", "skills_path", "tools_path", "max_steps", "progress"}
}


def _route_runtime_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if key in _ROUTE_OPTION_NAMES}
