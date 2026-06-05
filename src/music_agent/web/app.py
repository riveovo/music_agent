"""FastAPI application for the local Music Agent web service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from queue import Queue
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..errors import MusicAgentError
from ..paths import PROJECT_ROOT
from ..skills import build_skill_registry
from ..tools import build_tool_registry
from .artifacts import Artifact, ArtifactRegistry
from .sessions import SessionManager, WebConfig


MAX_UPLOAD_BYTES = 200 * 1024 * 1024
AUDIO_UPLOAD_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ncm", ".ogg", ".wav"}


class SessionCreateRequest(BaseModel):
    agent_engine: str | None = None
    openai_model: str | None = None
    skills_path: str | None = None
    tools_path: str | None = None
    max_steps: int | None = None
    runtime_options: dict[str, Any] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    content: str
    audio_artifact_id: str | None = None
    runtime_options: dict[str, Any] = Field(default_factory=dict)


def create_app(config: WebConfig | None = None) -> FastAPI:
    """Create the local web API and static UI app."""
    resolved_config = config or WebConfig()
    manager = SessionManager(resolved_config)
    artifacts = ArtifactRegistry(root=resolved_config.output_root)

    app = FastAPI(title="Music Agent Web", version="0.1.0")
    app.state.session_manager = manager
    app.state.artifact_registry = artifacts
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "music-agent-web",
            "ui_dist_available": _web_dist().exists(),
        }

    @app.get("/api/tools")
    async def list_tools() -> dict[str, Any]:
        try:
            registry = build_tool_registry(resolved_config.runtime_options, tools_path=resolved_config.tools_path)
            return {"ok": True, "tools": registry.summaries()}
        except MusicAgentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/skills")
    async def list_skills() -> dict[str, Any]:
        try:
            tool_registry = build_tool_registry(resolved_config.runtime_options, tools_path=resolved_config.tools_path)
            skill_registry = build_skill_registry(tool_registry, skills_path=resolved_config.skills_path)
            return {"ok": True, "skills": skill_registry.summaries()}
        except MusicAgentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions")
    async def create_session(body: SessionCreateRequest) -> dict[str, Any]:
        try:
            session = manager.create_session(
                agent_engine=body.agent_engine,
                openai_model=body.openai_model,
                skills_path=body.skills_path,
                tools_path=body.tools_path,
                max_steps=body.max_steps,
                runtime_options=body.runtime_options,
            )
            return {"ok": True, "session": session.as_dict()}
        except MusicAgentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageRequest) -> dict[str, Any]:
        try:
            session = manager.get(session_id)
            events: list[dict[str, Any]] = []
            runtime_updates = _message_runtime_updates(body, artifacts)
            result = await run_in_threadpool(
                session.ask,
                body.content,
                runtime_updates=runtime_updates,
                events=events.append,
            )
            enriched = _enriched_result(result, artifacts)
            return {"ok": True, "data": enriched, "events": events}
        except MusicAgentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/messages/stream")
    async def stream_message(session_id: str, body: MessageRequest) -> StreamingResponse:
        queue: Queue[dict[str, Any]] = Queue()

        async def run_turn() -> None:
            try:
                session = manager.get(session_id)
                runtime_updates = _message_runtime_updates(body, artifacts)
                result = await run_in_threadpool(
                    session.ask,
                    body.content,
                    runtime_updates=runtime_updates,
                    events=queue.put,
                )
                enriched = _enriched_result(result, artifacts)
                for chunk in _assistant_chunks(enriched):
                    queue.put({"type": "assistant_delta", "text": chunk})
                for artifact in enriched.get("artifacts", []):
                    queue.put({"type": "artifact", "artifact": artifact})
                queue.put({"type": "final", "data": enriched})
            except Exception as exc:
                queue.put({"type": "error", "error": str(exc), "error_type": type(exc).__name__})
            finally:
                queue.put({"type": "__done__"})

        async def event_stream() -> AsyncIterator[str]:
            task = asyncio.create_task(run_turn())
            try:
                while True:
                    event = await asyncio.to_thread(queue.get)
                    if event.get("type") == "__done__":
                        break
                    yield _sse(event)
            finally:
                await task

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str) -> dict[str, Any]:
        try:
            manager.get(session_id).clear()
            return {"ok": True}
        except MusicAgentError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/uploads/audio")
    async def upload_audio(
        file: UploadFile = File(...),
        session_id: str | None = Form(None),
    ) -> dict[str, Any]:
        if session_id is not None:
            try:
                manager.get(session_id)
            except MusicAgentError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        suffix = Path(file.filename or "audio").suffix.lower()
        if suffix not in AUDIO_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported audio upload extension.")

        target = artifacts.upload_path(session_id=session_id, filename=file.filename or "audio")
        size = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise MusicAgentError("Audio upload exceeds the 200 MB limit.")
                    handle.write(chunk)
            artifact = artifacts.register_path(target)
            return {"ok": True, "artifact": artifact.as_dict(), "audio_artifact_id": artifact.id}
        except MusicAgentError as exc:
            if target.exists():
                target.unlink()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str) -> FileResponse:
        try:
            artifact = artifacts.get(artifact_id)
            return FileResponse(
                artifact.path,
                media_type=artifact.mime_type,
                filename=artifact.name,
            )
        except MusicAgentError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    dist = _web_dist()
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    else:
        @app.get("/")
        async def missing_ui() -> dict[str, Any]:
            return {
                "ok": True,
                "message": "Web UI build is missing. Run `npm --prefix web install` and `npm --prefix web run build`.",
            }

    return app


def _message_runtime_updates(body: MessageRequest, artifacts: ArtifactRegistry) -> dict[str, Any]:
    updates = dict(body.runtime_options)
    if body.audio_artifact_id:
        artifact = artifacts.get(body.audio_artifact_id)
        if artifact.kind != "audio":
            raise MusicAgentError("audio_artifact_id must reference an audio artifact.")
        updates["audio"] = str(artifact.path)
    return updates


def _enriched_result(result: dict[str, Any], artifacts: ArtifactRegistry) -> dict[str, Any]:
    enriched = dict(result)
    enriched["artifacts"] = [artifact.as_dict() for artifact in artifacts.register_from_result(result)]
    return enriched


def _assistant_chunks(result: dict[str, Any], *, chunk_size: int = 120) -> list[str]:
    text = result.get("final_answer")
    if not isinstance(text, str) or not text:
        return []
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _sse(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "message")
    return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _web_dist() -> Path:
    return PROJECT_ROOT / "web" / "dist"
