"""Small audio helpers built around ffmpeg/ffprobe."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .errors import MusicAgentError


FALLBACK_TOOL_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


def require_audio_file(audio: str | Path) -> Path:
    path = Path(audio).expanduser()
    if not path.exists():
        raise MusicAgentError(f"Audio file does not exist: {path}")
    if not path.is_file():
        raise MusicAgentError(f"Audio path is not a file: {path}")
    return path


def require_tool(name: str) -> str:
    executable = _resolve_tool(name)
    if executable is None:
        raise MusicAgentError(
            f"Required tool '{name}' was not found on PATH. Install ffmpeg first."
        )
    return str(executable)


def run_tool(args: list[str]) -> subprocess.CompletedProcess[str]:
    resolved_args = [str(_resolve_tool(args[0]) or args[0]), *args[1:]]
    try:
        return subprocess.run(
            resolved_args,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MusicAgentError(f"Command not found: {resolved_args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 900:
            detail = detail[-900:]
        raise MusicAgentError(f"Command failed: {' '.join(resolved_args)}\n{detail}") from exc


def _resolve_tool(name: str) -> Path | None:
    path = Path(name)
    if path.is_absolute() or path.parent != Path("."):
        return path if path.exists() else None
    executable = shutil.which(name)
    if executable:
        return Path(executable)
    for directory in FALLBACK_TOOL_DIRS:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def ffprobe_json(audio: str | Path) -> dict[str, Any]:
    require_tool("ffprobe")
    audio_path = require_audio_file(audio)
    result = run_tool(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(audio_path),
        ]
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MusicAgentError(f"Could not parse ffprobe output for {audio_path}") from exc


def first_audio_stream(metadata: dict[str, Any]) -> dict[str, Any]:
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    raise MusicAgentError("No audio stream found in file.")


def parse_float(value: Any) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
