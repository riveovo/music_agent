"""Shared audio input discovery, conversion, and batch helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Callable, Iterator

from .audio import require_audio_file, run_tool, write_json
from .errors import MusicAgentError
from .paths import ensure_output_dir, slugify, timestamp


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".wave", ".mp3", ".flac", ".ncm")
NCM_CONVERTER_ENV = "MUSIC_AGENT_NCM_CONVERTER"
FALLBACK_TOOL_PATHS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


@dataclass(frozen=True)
class PreparedAudio:
    original_audio: Path
    processing_audio: Path
    conversion: dict[str, object]


def supported_audio_extensions_text() -> str:
    return ", ".join(SUPPORTED_AUDIO_EXTENSIONS)


def require_audio_input(audio: str | Path) -> Path:
    source = Path(audio).expanduser()
    if not source.exists():
        raise MusicAgentError(f"Audio input does not exist: {source}")
    if source.is_file() and source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise MusicAgentError(
            f"Unsupported audio format '{source.suffix}'. Supported formats: {supported_audio_extensions_text()}."
        )
    if not source.is_file() and not source.is_dir():
        raise MusicAgentError(f"Audio input is neither a file nor a directory: {source}")
    return source


def discover_audio_files(source_dir: Path, *, recursive: bool) -> tuple[list[Path], list[dict[str, str]]]:
    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(iterator):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            files.append(path)
        else:
            skipped.append({"path": str(path), "reason": "unsupported_format"})
    return files, skipped


@contextmanager
def prepared_audio_file(
    audio: str | Path,
    *,
    output_dir: Path,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Iterator[PreparedAudio]:
    audio_path = require_audio_file(audio)
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise MusicAgentError(
            f"Unsupported audio format '{audio_path.suffix}'. Supported formats: {supported_audio_extensions_text()}."
        )

    with tempfile.TemporaryDirectory(prefix="music_agent_audio_") as tmp:
        temp_dir = Path(tmp)
        prepared = prepare_audio_for_processing(
            audio_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )
        yield prepared


def prepare_audio_for_processing(
    audio_path: Path,
    *,
    output_dir: Path,
    temp_dir: Path,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> PreparedAudio:
    suffix = audio_path.suffix.lower()
    if suffix in (".wav", ".wave"):
        return PreparedAudio(
            original_audio=audio_path,
            processing_audio=audio_path,
            conversion={
                "required": False,
                "kept": False,
                "converted_audio": None,
                "tool": None,
            },
        )

    converted_path = (output_dir if keep_converted else temp_dir) / f"{slugify(audio_path.stem)}_converted.wav"
    if suffix in (".mp3", ".flac"):
        ffmpeg = find_executable("ffmpeg")
        if ffmpeg is None:
            raise MusicAgentError(
                "MP3/FLAC conversion requires ffmpeg. Add it to PATH or install it with Homebrew."
            )
        _report(progress, f"Audio conversion: {audio_path.suffix.lower()} -> wav via ffmpeg")
        convert_with_ffmpeg(ffmpeg, audio_path, converted_path)
        return PreparedAudio(
            original_audio=audio_path,
            processing_audio=converted_path,
            conversion={
                "required": True,
                "kept": keep_converted,
                "converted_audio": str(converted_path) if keep_converted else None,
                "tool": str(ffmpeg),
            },
        )

    if suffix == ".ncm":
        _report(progress, "Audio conversion: ncm -> wav via ncmdump + ffmpeg")
        decrypted = decrypt_ncm(
            audio_path,
            output_dir=temp_dir,
            ncm_converter=ncm_converter,
            expected_wav=converted_path,
        )
        if decrypted != converted_path:
            ffmpeg = find_executable("ffmpeg")
            if ffmpeg is None:
                raise MusicAgentError(
                    "NCM conversion requires ffmpeg after ncmdump. Add ffmpeg to PATH or install it with Homebrew."
                )
            convert_with_ffmpeg(ffmpeg, decrypted, converted_path)
        return PreparedAudio(
            original_audio=audio_path,
            processing_audio=converted_path,
            conversion={
                "required": True,
                "kept": keep_converted,
                "converted_audio": str(converted_path) if keep_converted else None,
                "tool": ncm_converter or str(find_executable("ncmdump") or "ncmdump"),
            },
        )

    raise MusicAgentError(f"Unsupported audio format: {audio_path.suffix}")


def make_batch_result(
    *,
    capability: str,
    input_path: Path,
    output_dir: Path,
    recursive: bool,
    results: list[dict[str, object]],
    failures: list[dict[str, str]],
    skipped: list[dict[str, str]],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result = {
        "capability": capability,
        "mode": "batch",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "recursive": recursive,
        "files_processed": len(results),
        "files_failed": len(failures),
        "files_skipped": len(skipped),
        "results": results,
        "failures": failures,
        "skipped": skipped,
    }
    if extra:
        result.update(extra)
    result_path = output_dir / f"{capability}_batch.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    return result


def default_batch_output_dir(kind: str, source: Path) -> Path:
    return ensure_output_dir(kind) / f"{slugify(source.stem, fallback='audio')}_{timestamp()}"


def batch_item_output_dir(output_dir: Path, source_dir: Path, audio_path: Path, *, recursive: bool, index: int) -> Path:
    rel_path = audio_path.relative_to(source_dir)
    rel_parent = rel_path.parent if recursive else Path()
    return output_dir / rel_parent / slugify(audio_path.stem, fallback=f"audio_{index:04d}")


def find_executable(name: str) -> Path | None:
    executable = shutil.which(name)
    if executable:
        return Path(executable)
    for directory in FALLBACK_TOOL_PATHS:
        candidate = directory / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def convert_with_ffmpeg(ffmpeg: Path, source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [
            str(ffmpeg),
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(target),
        ]
    )


def decrypt_ncm(
    audio_path: Path,
    *,
    output_dir: Path,
    ncm_converter: str | None,
    expected_wav: Path,
) -> Path:
    converter = ncm_converter or os.getenv(NCM_CONVERTER_ENV)
    if converter:
        before = _snapshot_files(output_dir)
        command = build_ncm_converter_command(converter, audio_path, output_dir, expected_wav)
        run_tool(command)
        if expected_wav.exists():
            return expected_wav
        candidate = _newest_converted_file(output_dir, before)
        if candidate is not None:
            return candidate
        raise MusicAgentError(
            f"NCM converter did not produce an audio file. Command: {' '.join(command)}"
        )

    ncmdump = find_executable("ncmdump")
    if ncmdump is None:
        raise MusicAgentError(
            "NCM conversion requires ncmdump. Add it to PATH, install it with Homebrew, "
            f"or set {NCM_CONVERTER_ENV}/--ncm-converter."
        )

    before = _snapshot_files(output_dir)
    run_tool([str(ncmdump), "-o", str(output_dir), str(audio_path)])
    candidate = _newest_converted_file(output_dir, before)
    if candidate is None:
        raise MusicAgentError(f"ncmdump did not produce an audio file for: {audio_path}")
    return candidate


def build_ncm_converter_command(
    converter: str,
    audio_path: Path,
    output_dir: Path,
    expected_wav: Path,
) -> list[str]:
    if "{input}" in converter or "{output}" in converter or "{output_dir}" in converter:
        formatted = converter.format(
            input=shlex.quote(str(audio_path)),
            output=shlex.quote(str(expected_wav)),
            output_dir=shlex.quote(str(output_dir)),
        )
        return shlex.split(formatted)
    return [converter, "-o", str(output_dir), str(audio_path)]


def _snapshot_files(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path for path in directory.rglob("*") if path.is_file()}


def _newest_converted_file(directory: Path, before: set[Path]) -> Path | None:
    candidates = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path not in before
        and path.suffix.lower() in (".wav", ".wave", ".mp3", ".flac")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
