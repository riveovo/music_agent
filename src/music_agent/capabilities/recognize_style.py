"""Heuristic music style recognition capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .analyze import analyze_audio
from ..audio import write_json
from ..audio_inputs import default_batch_output_dir, discover_audio_files, make_batch_result, require_audio_input
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp


STYLE_KEYWORDS = {
    "electronic": ("electronic", "edm", "techno", "synth", "电子"),
    "rock": ("rock", "guitar", "摇滚"),
    "lofi": ("lofi", "chill", "咖啡", "轻柔"),
    "classical": ("classical", "piano", "orchestra", "古典"),
    "ambient": ("ambient", "space", "冥想", "氛围"),
    "pop": ("pop", "流行"),
}


def recognize_style(
    audio: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Infer a coarse style label from file hints and basic audio metadata."""
    source = require_audio_input(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else (
        default_batch_output_dir("recognize_style", source) if source.is_dir() else ensure_output_dir("recognize_style")
    )
    if source.is_dir():
        return _recognize_directory(
            source,
            target_dir,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )
    return _recognize_single(
        source,
        target_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _recognize_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    recursive: bool,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    files, skipped = discover_audio_files(source_dir, recursive=recursive)
    if not files:
        raise MusicAgentError(f"No supported audio files found in directory: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Style recognition batch: found {len(files)} file(s)")
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        _report(progress, f"Style recognition batch: [{index}/{len(files)}] {rel_path}")
        try:
            results.append(
                _recognize_single(
                    audio_path,
                    output_dir,
                    keep_converted=keep_converted,
                    ncm_converter=ncm_converter,
                    progress=progress,
                    write_result_json=False,
                )
            )
        except MusicAgentError as exc:
            failures.append({"audio": str(audio_path), "error": str(exc)})
            _report(progress, f"Style recognition batch: failed {rel_path}: {exc}")

    result = make_batch_result(
        capability="recognize_style",
        input_path=source_dir,
        output_dir=output_dir,
        recursive=recursive,
        results=results,
        failures=failures,
        skipped=skipped,
        extra={"files_found": len(files)},
    )
    _report(progress, "Style recognition batch: complete")
    return result


def _recognize_single(
    audio_path: Path,
    output_dir: Path,
    *,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
    write_result_json: bool = True,
) -> dict[str, object]:
    _report(progress, f"Style recognition: preparing {audio_path.name}")
    analysis = analyze_audio(
        audio_path,
        output_dir=output_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )
    audio_path = Path(str(analysis["audio"]))
    stem = audio_path.stem.lower()
    loudness = analysis.get("loudness", {})
    mean_db = loudness.get("mean_db") if isinstance(loudness, dict) else None

    sidecar_style = _style_from_sidecar(audio_path)
    filename_style = _style_from_name(stem)
    style = sidecar_style or filename_style or _style_from_audio(analysis)
    energy = _energy_from_loudness(mean_db)
    mood = _mood_for(style, energy)
    confidence = _confidence(style, stem, mean_db)

    result = {
        "capability": "recognize_style",
        "audio": str(audio_path),
        "style": style,
        "mood": mood,
        "energy": energy,
        "confidence": confidence,
        "quality": "heuristic_mvp",
        "evidence": {
            "sidecar_hint": sidecar_style,
            "filename_hint": filename_style,
            "duration_seconds": analysis.get("duration_seconds"),
            "channels": analysis.get("channels"),
            "mean_db": mean_db,
        },
        "conversion": analysis.get("conversion"),
    }

    if write_result_json:
        result_path = output_dir / f"style_{slugify(audio_path.stem)}_{timestamp()}.json"
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Style recognition: wrote result JSON")
    return result


def _style_from_name(stem: str) -> str | None:
    for style, keywords in STYLE_KEYWORDS.items():
        if any(keyword in stem for keyword in keywords):
            return style
    return None


def _style_from_sidecar(audio_path: Path) -> str | None:
    sidecar = audio_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    style = payload.get("style")
    return style if isinstance(style, str) and style in STYLE_KEYWORDS else None


def _style_from_audio(analysis: dict[str, object]) -> str:
    duration = analysis.get("duration_seconds")
    loudness = analysis.get("loudness", {})
    mean_db = loudness.get("mean_db") if isinstance(loudness, dict) else None
    channels = analysis.get("channels")

    if isinstance(mean_db, (int, float)) and mean_db > -13:
        return "electronic"
    if isinstance(duration, (int, float)) and duration > 180:
        return "ambient"
    if channels == 1:
        return "lofi"
    return "pop"


def _energy_from_loudness(mean_db: object) -> str:
    if not isinstance(mean_db, (int, float)):
        return "unknown"
    if mean_db > -13:
        return "high"
    if mean_db > -24:
        return "medium"
    return "low"


def _mood_for(style: str, energy: str) -> str:
    if style == "electronic":
        return "bright" if energy != "low" else "focused"
    if style == "rock":
        return "driving"
    if style == "lofi":
        return "calm"
    if style == "classical":
        return "elegant"
    if style == "ambient":
        return "dreamy"
    return "uplifting" if energy == "high" else "neutral"


def _confidence(style: str, stem: str, mean_db: object) -> float:
    score = 0.42
    if _style_from_name(stem) == style:
        score += 0.24
    if isinstance(mean_db, (int, float)):
        score += 0.08
    return round(min(score, 0.78), 2)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
