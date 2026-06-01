"""Music style recognition capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .analyze import analyze_audio
from ..audio import write_json
from ..audio_inputs import (
    default_batch_output_dir,
    discover_audio_files,
    make_batch_result,
    prepared_audio_file,
    require_audio_input,
)
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp
from ..style_recognition import (
    EssentiaStyleConfig,
    has_complete_essentia_style_env,
    infer_energy_from_tags,
    mood_for_style_and_tags,
    recognize_style_with_essentia,
    resolve_essentia_style_config,
)


STYLE_KEYWORDS = {
    "electronic": ("electronic", "edm", "techno", "synth", "电子"),
    "rock": ("rock", "guitar", "摇滚"),
    "lofi": ("lofi", "chill", "咖啡", "轻柔"),
    "classical": ("classical", "piano", "orchestra", "古典"),
    "ambient": ("ambient", "space", "冥想", "氛围"),
    "pop": ("pop", "流行"),
}
STYLE_RECOGNITION_PROVIDERS = ("auto", "heuristic", "essentia")


def recognize_style(
    audio: str | Path,
    *,
    provider: str = "auto",
    output_dir: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    essentia_model_type: str | None = None,
    essentia_embedding_model_path: str | Path | None = None,
    essentia_classifier_model_path: str | Path | None = None,
    essentia_metadata_path: str | Path | None = None,
    essentia_top_k: int = 8,
    essentia_segment_seconds: float = 30.0,
    essentia_max_segments: int = 5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Infer a music style label from audio."""
    explicit_essentia_args = any(
        value is not None
        for value in (
            essentia_model_type,
            essentia_embedding_model_path,
            essentia_classifier_model_path,
            essentia_metadata_path,
        )
    )
    resolved_provider = _resolve_provider(provider, explicit_essentia_args=explicit_essentia_args)
    essentia_config: EssentiaStyleConfig | None = None
    if resolved_provider == "essentia":
        essentia_config = resolve_essentia_style_config(
            model_type=essentia_model_type,
            embedding_model_path=essentia_embedding_model_path,
            classifier_model_path=essentia_classifier_model_path,
            metadata_path=essentia_metadata_path,
            top_k=essentia_top_k,
            segment_seconds=essentia_segment_seconds,
            max_segments=essentia_max_segments,
        )

    source = require_audio_input(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else (
        default_batch_output_dir("recognize_style", source) if source.is_dir() else ensure_output_dir("recognize_style")
    )
    if source.is_dir():
        return _recognize_directory(
            source,
            target_dir,
            provider=resolved_provider,
            essentia_config=essentia_config,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )
    return _recognize_single(
        source,
        target_dir,
        provider=resolved_provider,
        essentia_config=essentia_config,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _recognize_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    provider: str,
    essentia_config: EssentiaStyleConfig | None,
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
                    provider=provider,
                    essentia_config=essentia_config,
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
        extra={"files_found": len(files), "provider": provider},
    )
    _report(progress, "Style recognition batch: complete")
    return result


def _recognize_single(
    audio_path: Path,
    output_dir: Path,
    *,
    provider: str,
    essentia_config: EssentiaStyleConfig | None,
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

    if provider == "essentia":
        if essentia_config is None:
            raise MusicAgentError("Internal error: missing Essentia style config.")
        with prepared_audio_file(
            audio_path,
            output_dir=output_dir,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        ) as prepared:
            style_output = recognize_style_with_essentia(
                prepared.processing_audio,
                essentia_config,
                progress=progress,
            )
        style = style_output.style
        confidence = style_output.confidence
        energy = infer_energy_from_tags(style_output.raw_tags, confidence)
        mood = mood_for_style_and_tags(style, style_output.raw_tags, energy)
        quality = "essentia_discogs_maest"
        provider_metadata: dict[str, object] = {
            "top_styles": style_output.top_styles,
            "raw_tags": style_output.raw_tags,
            "model": {
                "type": style_output.model_type,
                "embedding_model_path": str(style_output.embedding_model_path),
                "classifier_model_path": str(style_output.classifier_model_path),
                "metadata_path": str(style_output.metadata_path),
                "labels_count": style_output.labels_count,
            },
            "evidence": {
                "segments_analyzed": style_output.segments,
                "duration_seconds": analysis.get("duration_seconds"),
                "channels": analysis.get("channels"),
                "mean_db": mean_db,
            },
        }
    else:
        sidecar_style = _style_from_sidecar(audio_path)
        filename_style = _style_from_name(stem)
        style = sidecar_style or filename_style or _style_from_audio(analysis)
        energy = _energy_from_loudness(mean_db)
        mood = _mood_for(style, energy)
        confidence = _confidence(style, stem, mean_db)
        quality = "heuristic_mvp"
        provider_metadata = {
            "evidence": {
                "sidecar_hint": sidecar_style,
                "filename_hint": filename_style,
                "duration_seconds": analysis.get("duration_seconds"),
                "channels": analysis.get("channels"),
                "mean_db": mean_db,
            }
        }

    result = {
        "capability": "recognize_style",
        "provider": provider,
        "audio": str(audio_path),
        "style": style,
        "mood": mood,
        "energy": energy,
        "confidence": confidence,
        "quality": quality,
        "conversion": analysis.get("conversion"),
    } | provider_metadata

    if write_result_json:
        result_path = output_dir / f"style_{slugify(audio_path.stem)}_{timestamp()}.json"
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Style recognition: wrote result JSON")
    return result


def _resolve_provider(provider: str, *, explicit_essentia_args: bool) -> str:
    if provider not in STYLE_RECOGNITION_PROVIDERS:
        available = ", ".join(STYLE_RECOGNITION_PROVIDERS)
        raise MusicAgentError(f"Unknown style recognition provider '{provider}'. Available providers: {available}.")
    if provider == "auto":
        return "essentia" if explicit_essentia_args or has_complete_essentia_style_env() else "heuristic"
    return provider


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
