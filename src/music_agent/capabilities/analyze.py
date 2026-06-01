"""Audio analysis capability."""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Callable

from ..audio import (
    ffprobe_json,
    first_audio_stream,
    parse_float,
    parse_int,
    require_tool,
    run_tool,
    write_json,
)
from ..audio_inputs import (
    default_batch_output_dir,
    discover_audio_files,
    make_batch_result,
    prepared_audio_file,
    require_audio_input,
)
from ..errors import MusicAgentError
from ..music_analysis import ANALYSIS_PROVIDER_ENV, EssentiaAnalysisConfig, analyze_with_essentia
from ..paths import ensure_output_dir, slugify, timestamp


ANALYSIS_PROVIDERS = ("auto", "basic", "essentia")


def analyze_audio(
    audio: str | Path,
    *,
    provider: str = "auto",
    output_dir: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    essentia_max_sections: int = 12,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Analyze basic metadata and loudness for an audio file."""
    resolved_provider = _resolve_provider(provider)
    essentia_config = EssentiaAnalysisConfig(max_sections=essentia_max_sections)
    source = require_audio_input(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else (
        default_batch_output_dir("analyze", source) if source.is_dir() else ensure_output_dir("analyze")
    )
    if source.is_dir():
        return _analyze_directory(
            source,
            target_dir,
            provider=resolved_provider,
            essentia_config=essentia_config,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )

    return _analyze_single(
        source,
        target_dir,
        provider=resolved_provider,
        essentia_config=essentia_config,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _analyze_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    provider: str,
    essentia_config: EssentiaAnalysisConfig,
    recursive: bool,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    files, skipped = discover_audio_files(source_dir, recursive=recursive)
    if not files:
        raise MusicAgentError(f"No supported audio files found in directory: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Audio analysis batch: found {len(files)} file(s)")
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        _report(progress, f"Audio analysis batch: [{index}/{len(files)}] {rel_path}")
        try:
            results.append(
                _analyze_single(
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
            _report(progress, f"Audio analysis batch: failed {rel_path}: {exc}")

    result = make_batch_result(
        capability="analyze",
        input_path=source_dir,
        output_dir=output_dir,
        recursive=recursive,
        results=results,
        failures=failures,
        skipped=skipped,
        extra={"files_found": len(files), "provider": provider},
    )
    _report(progress, "Audio analysis batch: complete")
    return result


def _analyze_single(
    audio_path: Path,
    output_dir: Path,
    *,
    provider: str,
    essentia_config: EssentiaAnalysisConfig,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
    write_result_json: bool = True,
) -> dict[str, object]:
    _report(progress, f"Audio analysis: preparing {audio_path.name}")
    with prepared_audio_file(
        audio_path,
        output_dir=output_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    ) as prepared:
        metadata = ffprobe_json(prepared.processing_audio)
        loudness = _measure_loudness(prepared.processing_audio)
        essentia_output = None
        if provider == "essentia":
            essentia_output = analyze_with_essentia(
                prepared.processing_audio,
                config=essentia_config,
                progress=progress,
            )

    stream = first_audio_stream(metadata)
    fmt = metadata.get("format", {})

    duration = parse_float(stream.get("duration")) or parse_float(fmt.get("duration"))
    channels = parse_int(stream.get("channels"))
    sample_rate = parse_int(stream.get("sample_rate"))
    bitrate = parse_int(stream.get("bit_rate")) or parse_int(fmt.get("bit_rate"))
    codec = stream.get("codec_name") or "unknown"
    format_name = fmt.get("format_name") or "unknown"
    size_bytes = parse_int(fmt.get("size"))

    summary = _build_summary(duration, channels, sample_rate, codec, loudness)
    result = {
        "capability": "analyze",
        "provider": provider,
        "quality": "essentia_music_analysis" if provider == "essentia" else "basic_metadata",
        "audio": str(audio_path),
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "codec": codec,
        "format": format_name,
        "sample_rate": sample_rate,
        "channels": channels,
        "bitrate": bitrate,
        "size_bytes": size_bytes,
        "loudness": loudness,
        "conversion": prepared.conversion,
        "summary": summary,
    }
    if essentia_output is not None:
        result.update(
            {
                "tempo": essentia_output.tempo,
                "meter": essentia_output.meter,
                "tonal": essentia_output.tonal,
                "chords": essentia_output.chords,
                "spectral": essentia_output.spectral,
                "sections": essentia_output.sections,
                "descriptors": essentia_output.descriptors,
                "extractor": {
                    "name": "essentia_music_extractor",
                    "version": essentia_output.extractor_version,
                },
                "summary": _build_musical_summary(summary, essentia_output),
            }
        )

    if write_result_json:
        result_path = output_dir / f"analysis_{slugify(audio_path.stem)}_{timestamp()}.json"
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Audio analysis: wrote result JSON")
    return result


def _resolve_provider(provider: str) -> str:
    if provider not in ANALYSIS_PROVIDERS:
        available = ", ".join(ANALYSIS_PROVIDERS)
        raise MusicAgentError(f"Unknown analysis provider '{provider}'. Available providers: {available}.")
    if provider == "auto":
        env_provider = os.getenv(ANALYSIS_PROVIDER_ENV)
        return env_provider if env_provider in ("basic", "essentia") else "basic"
    return provider


def _measure_loudness(audio_path: Path) -> dict[str, float | None]:
    require_tool("ffmpeg")
    result = run_tool(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
    )
    text = result.stderr
    return {
        "mean_db": _extract_volume(text, "mean_volume"),
        "max_db": _extract_volume(text, "max_volume"),
    }


def _extract_volume(text: str, key: str) -> float | None:
    match = re.search(rf"{key}:\s*(-?\d+(?:\.\d+)?) dB", text)
    return float(match.group(1)) if match else None


def _build_summary(
    duration: float | None,
    channels: int | None,
    sample_rate: int | None,
    codec: str,
    loudness: dict[str, float | None],
) -> dict[str, str]:
    length = "unknown length"
    if duration is not None:
        if duration < 20:
            length = "short clip"
        elif duration < 150:
            length = "song-length track"
        else:
            length = "long-form audio"

    channel_text = "unknown channel layout"
    if channels == 1:
        channel_text = "mono"
    elif channels == 2:
        channel_text = "stereo"
    elif channels:
        channel_text = f"{channels}-channel"

    mean_db = loudness.get("mean_db")
    if mean_db is None:
        energy = "unknown energy"
    elif mean_db > -13:
        energy = "high energy"
    elif mean_db > -24:
        energy = "medium energy"
    else:
        energy = "low energy"

    fidelity = "unknown fidelity"
    if sample_rate:
        fidelity = "standard fidelity" if sample_rate >= 22050 else "low sample rate"

    return {
        "length": length,
        "channel_layout": channel_text,
        "energy": energy,
        "fidelity": fidelity,
        "codec_note": f"encoded as {codec}",
    }


def _build_musical_summary(
    base: dict[str, str],
    essentia_output: object,
) -> dict[str, object]:
    tempo = essentia_output.tempo
    tonal = essentia_output.tonal
    sections = essentia_output.sections
    bpm = tempo.get("bpm") if isinstance(tempo, dict) else None
    key = tonal.get("key") if isinstance(tonal, dict) else None
    scale = tonal.get("scale") if isinstance(tonal, dict) else None
    section_pattern = "-".join(str(section["label"]) for section in sections) if sections else None
    musical = {
        "tempo": f"{bpm} BPM" if bpm is not None else "unknown tempo",
        "key": f"{key} {scale}".strip() if key else "unknown key",
        "structure": section_pattern or "unknown structure",
    }
    return base | {"musical": musical}


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
