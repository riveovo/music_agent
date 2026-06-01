"""Audio analysis capability."""

from __future__ import annotations

from pathlib import Path
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
from ..paths import ensure_output_dir, slugify, timestamp


def analyze_audio(
    audio: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Analyze basic metadata and loudness for an audio file."""
    source = require_audio_input(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else (
        default_batch_output_dir("analyze", source) if source.is_dir() else ensure_output_dir("analyze")
    )
    if source.is_dir():
        return _analyze_directory(
            source,
            target_dir,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )

    return _analyze_single(
        source,
        target_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _analyze_directory(
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
        extra={"files_found": len(files)},
    )
    _report(progress, "Audio analysis batch: complete")
    return result


def _analyze_single(
    audio_path: Path,
    output_dir: Path,
    *,
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

    if write_result_json:
        result_path = output_dir / f"analysis_{slugify(audio_path.stem)}_{timestamp()}.json"
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Audio analysis: wrote result JSON")
    return result


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


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
