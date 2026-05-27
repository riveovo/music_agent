"""Audio analysis capability."""

from __future__ import annotations

from pathlib import Path
import re

from ..audio import (
    ffprobe_json,
    first_audio_stream,
    parse_float,
    parse_int,
    require_audio_file,
    require_tool,
    run_tool,
    write_json,
)
from ..paths import ensure_output_dir, slugify, timestamp


def analyze_audio(audio: str | Path) -> dict[str, object]:
    """Analyze basic metadata and loudness for an audio file."""
    audio_path = require_audio_file(audio)
    metadata = ffprobe_json(audio_path)
    stream = first_audio_stream(metadata)
    fmt = metadata.get("format", {})
    loudness = _measure_loudness(audio_path)

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
        "summary": summary,
    }

    output_dir = ensure_output_dir("analyze")
    result_path = output_dir / f"analysis_{slugify(audio_path.stem)}_{timestamp()}.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
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

