"""Placeholder voice conversion using ffmpeg filter presets."""

from __future__ import annotations

from pathlib import Path

from ..audio import require_audio_file, require_tool, run_tool, write_json
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp


PRESETS = {
    "bright": "asetrate=44100*1.06,aresample=44100,atempo=0.943,highpass=f=120,equalizer=f=3200:width_type=o:width=1:g=4",
    "deep": "asetrate=44100*0.92,aresample=44100,atempo=1.087,lowpass=f=7200,equalizer=f=180:width_type=o:width=1:g=4",
    "robot": "aresample=8000,aresample=44100,aecho=0.8:0.8:35:0.35,acrusher=level_in=1:level_out=0.8:bits=8:mode=log",
}


def convert_voice(
    audio: str | Path,
    preset: str = "bright",
    output: str | Path | None = None,
) -> dict[str, object]:
    """Apply a simple voice-transform preset for MVP validation."""
    require_tool("ffmpeg")
    audio_path = require_audio_file(audio)
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise MusicAgentError(f"Unknown voice preset '{preset}'. Available presets: {available}.")

    output_path = Path(output).expanduser() if output else _default_output_path(audio_path, preset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            PRESETS[preset],
            str(output_path),
        ]
    )

    result = {
        "capability": "convert_voice",
        "quality": "placeholder_mvp",
        "audio": str(audio_path),
        "preset": preset,
        "output_audio": str(output_path),
        "notes": "This is a lightweight ffmpeg transform, not neural voice conversion.",
    }
    result_path = output_path.with_suffix(".json")
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    return result


def _default_output_path(audio_path: Path, preset: str) -> Path:
    name = f"voice_{preset}_{slugify(audio_path.stem)}_{timestamp()}.wav"
    return ensure_output_dir("convert_voice") / name

