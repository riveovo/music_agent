"""Heuristic vocal/accompaniment separation using ffmpeg filters."""

from __future__ import annotations

from pathlib import Path

from ..audio import require_audio_file, require_tool, run_tool, write_json
from ..paths import ensure_output_dir, slugify, timestamp


def separate_stems(audio: str | Path, output_dir: str | Path | None = None) -> dict[str, object]:
    """Create MVP-quality vocal and accompaniment approximations."""
    require_tool("ffmpeg")
    audio_path = require_audio_file(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else _default_output_dir(audio_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = target_dir / "vocals.wav"
    accompaniment_path = target_dir / "accompaniment.wav"

    run_tool(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            "highpass=f=120,lowpass=f=3600,acompressor=threshold=-18dB:ratio=2:attack=20:release=250",
            str(vocals_path),
        ]
    )
    run_tool(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            "equalizer=f=1200:width_type=o:width=2:g=-10,lowpass=f=12000,highpass=f=40",
            str(accompaniment_path),
        ]
    )

    result = {
        "capability": "separate_stems",
        "quality": "heuristic_mvp",
        "audio": str(audio_path),
        "output_dir": str(target_dir),
        "stems": {
            "vocals": str(vocals_path),
            "accompaniment": str(accompaniment_path),
        },
        "notes": "This is a lightweight ffmpeg-filter approximation, not model-grade source separation.",
    }
    result_path = target_dir / "separation.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    return result


def _default_output_dir(audio_path: Path) -> Path:
    return ensure_output_dir("separate_stems") / f"{slugify(audio_path.stem)}_{timestamp()}"

