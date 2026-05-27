"""Heuristic music style recognition capability."""

from __future__ import annotations

import json
from pathlib import Path

from .analyze import analyze_audio
from ..audio import write_json
from ..paths import ensure_output_dir, slugify, timestamp


STYLE_KEYWORDS = {
    "electronic": ("electronic", "edm", "techno", "synth", "电子"),
    "rock": ("rock", "guitar", "摇滚"),
    "lofi": ("lofi", "chill", "咖啡", "轻柔"),
    "classical": ("classical", "piano", "orchestra", "古典"),
    "ambient": ("ambient", "space", "冥想", "氛围"),
    "pop": ("pop", "流行"),
}


def recognize_style(audio: str | Path) -> dict[str, object]:
    """Infer a coarse style label from file hints and basic audio metadata."""
    analysis = analyze_audio(audio)
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
    }

    output_dir = ensure_output_dir("recognize_style")
    result_path = output_dir / f"style_{slugify(audio_path.stem)}_{timestamp()}.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
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
