"""Lightweight standard-library music generation provider."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import wave

from ..audio import write_json
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp
from .base import GenerationRequest


SAMPLE_RATE = 22_050


STYLE_PROFILES = {
    "electronic": {
        "tempo": 128,
        "root": 55.0,
        "scale": [0, 2, 3, 5, 7, 10],
        "energy": 0.78,
        "wave": "pulse",
        "mood": "bright",
    },
    "rock": {
        "tempo": 112,
        "root": 49.0,
        "scale": [0, 2, 3, 5, 7, 10],
        "energy": 0.72,
        "wave": "drive",
        "mood": "driving",
    },
    "lofi": {
        "tempo": 82,
        "root": 65.4,
        "scale": [0, 2, 3, 5, 7, 9, 10],
        "energy": 0.46,
        "wave": "soft",
        "mood": "calm",
    },
    "classical": {
        "tempo": 92,
        "root": 261.6,
        "scale": [0, 2, 4, 5, 7, 9, 11],
        "energy": 0.50,
        "wave": "sine",
        "mood": "elegant",
    },
    "ambient": {
        "tempo": 68,
        "root": 110.0,
        "scale": [0, 2, 5, 7, 9],
        "energy": 0.34,
        "wave": "soft",
        "mood": "dreamy",
    },
    "pop": {
        "tempo": 104,
        "root": 130.8,
        "scale": [0, 2, 4, 5, 7, 9, 11],
        "energy": 0.62,
        "wave": "sine",
        "mood": "uplifting",
    },
}


class SynthGenerator:
    """Small deterministic synth fallback that needs no external dependencies."""

    name = "synth"

    def generate(self, request: GenerationRequest) -> dict[str, object]:
        _validate_request(request, max_duration=120)
        detected_style = detect_style(request.prompt, request.style)
        profile = STYLE_PROFILES[detected_style]
        output_path = request.output or _default_output_path(request.prompt, detected_style)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_samples = int(SAMPLE_RATE * request.duration)
        tempo = float(profile["tempo"])
        root = float(profile["root"])
        energy = float(profile["energy"])
        scale = list(profile["scale"])
        wave_shape = str(profile["wave"])

        chord_roots = [0, 5, 3, 7]
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            for index in range(total_samples):
                t = index / SAMPLE_RATE
                beat = t * tempo / 60.0
                chord = chord_roots[int(beat // 4) % len(chord_roots)]
                note = scale[int(beat * 2) % len(scale)]

                bass_freq = _freq(root / 2.0, chord)
                chord_freqs = [_freq(root, chord + interval) for interval in (0, 4, 7)]
                lead_freq = _freq(root * 2.0, chord + note)

                bass = 0.34 * math.sin(2 * math.pi * bass_freq * t)
                pad = sum(0.10 * _osc(freq, t, wave_shape) for freq in chord_freqs)
                lead = 0.16 * _osc(lead_freq, t, "sine")
                drums = _drum_layer(beat, detected_style)
                noise = 0.025 * math.sin(2 * math.pi * 7400 * t) if detected_style == "lofi" else 0.0

                envelope = min(
                    1.0,
                    index / (SAMPLE_RATE * 0.08),
                    (total_samples - index) / (SAMPLE_RATE * 0.18),
                )
                sample = (bass + pad + lead + drums + noise) * energy * max(0.0, envelope)
                wav.writeframes(struct.pack("<h", _clip16(sample)))

        result = {
            "capability": "generate",
            "provider": self.name,
            "quality": "synth_mvp",
            "prompt": request.prompt,
            "style": detected_style,
            "duration_seconds": round(request.duration, 3),
            "sample_rate": SAMPLE_RATE,
            "output_audio": str(output_path),
            "notes": "Generated with a lightweight standard-library synthesizer.",
        }
        result_path = output_path.with_suffix(".json")
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        return result


def detect_style(prompt: str, explicit_style: str | None = None) -> str:
    text = f"{explicit_style or ''} {prompt}".lower()
    if any(word in text for word in ("电子", "electronic", "edm", "techno", "synth")):
        return "electronic"
    if any(word in text for word in ("摇滚", "rock", "guitar")):
        return "rock"
    if any(word in text for word in ("lofi", "chill", "放松", "轻柔", "咖啡")):
        return "lofi"
    if any(word in text for word in ("古典", "classical", "piano", "orchestra")):
        return "classical"
    if any(word in text for word in ("氛围", "ambient", "冥想", "space")):
        return "ambient"
    return "pop"


def _validate_request(request: GenerationRequest, max_duration: float) -> None:
    if request.duration <= 0:
        raise MusicAgentError("--duration must be greater than 0 seconds.")
    if request.duration > max_duration:
        raise MusicAgentError(f"--duration is capped at {int(max_duration)} seconds for this provider.")
    if not request.prompt.strip():
        raise MusicAgentError("--prompt cannot be empty.")


def _default_output_path(prompt: str, style: str) -> Path:
    name = f"generated_{style}_{slugify(prompt)}_{timestamp()}.wav"
    return ensure_output_dir("generate") / name


def _freq(base: float, semitones: int) -> float:
    return base * (2 ** (semitones / 12))


def _osc(freq: float, t: float, shape: str) -> float:
    phase = 2 * math.pi * freq * t
    sine = math.sin(phase)
    if shape == "pulse":
        return 0.72 * sine + 0.28 * math.sin(phase * 2)
    if shape == "drive":
        return math.tanh(1.8 * sine)
    if shape == "soft":
        return 0.78 * sine + 0.22 * math.sin(phase / 2)
    return sine


def _drum_layer(beat: float, style: str) -> float:
    if style in {"ambient", "classical"}:
        return 0.0
    beat_pos = beat % 1.0
    kick = 0.0
    snare = 0.0
    hat = 0.0
    if beat_pos < 0.11:
        kick = 0.36 * math.exp(-beat_pos * 16) * math.sin(2 * math.pi * 58 * beat_pos)
    if 0.48 < (beat % 2.0) < 0.62:
        snare = 0.14 * math.sin(2 * math.pi * 1800 * beat_pos)
    if (beat * 2) % 1.0 < 0.06:
        hat = 0.06 * math.sin(2 * math.pi * 6200 * beat_pos)
    return kick + snare + hat


def _clip16(sample: float) -> int:
    return int(max(-1.0, min(1.0, sample)) * 32767)

