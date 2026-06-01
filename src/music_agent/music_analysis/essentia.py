"""Essentia-based musical analysis backend."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable, Iterable

from ..errors import MusicAgentError


ANALYSIS_PROVIDER_ENV = "MUSIC_AGENT_ANALYSIS_PROVIDER"
CHORD_HISTOGRAM_LABELS = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
    "Cm",
    "C#m",
    "Dm",
    "D#m",
    "Em",
    "Fm",
    "F#m",
    "Gm",
    "G#m",
    "Am",
    "A#m",
    "Bm",
)


@dataclass(frozen=True)
class EssentiaAnalysisConfig:
    max_sections: int = 12
    min_section_seconds: float = 12.0
    max_chord_events: int = 200


@dataclass(frozen=True)
class EssentiaAnalysisOutput:
    tempo: dict[str, object]
    meter: dict[str, object]
    tonal: dict[str, object]
    chords: dict[str, object]
    spectral: dict[str, object]
    sections: list[dict[str, object]]
    descriptors: dict[str, object]
    extractor_version: str | None


def analyze_with_essentia(
    audio_path: str | Path,
    *,
    config: EssentiaAnalysisConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> EssentiaAnalysisOutput:
    """Analyze musical descriptors with Essentia MusicExtractor."""
    cfg = config or EssentiaAnalysisConfig()
    standard = _import_essentia_standard()
    np = _import_numpy()

    _report(progress, "Essentia analysis: extracting music descriptors")
    extractor = standard.MusicExtractor(
        lowlevelStats=["mean", "stdev"],
        rhythmStats=["mean", "stdev"],
        tonalStats=["mean", "stdev"],
    )
    features, frame_features = extractor(str(audio_path))

    duration = _float_or_none(_pool_get(features, "metadata.audio_properties.length"))
    bpm = _float_or_none(_pool_get(features, "rhythm.bpm"))
    beats = _float_list(_pool_get(features, "rhythm.beats_position"))
    tonal_details = _run_tonal_extractor(standard, audio_path, progress=progress)

    tempo = _tempo(features, bpm, beats)
    tonal = _tonal(features, tonal_details)
    chords = _chords(features, tonal_details, duration, cfg.max_chord_events)
    spectral = _spectral(features)
    meter = _meter(beats, duration)
    sections = _sections(
        np,
        frame_features,
        beats,
        duration,
        bpm,
        max_sections=cfg.max_sections,
        min_section_seconds=cfg.min_section_seconds,
    )
    descriptors = {
        "rhythm": _descriptor_subset(
            features,
            (
                "rhythm.danceability",
                "rhythm.onset_rate",
                "rhythm.beats_count",
                "rhythm.bpm_histogram_first_peak_bpm",
                "rhythm.bpm_histogram_first_peak_weight",
                "rhythm.bpm_histogram_second_peak_bpm",
                "rhythm.bpm_histogram_second_peak_weight",
            ),
        ),
        "tonal": _descriptor_subset(
            features,
            (
                "tonal.tuning_frequency",
                "tonal.tuning_diatonic_strength",
                "tonal.tuning_equal_tempered_deviation",
                "tonal.tuning_nontempered_energy_ratio",
            ),
        ),
        "lowlevel": _descriptor_subset(
            features,
            (
                "lowlevel.dynamic_complexity",
                "lowlevel.average_loudness",
                "lowlevel.silence_rate_20dB.mean",
                "lowlevel.silence_rate_60dB.mean",
            ),
        ),
    }

    return EssentiaAnalysisOutput(
        tempo=tempo,
        meter=meter,
        tonal=tonal,
        chords=chords,
        spectral=spectral,
        sections=sections,
        descriptors=descriptors,
        extractor_version=_string_or_none(_pool_get(features, "metadata.version.extractor")),
    )


def _import_essentia_standard() -> object:
    try:
        standard = importlib.import_module("essentia.standard")
    except ImportError as exc:
        raise MusicAgentError(
            "Essentia analysis requires the essentia package. "
            "Install it with `python3.12 -m pip install -e \".[analysis-essentia]\"`."
        ) from exc
    if not hasattr(standard, "MusicExtractor"):
        raise MusicAgentError("Essentia is installed without MusicExtractor support.")
    return standard


def _import_numpy() -> object:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise MusicAgentError("Essentia analysis requires numpy.") from exc


def _run_tonal_extractor(
    standard: object,
    audio_path: str | Path,
    *,
    progress: Callable[[str], None] | None,
) -> dict[str, object] | None:
    if not hasattr(standard, "TonalExtractor") or not hasattr(standard, "MonoLoader"):
        return None
    try:
        _report(progress, "Essentia analysis: extracting chord progression")
        audio = standard.MonoLoader(filename=str(audio_path), sampleRate=44100, resampleQuality=4)()
        values = standard.TonalExtractor()(audio)
    except Exception:
        return None
    keys = (
        "chords_changes_rate",
        "chords_histogram",
        "chords_key",
        "chords_number_rate",
        "chords_progression",
        "chords_scale",
        "chords_strength",
        "hpcp",
        "hpcp_highres",
        "key_key",
        "key_scale",
        "key_strength",
    )
    return dict(zip(keys, values))


def _tempo(features: object, bpm: float | None, beats: list[float]) -> dict[str, object]:
    confidence = _float_or_none(_pool_get(features, "rhythm.bpm_histogram_first_peak_weight"))
    if confidence is None and beats:
        confidence = min(1.0, len(beats) / 64)
    return {
        "bpm": round(bpm, 3) if bpm is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "method": "essentia_music_extractor",
    }


def _meter(beats: list[float], duration: float | None) -> dict[str, object]:
    downbeats = beats[0::4] if len(beats) >= 4 else []
    return {
        "beats": [round(value, 3) for value in beats[:500]],
        "beats_count": len(beats),
        "downbeats": [round(value, 3) for value in downbeats[:200]],
        "downbeats_method": "estimated_every_4_beats" if downbeats else None,
        "duration_seconds": round(duration, 3) if duration is not None else None,
    }


def _tonal(features: object, tonal_details: dict[str, object] | None) -> dict[str, object]:
    key = _string_or_none(_pool_get(features, "tonal.key_edma.key"))
    scale = _string_or_none(_pool_get(features, "tonal.key_edma.scale"))
    strength = _float_or_none(_pool_get(features, "tonal.key_edma.strength"))
    if tonal_details:
        key = key or _string_or_none(tonal_details.get("key_key"))
        scale = scale or _string_or_none(tonal_details.get("key_scale"))
        strength = strength if strength is not None else _float_or_none(tonal_details.get("key_strength"))
    alternatives = []
    for profile in ("krumhansl", "temperley"):
        alt_key = _string_or_none(_pool_get(features, f"tonal.key_{profile}.key"))
        alt_scale = _string_or_none(_pool_get(features, f"tonal.key_{profile}.scale"))
        alt_strength = _float_or_none(_pool_get(features, f"tonal.key_{profile}.strength"))
        if alt_key:
            alternatives.append(
                {
                    "profile": profile,
                    "key": alt_key,
                    "scale": alt_scale,
                    "strength": round(alt_strength, 4) if alt_strength is not None else None,
                }
            )
    return {
        "key": key,
        "scale": scale,
        "key_strength": round(strength, 4) if strength is not None else None,
        "profile": "edma",
        "alternatives": alternatives,
    }


def _chords(
    features: object,
    tonal_details: dict[str, object] | None,
    duration: float | None,
    max_events: int,
) -> dict[str, object]:
    histogram = _float_list(
        tonal_details.get("chords_histogram") if tonal_details else None,
        fallback=_pool_get(features, "tonal.chords_histogram"),
    )
    progression = _string_list(tonal_details.get("chords_progression") if tonal_details else None)
    strengths = _float_list(tonal_details.get("chords_strength") if tonal_details else None)
    sequence = _chord_sequence(progression, strengths, duration, max_events)
    return {
        "key": _string_or_none(_pool_get(features, "tonal.chords_key"))
        or (_string_or_none(tonal_details.get("chords_key")) if tonal_details else None),
        "scale": _string_or_none(_pool_get(features, "tonal.chords_scale"))
        or (_string_or_none(tonal_details.get("chords_scale")) if tonal_details else None),
        "changes_rate": _round(_float_or_none(_pool_get(features, "tonal.chords_changes_rate"))),
        "number_rate": _round(_float_or_none(_pool_get(features, "tonal.chords_number_rate"))),
        "strength_mean": _round(_float_or_none(_pool_get(features, "tonal.chords_strength.mean"))),
        "histogram": _chord_histogram(histogram),
        "sequence": sequence,
        "sequence_method": "essentia_tonal_extractor_uniform_timing" if sequence else None,
    }


def _chord_sequence(
    progression: list[str],
    strengths: list[float],
    duration: float | None,
    max_events: int,
) -> list[dict[str, object]]:
    if not progression:
        return []
    limited = progression[:max_events]
    total = len(progression)
    step = float(duration or total) / total if total else 0.0
    merged: list[dict[str, object]] = []
    for index, chord in enumerate(limited):
        strength = strengths[index] if index < len(strengths) else None
        start = index * step
        end = (index + 1) * step
        if merged and merged[-1]["chord"] == chord:
            merged[-1]["end_seconds"] = round(end, 3)
            if strength is not None:
                values = merged[-1].setdefault("_strength_values", [])
                values.append(strength)
            continue
        item: dict[str, object] = {
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "chord": chord,
        }
        if strength is not None:
            item["_strength_values"] = [strength]
        merged.append(item)
    for item in merged:
        values = item.pop("_strength_values", None)
        if values:
            item["strength"] = round(sum(values) / len(values), 4)
    return merged


def _chord_histogram(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    labels = CHORD_HISTOGRAM_LABELS if len(values) == len(CHORD_HISTOGRAM_LABELS) else tuple(str(index) for index in range(len(values)))
    return {
        labels[index]: round(float(value), 6)
        for index, value in enumerate(values)
        if float(value) > 0
    }


def _spectral(features: object) -> dict[str, object]:
    return {
        "average_loudness": _round(_float_or_none(_pool_get(features, "lowlevel.average_loudness"))),
        "dynamic_complexity": _round(_float_or_none(_pool_get(features, "lowlevel.dynamic_complexity"))),
        "spectral_centroid_mean": _round(_float_or_none(_pool_get(features, "lowlevel.spectral_centroid.mean"))),
        "spectral_flux_mean": _round(_float_or_none(_pool_get(features, "lowlevel.spectral_flux.mean"))),
        "spectral_rolloff_mean": _round(_float_or_none(_pool_get(features, "lowlevel.spectral_rolloff.mean"))),
        "dissonance_mean": _round(_float_or_none(_pool_get(features, "lowlevel.dissonance.mean"))),
        "pitch_salience_mean": _round(_float_or_none(_pool_get(features, "lowlevel.pitch_salience.mean"))),
        "zero_crossing_rate_mean": _round(_float_or_none(_pool_get(features, "lowlevel.zerocrossingrate.mean"))),
        "ebu128_integrated": _round(_float_or_none(_pool_get(features, "lowlevel.loudness_ebu128.integrated"))),
        "ebu128_loudness_range": _round(_float_or_none(_pool_get(features, "lowlevel.loudness_ebu128.loudness_range"))),
    }


def _sections(
    np: object,
    frame_features: object,
    beats: list[float],
    duration: float | None,
    bpm: float | None,
    *,
    max_sections: int,
    min_section_seconds: float,
) -> list[dict[str, object]]:
    if duration is None or duration <= 0:
        return []
    boundaries = _section_boundaries(beats, duration, bpm, max_sections, min_section_seconds)
    feature_matrix = _section_feature_matrix(np, frame_features, boundaries, duration)
    labels = _section_labels(np, feature_matrix, len(boundaries) - 1)
    sections: list[dict[str, object]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        sections.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "label": labels[index],
                "interpretation": _section_interpretation(index, len(boundaries) - 1, labels[index]),
            }
        )
    return sections


def _section_boundaries(
    beats: list[float],
    duration: float,
    bpm: float | None,
    max_sections: int,
    min_section_seconds: float,
) -> list[float]:
    if len(beats) >= 8 and bpm:
        phrase_beats = 16
        phrase_seconds = 60 * phrase_beats / bpm
        while phrase_seconds < min_section_seconds and phrase_beats < 64:
            phrase_beats *= 2
            phrase_seconds = 60 * phrase_beats / bpm
        starts = [0.0]
        for beat_index in range(phrase_beats, len(beats), phrase_beats):
            value = beats[beat_index]
            if value - starts[-1] >= min_section_seconds * 0.75:
                starts.append(value)
        starts.append(duration)
    else:
        section_count = max(1, min(max_sections, round(duration / 30)))
        starts = [duration * index / section_count for index in range(section_count + 1)]

    starts = _merge_short_sections(sorted(set(round(value, 3) for value in starts)), duration, min_section_seconds)
    while len(starts) - 1 > max_sections:
        starts = _drop_smallest_internal_gap(starts)
    if starts[0] != 0.0:
        starts.insert(0, 0.0)
    if starts[-1] != duration:
        starts.append(duration)
    return starts


def _merge_short_sections(boundaries: list[float], duration: float, min_seconds: float) -> list[float]:
    if not boundaries:
        return [0.0, duration]
    result = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary != duration and boundary - result[-1] < min_seconds:
            continue
        result.append(boundary)
    if result[-1] != duration:
        result.append(duration)
    if len(result) == 1:
        result.append(duration)
    return result


def _drop_smallest_internal_gap(boundaries: list[float]) -> list[float]:
    if len(boundaries) <= 3:
        return boundaries
    gaps = [
        (boundaries[index + 1] - boundaries[index - 1], index)
        for index in range(1, len(boundaries) - 1)
    ]
    _, drop_index = min(gaps)
    return boundaries[:drop_index] + boundaries[drop_index + 1 :]


def _section_feature_matrix(np: object, frame_features: object, boundaries: list[float], duration: float) -> object | None:
    matrices = []
    for name in ("lowlevel.mfcc", "tonal.hpcp", "lowlevel.spectral_centroid", "lowlevel.spectral_flux"):
        matrix = _frame_matrix(np, _pool_get(frame_features, name))
        if matrix is not None:
            matrices.append(matrix)
    if not matrices:
        return None
    frame_count = min(matrix.shape[0] for matrix in matrices)
    if frame_count <= 0:
        return None
    merged = np.concatenate([matrix[:frame_count] for matrix in matrices], axis=1)
    section_vectors = []
    for start, end in zip(boundaries, boundaries[1:]):
        start_index = int((start / duration) * frame_count)
        end_index = max(start_index + 1, int((end / duration) * frame_count))
        end_index = min(frame_count, end_index)
        section_vectors.append(np.mean(merged[start_index:end_index], axis=0))
    return np.stack(section_vectors, axis=0) if section_vectors else None


def _frame_matrix(np: object, values: object) -> object | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0:
        return None
    return array


def _section_labels(np: object, matrix: object | None, section_count: int) -> list[str]:
    labels: list[str] = []
    prototypes: list[object] = []
    for index in range(section_count):
        if matrix is None:
            labels.append(chr(ord("A") + min(index, 25)))
            continue
        vector = matrix[index]
        best_label = None
        best_similarity = -1.0
        for prototype_index, prototype in enumerate(prototypes):
            similarity = _cosine_similarity(np, vector, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = chr(ord("A") + prototype_index)
        if best_label is not None and best_similarity >= 0.92:
            labels.append(best_label)
        else:
            labels.append(chr(ord("A") + min(len(prototypes), 25)))
            prototypes.append(vector)
    return labels


def _cosine_similarity(np: object, left: object, right: object) -> float:
    numerator = float(np.dot(left, right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _section_interpretation(index: int, total: int, label: str) -> str:
    if total <= 1:
        return "whole_track"
    if index == 0:
        return "intro_or_first_section"
    if index == total - 1 and label == "A":
        return "outro_or_return"
    return "repeated_material" if label in {"A", "B", "C"} else "contrasting_section"


def _descriptor_subset(features: object, names: Iterable[str]) -> dict[str, object]:
    return {
        name: _json_value(_pool_get(features, name))
        for name in names
        if _pool_get(features, name) is not None
    }


def _pool_get(pool: object, name: str) -> object | None:
    if pool is None:
        return None
    try:
        return pool[name]
    except Exception:
        return None


def _float_or_none(value: object, fallback: object | None = None) -> float | None:
    candidate = value if value is not None else fallback
    if candidate is None:
        return None
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _float_list(value: object, fallback: object | None = None) -> list[float]:
    candidate = value if value is not None else fallback
    if candidate is None:
        return []
    try:
        return [float(item) for item in candidate]
    except TypeError:
        return []


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    try:
        return [str(item) for item in value]
    except TypeError:
        return []


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _json_value(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
