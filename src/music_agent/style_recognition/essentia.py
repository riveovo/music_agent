"""Essentia-based music style recognition adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

from ..errors import MusicAgentError


ESSENTIA_STYLE_MODEL_TYPES = ("discogs519_maest_30s",)
ESSENTIA_STYLE_ENV = {
    "model_type": "MUSIC_AGENT_STYLE_ESSENTIA_MODEL_TYPE",
    "embedding_model_path": "MUSIC_AGENT_STYLE_ESSENTIA_EMBEDDING_MODEL_PATH",
    "classifier_model_path": "MUSIC_AGENT_STYLE_ESSENTIA_CLASSIFIER_MODEL_PATH",
    "metadata_path": "MUSIC_AGENT_STYLE_ESSENTIA_METADATA_PATH",
}


@dataclass(frozen=True)
class EssentiaStyleConfig:
    model_type: str
    embedding_model_path: Path
    classifier_model_path: Path
    metadata_path: Path
    top_k: int = 8
    segment_seconds: float = 30.0
    max_segments: int = 5


@dataclass(frozen=True)
class EssentiaStyleOutput:
    style: str
    confidence: float
    top_styles: list[dict[str, object]]
    raw_tags: list[dict[str, object]]
    segments: list[dict[str, object]]
    labels_count: int
    model_type: str
    embedding_model_path: Path
    classifier_model_path: Path
    metadata_path: Path


def has_complete_essentia_style_env() -> bool:
    return bool(
        os.getenv(ESSENTIA_STYLE_ENV["embedding_model_path"])
        and os.getenv(ESSENTIA_STYLE_ENV["classifier_model_path"])
        and os.getenv(ESSENTIA_STYLE_ENV["metadata_path"])
    )


def resolve_essentia_style_config(
    *,
    model_type: str | None = None,
    embedding_model_path: str | Path | None = None,
    classifier_model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    top_k: int = 8,
    segment_seconds: float = 30.0,
    max_segments: int = 5,
) -> EssentiaStyleConfig:
    resolved_model_type = model_type or os.getenv(ESSENTIA_STYLE_ENV["model_type"]) or "discogs519_maest_30s"
    if resolved_model_type not in ESSENTIA_STYLE_MODEL_TYPES:
        supported = ", ".join(ESSENTIA_STYLE_MODEL_TYPES)
        raise MusicAgentError(
            f"Unsupported Essentia style model_type '{resolved_model_type}'. Supported types: {supported}."
        )

    embedding = embedding_model_path or os.getenv(ESSENTIA_STYLE_ENV["embedding_model_path"])
    classifier = classifier_model_path or os.getenv(ESSENTIA_STYLE_ENV["classifier_model_path"])
    metadata = metadata_path or os.getenv(ESSENTIA_STYLE_ENV["metadata_path"])
    if not all([embedding, classifier, metadata]):
        raise MusicAgentError(
            "Essentia style recognition requires embedding_model_path, classifier_model_path, and metadata_path. "
            "Pass --essentia-embedding-model-path/--essentia-classifier-model-path/--essentia-metadata-path "
            "or set MUSIC_AGENT_STYLE_ESSENTIA_* environment variables."
        )

    embedding_path = _require_file(embedding, "Essentia embedding model")
    classifier_path = _require_file(classifier, "Essentia classifier model")
    metadata_file = _require_file(metadata, "Essentia metadata")
    if top_k <= 0:
        raise MusicAgentError("Essentia style top_k must be positive.")
    if segment_seconds <= 0:
        raise MusicAgentError("Essentia style segment_seconds must be positive.")
    if max_segments <= 0:
        raise MusicAgentError("Essentia style max_segments must be positive.")

    return EssentiaStyleConfig(
        model_type=resolved_model_type,
        embedding_model_path=embedding_path,
        classifier_model_path=classifier_path,
        metadata_path=metadata_file,
        top_k=top_k,
        segment_seconds=segment_seconds,
        max_segments=max_segments,
    )


def recognize_style_with_essentia(
    audio_path: str | Path,
    config: EssentiaStyleConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> EssentiaStyleOutput:
    """Recognize music style using Essentia TensorFlow models."""
    essentia, standard = _import_essentia()
    np = _import_numpy()
    labels = _load_labels(config.metadata_path)
    _report(progress, "Essentia style: loading audio at 16 kHz")
    audio = standard.MonoLoader(filename=str(audio_path), sampleRate=16000, resampleQuality=4)()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise MusicAgentError(f"Essentia style recognition could not load audio: {audio_path}")

    _report(progress, "Essentia style: loading embedding and classifier models")
    embedding_model = standard.TensorflowPredictMAEST(
        graphFilename=str(config.embedding_model_path),
        output="PartitionedCall/Identity_12",
    )
    classifier_model = standard.TensorflowPredict(
        graphFilename=str(config.classifier_model_path),
        inputs=["embeddings"],
        outputs=["PartitionedCall/Identity_1"],
    )

    windows = _make_windows(audio.size, 16000, config.segment_seconds, config.max_segments)
    _report(progress, f"Essentia style: analyzing {len(windows)} internal window(s)")
    segment_predictions: list[object] = []
    segments: list[dict[str, object]] = []
    for index, (start, end) in enumerate(windows, start=1):
        segment = audio[start:end]
        if segment.size == 0:
            continue
        _report(progress, f"Essentia style: window {index}/{len(windows)}")
        embeddings = embedding_model(segment)
        pool = essentia.Pool()
        pool.set("embeddings", embeddings)
        predictions = classifier_model(pool)["PartitionedCall/Identity_1"]
        vector = _prediction_vector(np, predictions, expected_size=len(labels))
        segment_predictions.append(vector)
        top_index = int(np.argmax(vector)) if vector.size else -1
        segments.append(
            {
                "index": index,
                "start_seconds": round(start / 16000, 3),
                "end_seconds": round(end / 16000, 3),
                "top_label": labels[top_index] if 0 <= top_index < len(labels) else None,
                "top_score": round(float(vector[top_index]), 6) if 0 <= top_index < vector.size else None,
            }
        )

    if not segment_predictions:
        raise MusicAgentError("Essentia style recognition produced no segment predictions.")
    scores = np.mean(np.stack(segment_predictions, axis=0), axis=0)
    raw_tags = _top_raw_tags(np, labels, scores, config.top_k)
    top_styles = _normalize_styles(raw_tags, config.top_k)
    style = str(top_styles[0]["style"]) if top_styles else "unknown"
    confidence = float(top_styles[0]["score"]) if top_styles else 0.0

    return EssentiaStyleOutput(
        style=style,
        confidence=round(confidence, 4),
        top_styles=top_styles,
        raw_tags=raw_tags,
        segments=segments,
        labels_count=len(labels),
        model_type=config.model_type,
        embedding_model_path=config.embedding_model_path,
        classifier_model_path=config.classifier_model_path,
        metadata_path=config.metadata_path,
    )


def _import_essentia() -> tuple[object, object]:
    try:
        essentia = importlib.import_module("essentia")
        standard = importlib.import_module("essentia.standard")
    except ImportError as exc:
        raise MusicAgentError(
            "Essentia style recognition requires essentia-tensorflow. "
            "Install the optional extra with `python3.12 -m pip install -e \".[style-essentia]\"`."
        ) from exc
    required = ("MonoLoader", "TensorflowPredictMAEST", "TensorflowPredict")
    missing = [name for name in required if not hasattr(standard, name)]
    if missing:
        raise MusicAgentError(
            "Essentia is installed without the required TensorFlow algorithms: "
            f"{', '.join(missing)}. Install essentia-tensorflow instead of plain essentia."
        )
    return essentia, standard


def _import_numpy() -> object:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise MusicAgentError("Essentia style recognition requires numpy.") from exc


def _load_labels(metadata_path: Path) -> list[str]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicAgentError(f"Could not read Essentia metadata JSON: {metadata_path}") from exc
    labels = _extract_labels(payload)
    if not labels:
        raise MusicAgentError(f"Essentia metadata did not contain class labels: {metadata_path}")
    return labels


def _extract_labels(payload: object) -> list[str]:
    if isinstance(payload, dict):
        for key in ("classes", "labels", "class_list"):
            labels = _extract_labels(payload.get(key))
            if labels:
                return labels
        if "metadata" in payload:
            labels = _extract_labels(payload["metadata"])
            if labels:
                return labels
    if isinstance(payload, list):
        labels: list[str] = []
        for item in payload:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                value = item.get("name") or item.get("label") or item.get("class")
                if isinstance(value, str):
                    labels.append(value)
        return labels
    return []


def _make_windows(
    total_samples: int,
    sample_rate: int,
    segment_seconds: float,
    max_segments: int,
) -> list[tuple[int, int]]:
    window = max(1, int(segment_seconds * sample_rate))
    if total_samples <= window:
        return [(0, total_samples)]
    duration = total_samples / sample_rate
    inner_start = duration * 0.1
    inner_end = duration * 0.9
    available_start = int(max(0.0, min(inner_start, duration - segment_seconds)) * sample_rate)
    available_end = int(max(available_start + window, inner_end * sample_rate))
    available_end = min(total_samples, available_end)
    latest_start = max(0, available_end - window)
    if max_segments == 1 or latest_start <= available_start:
        start = min(max(available_start, 0), max(0, total_samples - window))
        return [(start, min(total_samples, start + window))]
    step = (latest_start - available_start) / (max_segments - 1)
    starts = [round(available_start + step * index) for index in range(max_segments)]
    windows: list[tuple[int, int]] = []
    seen: set[int] = set()
    for start in starts:
        bounded = min(max(0, int(start)), max(0, total_samples - window))
        if bounded in seen:
            continue
        seen.add(bounded)
        windows.append((bounded, min(total_samples, bounded + window)))
    return windows or [(0, min(total_samples, window))]


def _prediction_vector(np: object, predictions: object, *, expected_size: int) -> object:
    array = np.asarray(predictions, dtype=np.float32)
    if array.ndim == 0:
        raise MusicAgentError("Essentia classifier returned a scalar prediction.")
    if array.ndim > 1:
        array = np.mean(array, axis=0)
    array = array.reshape(-1)
    if expected_size and array.size != expected_size:
        raise MusicAgentError(
            f"Essentia classifier returned {array.size} scores, but metadata has {expected_size} labels."
        )
    return array


def _top_raw_tags(np: object, labels: list[str], scores: object, top_k: int) -> list[dict[str, object]]:
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "label": labels[int(index)],
            "score": round(float(scores[int(index)]), 6),
        }
        for index in order
    ]


def _normalize_styles(raw_tags: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for tag in raw_tags:
        label = str(tag["label"])
        score = float(tag["score"])
        style = normalize_discogs_label(label)
        scores[style] = max(scores.get(style, 0.0), score)
        evidence.setdefault(style, []).append(label)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {
            "style": style,
            "score": round(score, 6),
            "evidence": evidence.get(style, [])[:3],
        }
        for style, score in ranked
    ]


def normalize_discogs_label(label: str) -> str:
    normalized = _normalize_text(label)
    if any(token in normalized for token in ("ambient", "drone", "new age", "field recording")):
        return "ambient"
    if any(token in normalized for token in ("hip hop", "hip-hop", "rap", "trap", "grime")):
        return "hiphop"
    if any(token in normalized for token in ("electronic", "house", "techno", "trance", "edm", "dubstep", "drum n bass", "breakbeat")):
        return "electronic"
    if any(token in normalized for token in ("rock", "punk", "metal", "grunge", "shoegaze", "post-rock")):
        return "rock"
    if any(token in normalized for token in ("pop", "ballad", "vocal", "kayokyoku")):
        return "pop"
    if any(token in normalized for token in ("r&b", "rhythm & blues", "soul", "funk", "gospel")):
        return "rnb"
    if "jazz" in normalized or any(token in normalized for token in ("bebop", "swing", "fusion")):
        return "jazz"
    if any(token in normalized for token in ("classical", "baroque", "opera", "oratorio", "symphony", "romantic")):
        return "classical"
    if any(token in normalized for token in ("folk", "country", "bluegrass", "americana")):
        return "folk"
    if any(token in normalized for token in ("latin", "salsa", "bossa nova", "reggaeton", "samba", "tango")):
        return "latin"
    if any(token in normalized for token in ("reggae", "dub", "dancehall", "ska")):
        return "reggae"
    if any(token in normalized for token in ("blues", "boogie woogie")):
        return "blues"
    if any(token in normalized for token in ("stage", "screen", "score", "soundtrack")):
        return "soundtrack"
    return _top_level_discogs_label(label)


def infer_energy_from_tags(raw_tags: Iterable[dict[str, object]], confidence: float) -> str:
    text = " ".join(str(tag.get("label", "")) for tag in raw_tags)
    normalized = _normalize_text(text)
    if confidence < 0.08:
        return "unknown"
    if any(token in normalized for token in ("hard", "techno", "metal", "punk", "drum n bass", "breakbeat", "dance", "house")):
        return "high"
    if any(token in normalized for token in ("ambient", "drone", "ballad", "downtempo", "minimal", "classical")):
        return "low"
    return "medium"


def mood_for_style_and_tags(style: str, raw_tags: Iterable[dict[str, object]], energy: str) -> str:
    text = " ".join(str(tag.get("label", "")) for tag in raw_tags)
    normalized = _normalize_text(text)
    if any(token in normalized for token in ("sad", "melanchol", "dark", "doom")):
        return "melancholic"
    if any(token in normalized for token in ("happy", "party", "dance", "disco")):
        return "upbeat"
    if style == "ambient" or energy == "low":
        return "calm"
    if style in {"rock", "hiphop", "electronic"} and energy == "high":
        return "driving"
    if style == "classical":
        return "elegant"
    return "neutral"


def _normalize_text(value: str) -> str:
    return value.replace("---", " ").replace("_", " ").replace("/", " ").lower()


def _top_level_discogs_label(label: str) -> str:
    top_level = label.split("---", 1)[0].strip().lower()
    return top_level.replace("&", "and").replace(" ", "_") or "unknown"


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise MusicAgentError(f"{label} file does not exist: {resolved}")
    if not resolved.is_file():
        raise MusicAgentError(f"{label} path is not a file: {resolved}")
    return resolved


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
