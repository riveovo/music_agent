"""Curate vocal slices by clustering singer embeddings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from ..audio import write_json
from ..audio_inputs import (
    discover_audio_files,
    prepared_audio_file,
    require_audio_input,
    supported_audio_extensions_text,
)
from ..errors import MusicAgentError
from ..paths import PROJECT_ROOT, ensure_output_dir, slugify, timestamp


DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class Candidate:
    index: int
    original_audio: Path
    staged_audio: Path
    relative_audio: Path
    duration_seconds: float
    conversion: dict[str, object]
    embedding: object | None = None
    cluster: int | None = None
    score: float | None = None
    decision: str = "review"
    reason: str = "pending"
    output_audio: Path | None = None


def curate_vocal_slices(
    audio: str | Path,
    output_dir: str | Path | None = None,
    *,
    recursive: bool = False,
    min_length_ms: int = 3000,
    max_length_ms: int = 10000,
    distance_threshold: float = 0.32,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    model_cache_dir: str | Path | None = None,
    device: str | None = "auto",
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Select the dominant singer cluster from a folder of vocal slices."""
    source = require_audio_input(audio)
    if source.is_file():
        source_dir = source.parent
        files = [source]
        skipped: list[dict[str, str]] = []
    else:
        source_dir = source
        files, skipped = discover_audio_files(source, recursive=recursive)

    if not files:
        raise MusicAgentError(f"No supported vocal slice files found. Supported inputs: {supported_audio_extensions_text()}.")
    _validate_lengths(min_length_ms, max_length_ms)
    if distance_threshold <= 0:
        raise MusicAgentError("distance_threshold must be greater than 0.")

    target_dir = Path(output_dir).expanduser() if output_dir else _default_output_dir(source)
    target_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = target_dir / "accepted"
    rejected_dir = target_dir / "rejected"
    review_dir = target_dir / "review"
    for directory in (accepted_dir, rejected_dir, review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _report(progress, f"Vocal curation: found {len(files)} slice(s)")
    model = _load_embedding_model(embedding_model, model_cache_dir=model_cache_dir, device=device)

    with tempfile.TemporaryDirectory(prefix="music_agent_curate_") as tmp:
        staging_dir = Path(tmp) / "staged"
        candidates = _prepare_candidates(
            files,
            source_dir=source_dir,
            staging_dir=staging_dir,
            output_dir=target_dir,
            recursive=recursive,
            min_length_ms=min_length_ms,
            max_length_ms=max_length_ms,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )
        valid_candidates = [candidate for candidate in candidates if candidate.reason == "valid_for_clustering"]
        if valid_candidates:
            _report(progress, f"Vocal curation: extracting embeddings for {len(valid_candidates)} valid slice(s)")
            for candidate in valid_candidates:
                candidate.embedding = _extract_embedding(model, candidate.staged_audio)
            labels = _cluster_embeddings(
                [candidate.embedding for candidate in valid_candidates],
                distance_threshold=distance_threshold,
            )
            for candidate, label in zip(valid_candidates, labels):
                candidate.cluster = int(label)

            target_cluster, cluster_summaries = _choose_longest_duration_cluster(valid_candidates)
            _score_and_decide(valid_candidates, target_cluster)
        else:
            target_cluster = None
            cluster_summaries = {}

        _copy_outputs(
            candidates,
            accepted_dir=accepted_dir,
            rejected_dir=rejected_dir,
            review_dir=review_dir,
        )

    records = [_candidate_record(candidate) for candidate in candidates]
    counts = _decision_counts(candidates)
    csv_path = _write_cluster_csv(target_dir / "clusters.csv", records)
    result = {
        "capability": "curate_vocal_slices",
        "provider": "speechbrain_ecapa",
        "selection": "longest_total_duration_cluster",
        "input": str(source),
        "output_dir": str(target_dir),
        "recursive": recursive,
        "embedding_model": embedding_model,
        "parameters": {
            "min_length_ms": min_length_ms,
            "max_length_ms": max_length_ms,
            "distance_threshold": distance_threshold,
        },
        "files_found": len(files),
        "files_skipped": len(skipped),
        "target_cluster": target_cluster,
        "clusters": cluster_summaries,
        "decisions": counts,
        "outputs": {
            "accepted": str(accepted_dir),
            "rejected": str(rejected_dir),
            "review": str(review_dir),
            "clusters_csv": str(csv_path),
        },
        "items": records,
        "skipped": skipped,
        "notes": "The accepted set is the singer cluster with the longest total slice duration.",
    }
    result_path = target_dir / "curation.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    _report(progress, "Vocal curation: complete")
    return result


def _prepare_candidates(
    files: list[Path],
    *,
    source_dir: Path,
    staging_dir: Path,
    output_dir: Path,
    recursive: bool,
    min_length_ms: int,
    max_length_ms: int,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    min_seconds = min_length_ms / 1000
    max_seconds = max_length_ms / 1000
    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        rel_parent = rel_path.parent if recursive else Path()
        staged_audio = staging_dir / rel_parent / f"{slugify(audio_path.stem, fallback=f'slice_{index:04d}')}.wav"
        _report(progress, f"Vocal curation: preparing [{index}/{len(files)}] {rel_path}")
        with prepared_audio_file(
            audio_path,
            output_dir=output_dir,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        ) as prepared:
            staged_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prepared.processing_audio, staged_audio)
            duration = _audio_duration_seconds(staged_audio)
            reason = "valid_for_clustering"
            if duration < min_seconds:
                reason = "shorter_than_min_length"
            elif duration > max_seconds:
                reason = "longer_than_max_length"
            candidates.append(
                Candidate(
                    index=index,
                    original_audio=audio_path,
                    staged_audio=staged_audio,
                    relative_audio=rel_parent / staged_audio.name,
                    duration_seconds=duration,
                    conversion=prepared.conversion,
                    reason=reason,
                    decision="review" if reason != "valid_for_clustering" else "pending",
                )
            )
    return candidates


def _score_and_decide(candidates: list[Candidate], target_cluster: int) -> None:
    np = _load_numpy()
    target_embeddings = [candidate.embedding for candidate in candidates if candidate.cluster == target_cluster]
    centroid = _normalize(np.mean(np.stack(target_embeddings), axis=0), np_module=np)
    for candidate in candidates:
        if candidate.embedding is None:
            candidate.decision = "review"
            candidate.reason = candidate.reason or "missing_embedding"
            continue
        candidate.score = float(np.dot(_normalize(candidate.embedding, np_module=np), centroid))
        if candidate.cluster == target_cluster:
            candidate.decision = "accepted"
            candidate.reason = "dominant_longest_duration_cluster"
        else:
            candidate.decision = "rejected"
            candidate.reason = "non_target_cluster"


def _choose_longest_duration_cluster(candidates: list[Candidate]) -> tuple[int, dict[str, dict[str, object]]]:
    summaries: dict[int, dict[str, object]] = {}
    for candidate in candidates:
        if candidate.cluster is None:
            continue
        summary = summaries.setdefault(candidate.cluster, {"count": 0, "total_duration_seconds": 0.0})
        summary["count"] = int(summary["count"]) + 1
        summary["total_duration_seconds"] = float(summary["total_duration_seconds"]) + candidate.duration_seconds
    if not summaries:
        raise MusicAgentError("Could not build clusters from vocal slices.")
    target_cluster = max(
        summaries,
        key=lambda cluster: (float(summaries[cluster]["total_duration_seconds"]), int(summaries[cluster]["count"])),
    )
    result = {
        str(cluster): {
            "count": int(summary["count"]),
            "total_duration_seconds": round(float(summary["total_duration_seconds"]), 6),
            "selected_as_target": cluster == target_cluster,
        }
        for cluster, summary in sorted(summaries.items())
    }
    return target_cluster, result


def _copy_outputs(
    candidates: list[Candidate],
    *,
    accepted_dir: Path,
    rejected_dir: Path,
    review_dir: Path,
) -> None:
    for candidate in candidates:
        if candidate.decision == "accepted":
            base_dir = accepted_dir
        elif candidate.decision == "rejected":
            base_dir = rejected_dir
        else:
            base_dir = review_dir
            if candidate.reason == "valid_for_clustering":
                candidate.reason = "not_clustered"
        destination = _unique_destination(base_dir / candidate.relative_audio)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.staged_audio, destination)
        candidate.output_audio = destination


def _candidate_record(candidate: Candidate) -> dict[str, object]:
    return {
        "index": candidate.index,
        "audio": str(candidate.original_audio),
        "output_audio": str(candidate.output_audio) if candidate.output_audio else None,
        "duration_seconds": round(candidate.duration_seconds, 6),
        "cluster": candidate.cluster,
        "score": round(candidate.score, 6) if candidate.score is not None else None,
        "decision": candidate.decision,
        "reason": candidate.reason,
        "conversion": candidate.conversion,
    }


def _write_cluster_csv(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "audio", "output_audio", "duration_seconds", "cluster", "score", "decision", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    return path


def _decision_counts(candidates: list[Candidate]) -> dict[str, int]:
    counts = {"accepted": 0, "rejected": 0, "review": 0}
    for candidate in candidates:
        counts[candidate.decision] = counts.get(candidate.decision, 0) + 1
    return counts


def _cluster_embeddings(embeddings: list[object], *, distance_threshold: float) -> list[int]:
    np = _load_numpy()
    if len(embeddings) == 1:
        return [0]
    matrix = np.stack([_normalize(embedding, np_module=np) for embedding in embeddings])
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise MusicAgentError(
            "Vocal slice curation requires scikit-learn. Install it with: "
            'python3.12 -m pip install -e ".[vocal-curation]"'
        ) from exc
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    return [int(label) for label in clusterer.fit_predict(matrix)]


def _extract_embedding(model: object, audio_path: Path) -> object:
    np = _load_numpy()
    try:
        signal = model.load_audio(str(audio_path))
        embedding = model.encode_batch(signal)
        if hasattr(embedding, "detach"):
            embedding = embedding.detach()
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu()
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        return _normalize(np.asarray(embedding, dtype="float32").reshape(-1), np_module=np)
    except Exception as exc:
        raise MusicAgentError(f"Could not extract singer embedding for: {audio_path}") from exc


def _load_embedding_model(
    model_name: str,
    *,
    model_cache_dir: str | Path | None,
    device: str | None,
) -> object:
    try:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
    except ImportError as exc:
        raise MusicAgentError(
            "Vocal slice curation requires SpeechBrain. Install it with: "
            'python3.12 -m pip install -e ".[vocal-curation]"'
        ) from exc

    resolved_device = _resolve_device(device)
    cache_dir = Path(model_cache_dir).expanduser() if model_cache_dir else _default_model_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source=model_name,
        savedir=str(cache_dir),
        run_opts={"device": resolved_device},
    )


def _resolve_device(device: str | None) -> str:
    value = (device or "auto").lower()
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise MusicAgentError(
            "Vocal slice curation requires soundfile. Install it with: "
            'python3.12 -m pip install -e ".[vocal-curation]"'
        ) from exc
    try:
        info = sf.info(audio_path)
    except Exception as exc:  # pragma: no cover - backend-specific exception types vary.
        raise MusicAgentError(f"Could not read audio duration: {audio_path}") from exc
    if info.samplerate <= 0:
        raise MusicAgentError(f"Invalid audio sample rate: {audio_path}")
    return float(info.frames) / float(info.samplerate)


def _load_numpy() -> object:
    try:
        import numpy as np
    except ImportError as exc:
        raise MusicAgentError(
            "Vocal slice curation requires numpy. Install it with: "
            'python3.12 -m pip install -e ".[vocal-curation]"'
        ) from exc
    return np


def _normalize(vector: object, *, np_module: object) -> object:
    np = np_module
    array = np.asarray(vector, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return array
    return array / norm


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _validate_lengths(min_length_ms: int, max_length_ms: int) -> None:
    if min_length_ms <= 0:
        raise MusicAgentError("min_length_ms must be greater than 0.")
    if max_length_ms <= 0:
        raise MusicAgentError("max_length_ms must be greater than 0.")
    if max_length_ms < min_length_ms:
        raise MusicAgentError("max_length_ms must be greater than or equal to min_length_ms.")


def _default_output_dir(source: Path) -> Path:
    return ensure_output_dir("curate_vocal_slices") / f"{slugify(source.stem, fallback='vocal_slices')}_{timestamp()}"


def _default_model_cache_dir(model_name: str) -> Path:
    return PROJECT_ROOT / "models" / "speechbrain" / slugify(model_name.replace("/", "_"), fallback="embedding_model")


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
