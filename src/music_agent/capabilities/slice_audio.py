"""Audio slicing and batch conversion capability.

The slicer follows the RMS-based silence detection approach used by
openvpi/audio-slicer, adapted for this project's capability API and batch
workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..audio import write_json
from ..audio_inputs import (
    batch_item_output_dir,
    discover_audio_files,
    make_batch_result,
    prepared_audio_file,
    require_audio_input,
    supported_audio_extensions_text,
)
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp


@dataclass(frozen=True)
class SliceParameters:
    min_length_ms: int = 3000
    max_length_ms: int = 10000
    hop_size_ms: int = 10
    max_sil_kept_ms: int = 500
    db_threshold: float | None = None
    min_interval_ms: int | None = None

    def validate(self) -> None:
        if self.min_length_ms <= 0:
            raise MusicAgentError("min_length_ms must be greater than 0.")
        if self.max_length_ms <= 0:
            raise MusicAgentError("max_length_ms must be greater than 0.")
        if self.max_length_ms < self.min_length_ms:
            raise MusicAgentError("max_length_ms must be greater than or equal to min_length_ms.")
        if self.hop_size_ms <= 0:
            raise MusicAgentError("hop_size_ms must be greater than 0.")
        if self.max_sil_kept_ms < 0:
            raise MusicAgentError("max_sil_kept_ms must be greater than or equal to 0.")
        if self.min_interval_ms is not None and self.min_interval_ms <= 0:
            raise MusicAgentError("min_interval_ms must be greater than 0.")
        if self.min_interval_ms is not None and self.min_interval_ms < self.hop_size_ms:
            raise MusicAgentError("Audio slicing requires min_interval_ms >= hop_size_ms.")
        if self.max_sil_kept_ms and self.max_sil_kept_ms < self.hop_size_ms:
            raise MusicAgentError("Audio slicing requires max_sil_kept_ms >= hop_size_ms when max_sil_kept_ms is non-zero.")

    def as_dict(self) -> dict[str, object]:
        return {
            "min_length_ms": self.min_length_ms,
            "max_length_ms": self.max_length_ms,
        }


@dataclass(frozen=True)
class ResolvedSliceParameters:
    min_length_ms: int
    max_length_ms: int
    db_threshold: float
    min_interval_ms: int
    hop_size_ms: int
    max_sil_kept_ms: int

    def as_dict(self) -> dict[str, object]:
        return {
            "min_length_ms": self.min_length_ms,
            "max_length_ms": self.max_length_ms,
            "db_threshold": round(self.db_threshold, 3),
            "min_interval_ms": self.min_interval_ms,
            "hop_size_ms": self.hop_size_ms,
            "max_sil_kept_ms": self.max_sil_kept_ms,
        }


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sample: int
    end_sample: int
    sample_rate: int
    audio: object

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return max(0, self.end_sample - self.start_sample) / self.sample_rate


def slice_audio(
    audio: str | Path,
    output_dir: str | Path | None = None,
    *,
    recursive: bool = False,
    min_length_ms: int = 3000,
    max_length_ms: int = 10000,
    hop_size_ms: int = 10,
    max_sil_kept_ms: int = 500,
    db_threshold: float | None = None,
    min_interval_ms: int | None = None,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Slice one audio file or every supported audio file in a directory."""
    source = require_audio_input(audio)

    params = SliceParameters(
        min_length_ms=min_length_ms,
        max_length_ms=max_length_ms,
        hop_size_ms=hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
        db_threshold=db_threshold,
        min_interval_ms=min_interval_ms,
    )
    params.validate()

    base_output_dir = Path(output_dir).expanduser() if output_dir else _default_output_dir(source)
    if source.is_dir():
        return _slice_directory(
            source,
            base_output_dir,
            recursive=recursive,
            params=params,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )

    return _slice_single_file(
        source,
        base_output_dir,
        params=params,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _slice_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    recursive: bool,
    params: SliceParameters,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    files, skipped = discover_audio_files(source_dir, recursive=recursive)
    if not files:
        raise MusicAgentError(f"No supported audio files found in directory: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Audio slicing batch: found {len(files)} file(s)")
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        file_output_dir = batch_item_output_dir(output_dir, source_dir, audio_path, recursive=recursive, index=index)
        _report(progress, f"Audio slicing batch: [{index}/{len(files)}] {rel_path}")
        try:
            results.append(
                _slice_single_file(
                    audio_path,
                    file_output_dir,
                    params=params,
                    keep_converted=keep_converted,
                    ncm_converter=ncm_converter,
                    progress=progress,
                    write_result_json=False,
                )
            )
        except MusicAgentError as exc:
            failures.append({"audio": str(audio_path), "error": str(exc)})
            _report(progress, f"Audio slicing batch: failed {rel_path}: {exc}")

    result = make_batch_result(
        capability="slice_audio",
        input_path=source_dir,
        output_dir=output_dir,
        recursive=recursive,
        results=results,
        failures=failures,
        skipped=skipped,
        extra={"parameters": params.as_dict(), "files_found": len(files)},
    )
    _report(progress, "Audio slicing batch: complete")
    return result


def _slice_single_file(
    audio_path: Path,
    output_dir: Path,
    *,
    params: SliceParameters,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
    write_result_json: bool = True,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Audio slicing: preparing {audio_path.name}")

    with prepared_audio_file(
        audio_path,
        output_dir=output_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    ) as prepared:
        chunks, sample_rate, channels, resolved_params = _slice_wav(
            prepared.processing_audio,
            params=params,
            progress=progress,
        )
        chunk_results = _write_chunks(
            prepared.original_audio,
            output_dir=output_dir,
            chunks=chunks,
            sample_rate=sample_rate,
            progress=progress,
        )
        conversion = prepared.conversion

    result = {
        "capability": "slice_audio",
        "mode": "single",
        "input": str(audio_path),
        "source_format": audio_path.suffix.lower().lstrip("."),
        "output_dir": str(output_dir),
        "sample_rate": sample_rate,
        "channels": channels,
        "chunk_count": len(chunk_results),
        "chunks": chunk_results,
        "conversion": conversion,
        "parameters": params.as_dict(),
        "effective_parameters": resolved_params.as_dict(),
        "notes": f"Sliced locally with RMS-based silence detection. Supported inputs: {supported_audio_extensions_text()}.",
    }
    if write_result_json:
        result_path = output_dir / "slice_audio.json"
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Audio slicing: wrote result JSON")
    return result


def _slice_wav(
    wav_path: Path,
    *,
    params: SliceParameters,
    progress: Callable[[str], None] | None,
) -> tuple[list[AudioChunk], int, int, ResolvedSliceParameters]:
    np, sf = _load_audio_libraries()
    _report(progress, "Audio slicing: loading WAV")
    try:
        waveform, sample_rate = sf.read(wav_path, always_2d=True, dtype="float32")
    except Exception as exc:  # pragma: no cover - soundfile exception hierarchy varies.
        raise MusicAgentError(f"Could not read WAV audio for slicing: {wav_path}") from exc

    if waveform.shape[0] == 0:
        raise MusicAgentError(f"Audio file is empty: {wav_path}")

    channels = int(waveform.shape[1])
    _report(progress, "Audio slicing: estimating silence parameters")
    resolved_params = _resolve_slice_parameters(waveform, sample_rate=sample_rate, params=params, np_module=np)
    _report(
        progress,
        "Audio slicing: detecting silent regions "
        f"(threshold={resolved_params.db_threshold:.1f}dB, interval={resolved_params.min_interval_ms}ms)",
    )
    boundaries = _detect_slice_boundaries(
        waveform,
        sample_rate=sample_rate,
        params=resolved_params,
        np_module=np,
    )
    chunks = [
        AudioChunk(
            index=index,
            start_sample=start,
            end_sample=end,
            sample_rate=sample_rate,
            audio=waveform[start:end],
        )
        for index, (start, end) in enumerate(boundaries, start=1)
        if end > start
    ]
    if not chunks:
        chunks = [
            AudioChunk(
                index=1,
                start_sample=0,
                end_sample=int(waveform.shape[0]),
                sample_rate=sample_rate,
                audio=waveform,
            )
        ]
    return chunks, int(sample_rate), channels, resolved_params


def _resolve_slice_parameters(
    waveform: object,
    *,
    sample_rate: int,
    params: SliceParameters,
    np_module: object,
) -> ResolvedSliceParameters:
    np = np_module
    samples = waveform.mean(axis=1) if len(waveform.shape) > 1 else waveform
    hop_samples = max(1, round(sample_rate * params.hop_size_ms / 1000))
    frame_length = max(1, round(sample_rate * 0.05))
    rms_list = _frame_rms(samples, frame_length=frame_length, hop_length=hop_samples, np_module=np)
    threshold = params.db_threshold
    if threshold is None:
        threshold = _estimate_db_threshold(rms_list, np_module=np)
    min_interval = params.min_interval_ms
    if min_interval is None:
        min_interval = _estimate_min_interval_ms(
            rms_list,
            db_threshold=threshold,
            min_length_ms=params.min_length_ms,
            max_length_ms=params.max_length_ms,
            hop_size_ms=params.hop_size_ms,
            np_module=np,
        )
    min_interval = max(params.hop_size_ms, min(min_interval, params.min_length_ms))
    max_sil_kept_ms = min(params.max_sil_kept_ms, max(params.hop_size_ms, params.min_length_ms // 2))
    return ResolvedSliceParameters(
        min_length_ms=params.min_length_ms,
        max_length_ms=params.max_length_ms,
        db_threshold=float(threshold),
        min_interval_ms=int(min_interval),
        hop_size_ms=params.hop_size_ms,
        max_sil_kept_ms=max_sil_kept_ms,
    )


def _estimate_db_threshold(rms_list: object, *, np_module: object) -> float:
    np = np_module
    if rms_list.shape[0] == 0:
        return -40.0
    db = 20.0 * np.log10(np.maximum(rms_list, 1e-8))
    p10, p80 = np.percentile(db, [10, 80])
    dynamic_range = float(p80 - p10)
    if dynamic_range < 12.0:
        return float(max(-60.0, min(-30.0, p10 - 6.0)))
    return float(max(-60.0, min(-30.0, p10 + dynamic_range * 0.35)))


def _estimate_min_interval_ms(
    rms_list: object,
    *,
    db_threshold: float,
    min_length_ms: int,
    max_length_ms: int,
    hop_size_ms: int,
    np_module: object,
) -> int:
    np = np_module
    threshold = 10 ** (db_threshold / 20.0)
    silence_runs_ms: list[int] = []
    start: int | None = None
    for index, rms in enumerate(rms_list):
        if rms < threshold:
            if start is None:
                start = index
            continue
        if start is not None:
            duration_ms = (index - start) * hop_size_ms
            if duration_ms >= 80:
                silence_runs_ms.append(duration_ms)
            start = None
    if start is not None:
        duration_ms = (len(rms_list) - start) * hop_size_ms
        if duration_ms >= 80:
            silence_runs_ms.append(duration_ms)

    lower = 120
    upper = max(lower, min(700, max_length_ms // 8))
    fallback = max(lower, min(upper, round(min_length_ms * 0.08)))
    if not silence_runs_ms:
        return fallback
    estimated = int(round(float(np.percentile(silence_runs_ms, 35))))
    return max(lower, min(upper, estimated))


def _detect_slice_boundaries(
    waveform: object,
    *,
    sample_rate: int,
    params: ResolvedSliceParameters,
    np_module: object,
) -> list[tuple[int, int]]:
    np = np_module
    samples = waveform.mean(axis=1) if len(waveform.shape) > 1 else waveform
    hop_samples = max(1, round(sample_rate * params.hop_size_ms / 1000))
    min_interval_samples = max(1, round(sample_rate * params.min_interval_ms / 1000))
    win_size = max(1, min(min_interval_samples, 4 * hop_samples))
    min_length_frames = max(1, round(sample_rate * params.min_length_ms / 1000 / hop_samples))
    min_interval_frames = max(1, round(min_interval_samples / hop_samples))
    max_sil_kept_frames = max(0, round(sample_rate * params.max_sil_kept_ms / 1000 / hop_samples))

    total_samples = int(samples.shape[0])
    total_hop_frames = (total_samples + hop_samples - 1) // hop_samples
    if total_hop_frames <= min_length_frames:
        return _enforce_length_bounds([(0, total_samples)], rms_list=None, sample_rate=sample_rate, params=params, np_module=np)

    rms_list = _frame_rms(samples, frame_length=win_size, hop_length=hop_samples, np_module=np)
    if rms_list.shape[0] == 0:
        return [(0, total_samples)]

    threshold = 10 ** (params.db_threshold / 20.0)
    silence_tags: list[tuple[int, int]] = []
    silence_start: int | None = None
    clip_start = 0

    for index, rms in enumerate(rms_list):
        if rms < threshold:
            if silence_start is None:
                silence_start = index
            continue

        if silence_start is None:
            continue

        is_leading_silence = silence_start == 0 and index > max_sil_kept_frames
        can_slice_middle = index - silence_start >= min_interval_frames and index - clip_start >= min_length_frames
        if not is_leading_silence and not can_slice_middle:
            silence_start = None
            continue

        silence_tags.append(
            _choose_silence_cut(
                rms_list,
                silence_start=silence_start,
                silence_end=index,
                max_sil_kept_frames=max_sil_kept_frames,
                leading=silence_start == 0,
            )
        )
        clip_start = silence_tags[-1][1]
        silence_start = None

    total_frames = int(rms_list.shape[0])
    if silence_start is not None and total_frames - silence_start >= min_interval_frames:
        silence_end = min(total_frames, silence_start + max(1, max_sil_kept_frames))
        pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
        silence_tags.append((pos, total_frames + 1))

    if not silence_tags:
        return _enforce_length_bounds(
            [(0, total_samples)],
            rms_list=rms_list,
            sample_rate=sample_rate,
            params=params,
            np_module=np,
        )

    boundaries: list[tuple[int, int]] = []
    if silence_tags[0][0] > 0:
        boundaries.append((0, min(total_samples, silence_tags[0][0] * hop_samples)))
    for left, right in zip(silence_tags, silence_tags[1:]):
        boundaries.append(
            (
                min(total_samples, left[1] * hop_samples),
                min(total_samples, right[0] * hop_samples),
            )
        )
    if silence_tags[-1][1] < total_frames:
        boundaries.append((min(total_samples, silence_tags[-1][1] * hop_samples), total_samples))

    boundaries = [(start, end) for start, end in boundaries if end > start]
    return _enforce_length_bounds(
        boundaries,
        rms_list=rms_list,
        sample_rate=sample_rate,
        params=params,
        np_module=np,
    )


def _enforce_length_bounds(
    boundaries: list[tuple[int, int]],
    *,
    rms_list: object | None,
    sample_rate: int,
    params: ResolvedSliceParameters,
    np_module: object,
) -> list[tuple[int, int]]:
    if not boundaries:
        return []
    min_samples = max(1, round(sample_rate * params.min_length_ms / 1000))
    max_samples = max(min_samples, round(sample_rate * params.max_length_ms / 1000))
    hop_samples = max(1, round(sample_rate * params.hop_size_ms / 1000))

    split: list[tuple[int, int]] = []
    for start, end in boundaries:
        split.extend(
            _split_long_boundary(
                start,
                end,
                min_samples=min_samples,
                max_samples=max_samples,
                hop_samples=hop_samples,
                rms_list=rms_list,
                np_module=np_module,
            )
        )
    return _merge_short_boundaries(split, min_samples=min_samples, max_samples=max_samples)


def _split_long_boundary(
    start: int,
    end: int,
    *,
    min_samples: int,
    max_samples: int,
    hop_samples: int,
    rms_list: object | None,
    np_module: object,
) -> list[tuple[int, int]]:
    if end <= start:
        return []
    length = end - start
    if length <= max_samples:
        return [(start, end)]

    segment_count = (length + max_samples - 1) // max_samples
    while segment_count > 1 and length < segment_count * min_samples:
        segment_count -= 1
    if segment_count <= 1:
        return [(start, end)]

    segments: list[tuple[int, int]] = []
    cursor = start
    target_length = length / segment_count
    search_radius = max(hop_samples, round((max_samples - min_samples) * 0.35))
    for segment_index in range(1, segment_count):
        ideal = round(start + target_length * segment_index)
        remaining_segments = segment_count - segment_index
        low = max(cursor + min_samples, end - remaining_segments * max_samples, ideal - search_radius)
        high = min(cursor + max_samples, end - remaining_segments * min_samples, ideal + search_radius)
        if high <= low:
            cut = min(cursor + max_samples, end - remaining_segments * min_samples)
        else:
            cut = _choose_quietest_cut(
                low,
                high,
                hop_samples=hop_samples,
                rms_list=rms_list,
                np_module=np_module,
            )
        if cut <= cursor:
            cut = min(cursor + max_samples, end)
        segments.append((cursor, cut))
        cursor = cut
    segments.append((cursor, end))
    return segments


def _choose_quietest_cut(
    low_sample: int,
    high_sample: int,
    *,
    hop_samples: int,
    rms_list: object | None,
    np_module: object,
) -> int:
    if rms_list is None or rms_list.shape[0] == 0:
        return (low_sample + high_sample) // 2
    np = np_module
    low_frame = max(0, low_sample // hop_samples)
    high_frame = min(int(rms_list.shape[0]) - 1, max(low_frame, high_sample // hop_samples))
    if high_frame <= low_frame:
        return min(high_sample, max(low_sample, low_frame * hop_samples))
    local = rms_list[low_frame : high_frame + 1]
    frame = int(np.argmin(local)) + low_frame
    return min(high_sample, max(low_sample, frame * hop_samples))


def _merge_short_boundaries(
    boundaries: list[tuple[int, int]],
    *,
    min_samples: int,
    max_samples: int,
) -> list[tuple[int, int]]:
    items = [(start, end) for start, end in boundaries if end > start]
    index = 0
    while index < len(items):
        start, end = items[index]
        if end - start >= min_samples or len(items) == 1:
            index += 1
            continue

        if index > 0 and end - items[index - 1][0] <= max_samples:
            items[index - 1] = (items[index - 1][0], end)
            del items[index]
            index = max(0, index - 1)
            continue
        if index + 1 < len(items) and items[index + 1][1] - start <= max_samples:
            items[index] = (start, items[index + 1][1])
            del items[index + 1]
            continue
        index += 1
    return items


def _choose_silence_cut(
    rms_list: object,
    *,
    silence_start: int,
    silence_end: int,
    max_sil_kept_frames: int,
    leading: bool,
) -> tuple[int, int]:
    if max_sil_kept_frames <= 0:
        pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
        return (0, pos) if leading else (pos, pos)

    silence_length = silence_end - silence_start
    if silence_length <= max_sil_kept_frames:
        pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
        return (0, pos) if leading else (pos, pos)

    if silence_length <= max_sil_kept_frames * 2:
        middle_start = silence_end - max_sil_kept_frames
        middle_end = silence_start + max_sil_kept_frames + 1
        pos = int(rms_list[middle_start:middle_end].argmin()) + middle_start
        pos_l = int(rms_list[silence_start : silence_start + max_sil_kept_frames + 1].argmin()) + silence_start
        pos_r = int(rms_list[silence_end - max_sil_kept_frames : silence_end + 1].argmin()) + silence_end - max_sil_kept_frames
        if leading:
            return (0, pos_r)
        return (min(pos_l, pos), max(pos_r, pos))

    pos_l = int(rms_list[silence_start : silence_start + max_sil_kept_frames + 1].argmin()) + silence_start
    pos_r = int(rms_list[silence_end - max_sil_kept_frames : silence_end + 1].argmin()) + silence_end - max_sil_kept_frames
    return (0, pos_r) if leading else (pos_l, pos_r)


def _frame_rms(samples: object, *, frame_length: int, hop_length: int, np_module: object) -> object:
    np = np_module
    padding = (frame_length // 2, frame_length // 2)
    padded = np.pad(samples, padding, mode="constant")
    if padded.shape[0] < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.shape[0]), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop_length]
    power = np.mean(np.abs(windows) ** 2, axis=-1)
    return np.sqrt(power)


def _write_chunks(
    source_audio: Path,
    output_dir: Path,
    chunks: list[AudioChunk],
    *,
    sample_rate: int,
    progress: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    _, sf = _load_audio_libraries()
    _report(progress, f"Audio slicing: writing {len(chunks)} chunk(s)")
    results: list[dict[str, object]] = []
    base = slugify(source_audio.stem)
    for chunk in chunks:
        chunk_path = output_dir / f"{base}_slice_{chunk.index:04d}.wav"
        sf.write(chunk_path, chunk.audio, sample_rate)
        results.append(
            {
                "index": chunk.index,
                "audio": str(chunk_path),
                "start_seconds": round(chunk.start_seconds, 6),
                "end_seconds": round(chunk.end_seconds, 6),
                "duration_seconds": round(chunk.duration_seconds, 6),
                "start_sample": chunk.start_sample,
                "end_sample": chunk.end_sample,
            }
        )
    return results


def _load_audio_libraries() -> tuple[object, object]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise MusicAgentError(
            "Audio slicing requires optional dependencies. Install them with: "
            'python3.12 -m pip install -e ".[audio-slice]"'
        ) from exc
    return np, sf


def _default_output_dir(source: Path) -> Path:
    return ensure_output_dir("slice_audio") / f"{slugify(source.stem, fallback='audio')}_{timestamp()}"


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
