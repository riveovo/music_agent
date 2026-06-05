"""Vocal/accompaniment separation capability."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from ..audio import require_tool, run_tool, write_json
from ..audio_inputs import (
    batch_item_output_dir,
    default_batch_output_dir,
    discover_audio_files,
    make_batch_result,
    prepared_audio_file,
    require_audio_input,
)
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp
from ..separation import resolve_msst_config, resolve_msst_stage_config, run_msst_stage, separate_with_msst


INSTRUMENTAL_STEMS = ("instrumental", "other", "accompaniment", "no_vocals")
DEHARMONY_STEMS = ("karaoke", "lead", "lead_vocals", "main_vocals", "vocals", "vocal")
DEREVERB_STEMS = ("dry", "noreverb", "no_reverb", "vocals", "vocal")


def separate_stems(
    audio: str | Path,
    output_dir: str | Path | None = None,
    *,
    provider: str = "auto",
    model_type: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | None = None,
    use_tta: bool = False,
    instrumental_model_type: str | None = None,
    instrumental_model_path: str | Path | None = None,
    instrumental_config_path: str | Path | None = None,
    deharmony_model_type: str | None = None,
    deharmony_model_path: str | Path | None = None,
    deharmony_config_path: str | Path | None = None,
    dereverb_model_type: str | None = None,
    dereverb_model_path: str | Path | None = None,
    dereverb_config_path: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Separate vocals and accompaniment using a selected backend."""
    source = require_audio_input(audio)
    target_dir = Path(output_dir).expanduser() if output_dir else (
        default_batch_output_dir("separate_stems", source) if source.is_dir() else _default_output_dir(source)
    )
    if source.is_dir():
        return _separate_directory(
            source,
            target_dir,
            provider=provider,
            model_type=model_type,
            model_path=model_path,
            config_path=config_path,
            device=device,
            use_tta=use_tta,
            instrumental_model_type=instrumental_model_type,
            instrumental_model_path=instrumental_model_path,
            instrumental_config_path=instrumental_config_path,
            deharmony_model_type=deharmony_model_type,
            deharmony_model_path=deharmony_model_path,
            deharmony_config_path=deharmony_config_path,
            dereverb_model_type=dereverb_model_type,
            dereverb_model_path=dereverb_model_path,
            dereverb_config_path=dereverb_config_path,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )

    with prepared_audio_file(
        source,
        output_dir=target_dir,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    ) as prepared:
        return _separate_prepared(
            prepared.original_audio,
            prepared.processing_audio,
            target_dir,
            provider=provider,
            model_type=model_type,
            model_path=model_path,
            config_path=config_path,
            device=device,
            use_tta=use_tta,
            instrumental_model_type=instrumental_model_type,
            instrumental_model_path=instrumental_model_path,
            instrumental_config_path=instrumental_config_path,
            deharmony_model_type=deharmony_model_type,
            deharmony_model_path=deharmony_model_path,
            deharmony_config_path=deharmony_config_path,
            dereverb_model_type=dereverb_model_type,
            dereverb_model_path=dereverb_model_path,
            dereverb_config_path=dereverb_config_path,
            conversion=prepared.conversion,
            progress=progress,
        )


def _separate_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    provider: str,
    model_type: str | None,
    model_path: str | Path | None,
    config_path: str | Path | None,
    device: str | None,
    use_tta: bool,
    instrumental_model_type: str | None,
    instrumental_model_path: str | Path | None,
    instrumental_config_path: str | Path | None,
    deharmony_model_type: str | None,
    deharmony_model_path: str | Path | None,
    deharmony_config_path: str | Path | None,
    dereverb_model_type: str | None,
    dereverb_model_path: str | Path | None,
    dereverb_config_path: str | Path | None,
    recursive: bool,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    files, skipped = discover_audio_files(source_dir, recursive=recursive)
    if not files:
        raise MusicAgentError(f"No supported audio files found in directory: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Separation batch: found {len(files)} file(s)")
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        item_dir = batch_item_output_dir(output_dir, source_dir, audio_path, recursive=recursive, index=index)
        _report(progress, f"Separation batch: [{index}/{len(files)}] {rel_path}")
        try:
            with prepared_audio_file(
                audio_path,
                output_dir=item_dir,
                keep_converted=keep_converted,
                ncm_converter=ncm_converter,
                progress=progress,
            ) as prepared:
                results.append(
                    _separate_prepared(
                        prepared.original_audio,
                        prepared.processing_audio,
                        item_dir,
                        provider=provider,
                        model_type=model_type,
                        model_path=model_path,
                        config_path=config_path,
                        device=device,
                        use_tta=use_tta,
                        instrumental_model_type=instrumental_model_type,
                        instrumental_model_path=instrumental_model_path,
                        instrumental_config_path=instrumental_config_path,
                        deharmony_model_type=deharmony_model_type,
                        deharmony_model_path=deharmony_model_path,
                        deharmony_config_path=deharmony_config_path,
                        dereverb_model_type=dereverb_model_type,
                        dereverb_model_path=dereverb_model_path,
                        dereverb_config_path=dereverb_config_path,
                        conversion=prepared.conversion,
                        progress=progress,
                    )
                )
        except MusicAgentError as exc:
            failures.append({"audio": str(audio_path), "error": str(exc)})
            _report(progress, f"Separation batch: failed {rel_path}: {exc}")

    result = make_batch_result(
        capability="separate_stems",
        input_path=source_dir,
        output_dir=output_dir,
        recursive=recursive,
        results=results,
        failures=failures,
        skipped=skipped,
        extra={"files_found": len(files), "provider": provider},
    )
    _report(progress, "Separation batch: complete")
    return result


def _separate_prepared(
    original_audio: Path,
    processing_audio: Path,
    target_dir: Path,
    *,
    provider: str,
    model_type: str | None,
    model_path: str | Path | None,
    config_path: str | Path | None,
    device: str | None,
    use_tta: bool,
    instrumental_model_type: str | None,
    instrumental_model_path: str | Path | None,
    instrumental_config_path: str | Path | None,
    deharmony_model_type: str | None,
    deharmony_model_path: str | Path | None,
    deharmony_config_path: str | Path | None,
    dereverb_model_type: str | None,
    dereverb_model_path: str | Path | None,
    dereverb_config_path: str | Path | None,
    conversion: dict[str, object],
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    backend = provider.strip().lower()

    if backend == "auto":
        config = resolve_msst_config(
            model_type=model_type,
            model_path=model_path,
            config_path=config_path,
            device=device,
            use_tta=use_tta,
            require_complete=False,
        )
        if config is not None:
            return _separate_msst(
                original_audio,
                processing_audio,
                target_dir,
                config,
                conversion=conversion,
                progress=progress,
                device=device,
                use_tta=use_tta,
                instrumental_model_type=instrumental_model_type,
                instrumental_model_path=instrumental_model_path,
                instrumental_config_path=instrumental_config_path,
                deharmony_model_type=deharmony_model_type,
                deharmony_model_path=deharmony_model_path,
                deharmony_config_path=deharmony_config_path,
                dereverb_model_type=dereverb_model_type,
                dereverb_model_path=dereverb_model_path,
                dereverb_config_path=dereverb_config_path,
            )
        return _separate_heuristic(original_audio, processing_audio, target_dir, conversion=conversion, progress=progress)

    if backend == "msst":
        config = resolve_msst_config(
            model_type=model_type,
            model_path=model_path,
            config_path=config_path,
            device=device,
            use_tta=use_tta,
            require_complete=True,
        )
        if config is None:
            raise MusicAgentError("MSST separation configuration could not be resolved.")
        return _separate_msst(
            original_audio,
            processing_audio,
            target_dir,
            config,
            conversion=conversion,
            progress=progress,
            device=device,
            use_tta=use_tta,
            instrumental_model_type=instrumental_model_type,
            instrumental_model_path=instrumental_model_path,
            instrumental_config_path=instrumental_config_path,
            deharmony_model_type=deharmony_model_type,
            deharmony_model_path=deharmony_model_path,
            deharmony_config_path=deharmony_config_path,
            dereverb_model_type=dereverb_model_type,
            dereverb_model_path=dereverb_model_path,
            dereverb_config_path=dereverb_config_path,
        )

    if backend == "heuristic":
        return _separate_heuristic(original_audio, processing_audio, target_dir, conversion=conversion, progress=progress)

    raise MusicAgentError("Unknown separation provider. Available providers: auto, heuristic, msst.")


def _separate_heuristic(
    original_audio: Path,
    processing_audio: Path,
    target_dir: Path,
    *,
    conversion: dict[str, object],
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Create lightweight vocal and accompaniment approximations."""
    _report(progress, "Heuristic separation: checking ffmpeg")
    require_tool("ffmpeg")
    target_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = target_dir / "vocals.wav"
    accompaniment_path = target_dir / "accompaniment.wav"

    _report(progress, "Heuristic separation: extracting vocal approximation")
    run_tool(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(processing_audio),
            "-af",
            "highpass=f=120,lowpass=f=3600,acompressor=threshold=-18dB:ratio=2:attack=20:release=250",
            str(vocals_path),
        ]
    )
    _report(progress, "Heuristic separation: extracting accompaniment approximation")
    run_tool(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostats",
            "-i",
            str(processing_audio),
            "-af",
            "equalizer=f=1200:width_type=o:width=2:g=-10,lowpass=f=12000,highpass=f=40",
            str(accompaniment_path),
        ]
    )

    result = {
        "capability": "separate_stems",
        "provider": "heuristic",
        "quality": "heuristic_mvp",
        "audio": str(original_audio),
        "output_dir": str(target_dir),
        "stems": {
            "vocals": str(vocals_path),
            "accompaniment": str(accompaniment_path),
        },
        "conversion": conversion,
        "notes": "This is a lightweight ffmpeg-filter approximation, not model-grade source separation.",
    }
    result_path = target_dir / "separation.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    _report(progress, "Heuristic separation: complete")
    return result


def _separate_msst(
    original_audio: Path,
    processing_audio: Path,
    target_dir: Path,
    config: object,
    *,
    conversion: dict[str, object],
    progress: Callable[[str], None] | None = None,
    device: str | None = None,
    use_tta: bool = False,
    instrumental_model_type: str | None = None,
    instrumental_model_path: str | Path | None = None,
    instrumental_config_path: str | Path | None = None,
    deharmony_model_type: str | None = None,
    deharmony_model_path: str | Path | None = None,
    deharmony_config_path: str | Path | None = None,
    dereverb_model_type: str | None = None,
    dereverb_model_path: str | Path | None = None,
    dereverb_config_path: str | Path | None = None,
) -> dict[str, object]:
    output = separate_with_msst(processing_audio, target_dir, config, progress=progress)
    postprocess: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}

    instrumental_config = resolve_msst_stage_config(
        "instrumental",
        model_type=instrumental_model_type,
        model_path=instrumental_model_path,
        config_path=instrumental_config_path,
        device=device,
        use_tta=use_tta,
    )
    if instrumental_config is not None:
        stage = run_msst_stage(
            processing_audio,
            output.accompaniment,
            instrumental_config,
            preferred_stems=INSTRUMENTAL_STEMS,
            stage_name="instrumental refinement",
            progress=progress,
        )
        postprocess.append(_stage_result("instrumental_refinement", stage))

    current_vocals = output.vocals
    deharmony_config = resolve_msst_stage_config(
        "deharmony",
        model_type=deharmony_model_type,
        model_path=deharmony_model_path,
        config_path=deharmony_config_path,
        device=device,
        use_tta=use_tta,
    )
    if deharmony_config is not None:
        artifacts["primary_vocals"] = str(_copy_if_missing(output.vocals, target_dir / "vocals_primary.wav"))
        stage_path = target_dir / "vocals_deharmonized.wav"
        stage = run_msst_stage(
            current_vocals,
            stage_path,
            deharmony_config,
            preferred_stems=DEHARMONY_STEMS,
            stage_name="deharmony",
            progress=progress,
        )
        current_vocals = stage.output_audio
        artifacts["deharmonized_vocals"] = str(current_vocals)
        postprocess.append(_stage_result("deharmony", stage))

    dereverb_config = resolve_msst_stage_config(
        "dereverb",
        model_type=dereverb_model_type,
        model_path=dereverb_model_path,
        config_path=dereverb_config_path,
        device=device,
        use_tta=use_tta,
    )
    if dereverb_config is not None:
        if "primary_vocals" not in artifacts:
            artifacts["primary_vocals"] = str(_copy_if_missing(output.vocals, target_dir / "vocals_primary.wav"))
        stage_path = target_dir / "vocals_dereverbed.wav"
        stage = run_msst_stage(
            current_vocals,
            stage_path,
            dereverb_config,
            preferred_stems=DEREVERB_STEMS,
            stage_name="dereverb",
            progress=progress,
        )
        current_vocals = stage.output_audio
        artifacts["dereverbed_vocals"] = str(current_vocals)
        postprocess.append(_stage_result("dereverb", stage))

    if current_vocals != output.vocals:
        shutil.copy2(current_vocals, output.vocals)

    result = {
        "capability": "separate_stems",
        "provider": "msst",
        "quality": "msst_roformer",
        "audio": str(original_audio),
        "output_dir": str(target_dir),
        "model_type": output.model_type,
        "model_path": str(output.model_path),
        "config_path": str(output.config_path),
        "device": output.device,
        "sample_rate": output.sample_rate,
        "source_stems": sorted(output.source_stems),
        "stems": {
            "vocals": str(output.vocals),
            "accompaniment": str(output.accompaniment),
        },
        "postprocess": postprocess,
        "artifacts": artifacts,
        "conversion": conversion,
        "notes": "Separated locally with the in-project MSST-compatible RoFormer backend.",
    }
    result_path = target_dir / "separation.json"
    write_json(result_path, result | {"result_json": str(result_path)})
    result["result_json"] = str(result_path)
    _report(progress, "Separation: wrote result JSON")
    return result


def _default_output_dir(audio_path: Path) -> Path:
    return ensure_output_dir("separate_stems") / f"{slugify(audio_path.stem)}_{timestamp()}"


def _stage_result(stage_name: str, stage: object) -> dict[str, object]:
    return {
        "stage": stage_name,
        "model_type": stage.model_type,
        "model_path": str(stage.model_path),
        "config_path": str(stage.config_path),
        "device": stage.device,
        "selected_stem": stage.selected_stem,
        "source_stems": sorted(stage.source_stems),
        "output_audio": str(stage.output_audio),
    }


def _copy_if_missing(source: Path, destination: Path) -> Path:
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
