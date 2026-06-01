"""Voice conversion capability with placeholder and SVCFusion providers."""

from __future__ import annotations

from pathlib import Path
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
from ..voice_conversion import (
    SVCFusionConfig,
    convert_with_svcfusion,
    has_complete_svcfusion_env,
    resolve_svcfusion_config,
)


PRESETS = {
    "bright": "asetrate=44100*1.06,aresample=44100,atempo=0.943,highpass=f=120,equalizer=f=3200:width_type=o:width=1:g=4",
    "deep": "asetrate=44100*0.92,aresample=44100,atempo=1.087,lowpass=f=7200,equalizer=f=180:width_type=o:width=1:g=4",
    "robot": "aresample=8000,aresample=44100,aecho=0.8:0.8:35:0.35,acrusher=level_in=1:level_out=0.8:bits=8:mode=log",
}
VOICE_CONVERSION_PROVIDERS = ("auto", "placeholder", "svcfusion")


def convert_voice(
    audio: str | Path,
    preset: str = "bright",
    output: str | Path | None = None,
    *,
    provider: str = "auto",
    output_dir: str | Path | None = None,
    recursive: bool = False,
    keep_converted: bool = False,
    ncm_converter: str | None = None,
    svcfusion_model_type: str | None = None,
    svcfusion_model_path: str | Path | None = None,
    svcfusion_config_path: str | Path | None = None,
    svcfusion_speaker: str | None = None,
    svcfusion_device: str | None = None,
    svcfusion_source_path: str | Path | None = None,
    svcfusion_f0_method: str = "rmvpe",
    svcfusion_key_change: float = 0.0,
    svcfusion_formant_shift_key: float = 0.0,
    svcfusion_method: str = "auto",
    svcfusion_threshold: float = -60.0,
    svcfusion_infer_step: str = "auto",
    svcfusion_t_start: str = "auto",
    svcfusion_vocal_register_factor: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Convert voice with either the placeholder preset or SVCFusion."""
    explicit_svcfusion_args = any(
        value is not None
        for value in (
            svcfusion_model_type,
            svcfusion_model_path,
            svcfusion_config_path,
            svcfusion_speaker,
            svcfusion_device,
            svcfusion_source_path,
        )
    )
    resolved_provider = _resolve_provider(provider, explicit_svcfusion_args=explicit_svcfusion_args)
    svcfusion_config: SVCFusionConfig | None = None
    if resolved_provider == "placeholder":
        _validate_preset(preset)
    else:
        svcfusion_config = resolve_svcfusion_config(
            model_type=svcfusion_model_type,
            model_path=svcfusion_model_path,
            config_path=svcfusion_config_path,
            speaker=svcfusion_speaker,
            device=svcfusion_device,
            source_path=svcfusion_source_path,
            f0_method=svcfusion_f0_method,
            key_change=svcfusion_key_change,
            formant_shift_key=svcfusion_formant_shift_key,
            method=svcfusion_method,
            threshold=svcfusion_threshold,
            infer_step=svcfusion_infer_step,
            t_start=svcfusion_t_start,
            vocal_register_factor=svcfusion_vocal_register_factor,
        )

    source = require_audio_input(audio)
    if source.is_dir():
        if output is not None:
            raise MusicAgentError("Batch voice conversion uses --output-dir; --output is only for single-file output.")
        target_dir = Path(output_dir).expanduser() if output_dir else default_batch_output_dir("convert_voice", source)
        return _convert_voice_directory(
            source,
            target_dir,
            provider=resolved_provider,
            preset=preset,
            svcfusion_config=svcfusion_config,
            recursive=recursive,
            keep_converted=keep_converted,
            ncm_converter=ncm_converter,
            progress=progress,
        )

    target_dir = Path(output_dir).expanduser() if output_dir else ensure_output_dir("convert_voice")
    output_path = Path(output).expanduser() if output else _default_output_path(
        source,
        preset,
        provider=resolved_provider,
        output_dir=target_dir,
    )
    return _convert_voice_single(
        source,
        provider=resolved_provider,
        preset=preset,
        svcfusion_config=svcfusion_config,
        output_path=output_path,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    )


def _convert_voice_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    provider: str,
    preset: str,
    svcfusion_config: SVCFusionConfig | None,
    recursive: bool,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
) -> dict[str, object]:
    files, skipped = discover_audio_files(source_dir, recursive=recursive)
    if not files:
        raise MusicAgentError(f"No supported audio files found in directory: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _report(progress, f"Voice conversion batch: found {len(files)} file(s)")
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, audio_path in enumerate(files, start=1):
        rel_path = audio_path.relative_to(source_dir)
        item_dir = batch_item_output_dir(output_dir, source_dir, audio_path, recursive=recursive, index=index)
        output_path = _default_output_path(audio_path, preset, provider=provider, output_dir=item_dir)
        _report(progress, f"Voice conversion batch: [{index}/{len(files)}] {rel_path}")
        try:
            results.append(
                _convert_voice_single(
                    audio_path,
                    provider=provider,
                    preset=preset,
                    svcfusion_config=svcfusion_config,
                    output_path=output_path,
                    keep_converted=keep_converted,
                    ncm_converter=ncm_converter,
                    progress=progress,
                    write_result_json=False,
                )
            )
        except MusicAgentError as exc:
            failures.append({"audio": str(audio_path), "error": str(exc)})
            _report(progress, f"Voice conversion batch: failed {rel_path}: {exc}")

    result = make_batch_result(
        capability="convert_voice",
        input_path=source_dir,
        output_dir=output_dir,
        recursive=recursive,
        results=results,
        failures=failures,
        skipped=skipped,
        extra={"files_found": len(files), "provider": provider, "preset": preset},
    )
    _report(progress, "Voice conversion batch: complete")
    return result


def _convert_voice_single(
    audio_path: Path,
    *,
    provider: str,
    preset: str,
    svcfusion_config: SVCFusionConfig | None,
    output_path: Path,
    keep_converted: bool,
    ncm_converter: str | None,
    progress: Callable[[str], None] | None,
    write_result_json: bool = True,
) -> dict[str, object]:
    _report(progress, f"Voice conversion: preparing {audio_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared_audio_file(
        audio_path,
        output_dir=output_path.parent,
        keep_converted=keep_converted,
        ncm_converter=ncm_converter,
        progress=progress,
    ) as prepared:
        if provider == "placeholder":
            require_tool("ffmpeg")
            _report(progress, f"Voice conversion: applying '{preset}' preset")
            run_tool(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-nostats",
                    "-i",
                    str(prepared.processing_audio),
                    "-af",
                    PRESETS[preset],
                    str(output_path),
                ]
            )
            quality = "placeholder_mvp"
            provider_metadata: dict[str, object] = {}
            notes = "This is a lightweight ffmpeg transform, not neural voice conversion."
        else:
            if svcfusion_config is None:
                raise MusicAgentError("Internal error: missing SVCFusion config.")
            svcfusion_output = convert_with_svcfusion(
                prepared.processing_audio,
                output_path,
                svcfusion_config,
                progress=progress,
            )
            quality = f"svcfusion_{svcfusion_output.model_type}"
            provider_metadata = {
                "svcfusion": {
                    "model_type": svcfusion_output.model_type,
                    "model_path": str(svcfusion_output.model_path),
                    "config_path": str(svcfusion_output.config_path) if svcfusion_output.config_path else None,
                    "speaker": svcfusion_output.speaker,
                    "device": svcfusion_output.device,
                    "source_path": str(svcfusion_output.source_path) if svcfusion_output.source_path else None,
                    "parameters": svcfusion_output.parameters,
                }
            }
            notes = "Converted with an SVCFusion-compatible backend."

    result = {
        "capability": "convert_voice",
        "provider": provider,
        "quality": quality,
        "audio": str(audio_path),
        "preset": preset,
        "output_audio": str(output_path),
        "conversion": prepared.conversion,
        "notes": notes,
    } | provider_metadata
    if write_result_json:
        result_path = output_path.with_suffix(".json")
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        _report(progress, "Voice conversion: wrote result JSON")
    return result


def _default_output_path(
    audio_path: Path,
    preset: str,
    *,
    provider: str,
    output_dir: Path | None = None,
) -> Path:
    tag = preset if provider == "placeholder" else "svcfusion"
    name = f"voice_{tag}_{slugify(audio_path.stem)}_{timestamp()}.wav"
    return (output_dir or ensure_output_dir("convert_voice")) / name


def _resolve_provider(provider: str, *, explicit_svcfusion_args: bool = False) -> str:
    if provider not in VOICE_CONVERSION_PROVIDERS:
        available = ", ".join(VOICE_CONVERSION_PROVIDERS)
        raise MusicAgentError(f"Unknown voice conversion provider '{provider}'. Available providers: {available}.")
    if provider == "auto":
        return "svcfusion" if explicit_svcfusion_args or has_complete_svcfusion_env() else "placeholder"
    return provider


def _validate_preset(preset: str) -> None:
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise MusicAgentError(f"Unknown voice preset '{preset}'. Available presets: {available}.")


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
