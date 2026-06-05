"""Built-in external tools for music capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..capabilities import (
    analyze_audio,
    convert_voice,
    curate_vocal_slices,
    generate_music,
    recognize_style,
    separate_stems,
    slice_audio,
)
from ..errors import MusicAgentError
from .core import ToolRegistry, ToolSpec


def register_music_tools(
    registry: ToolRegistry,
    runtime_options: Mapping[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    registry.register(
        ToolSpec(
            name="music.generate",
            description="Generate a short music clip from a natural-language prompt.",
            parameters=_schema(
                {
                    "prompt": {"type": "string", "description": "Music idea or prompt. Defaults to the user request."},
                    "duration": {"type": "number", "description": "Duration in seconds."},
                    "style": {"type": "string", "description": "Optional style hint."},
                    "provider": {"type": "string", "description": "Generation provider, e.g. synth or musicgen."},
                    "model": {"type": "string", "description": "Optional generation model name."},
                    "guidance_scale": {"type": "number", "description": "MusicGen guidance scale."},
                    "seed": {"type": "integer", "description": "Optional random seed."},
                    "output": {"type": "string", "description": "Optional output WAV path."},
                }
            ),
            handler=lambda args: generate_music(
                prompt=str(args.get("prompt") or runtime_options.get("request") or ""),
                duration=float(args.get("duration", runtime_options.get("duration", 8.0))),
                output=args.get("output") or runtime_options.get("output"),
                style=args.get("style") or runtime_options.get("style"),
                provider=str(args.get("provider") or runtime_options.get("provider") or "synth"),
                model=args.get("model") or runtime_options.get("model"),
                guidance_scale=float(args.get("guidance_scale", runtime_options.get("guidance_scale", 3.0))),
                seed=args.get("seed", runtime_options.get("seed")),
            ),
            danger_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="music.analyze_audio",
            description="Analyze audio metadata, loudness, and optional musical structure.",
            parameters=_audio_schema(
                {
                    "provider": {"type": "string", "description": "Analysis provider: auto, basic, or essentia."},
                    "output_dir": {"type": "string", "description": "Optional output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "keep_converted": {"type": "boolean", "description": "Keep converted WAV intermediates."},
                    "ncm_converter": {"type": "string", "description": "Optional ncmdump-compatible converter."},
                    "essentia_max_sections": {"type": "integer", "description": "Maximum structure sections for Essentia."},
                }
            ),
            handler=lambda args: analyze_audio(
                _required_audio(args, runtime_options, "music.analyze_audio"),
                provider=str(args.get("provider") or runtime_options.get("analysis_provider") or "auto"),
                output_dir=args.get("output_dir") or runtime_options.get("audio_output_dir"),
                recursive=bool(args.get("recursive", runtime_options.get("audio_recursive", False))),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("audio_ncm_converter"),
                essentia_max_sections=int(args.get("essentia_max_sections", runtime_options.get("analysis_essentia_max_sections", 12))),
                progress=progress,
            ),
            danger_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="music.recognize_style",
            description="Recognize style or genre from an audio file or batch directory.",
            parameters=_audio_schema(
                {
                    "provider": {"type": "string", "description": "Style provider: auto, heuristic, or essentia."},
                    "output_dir": {"type": "string", "description": "Optional output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "keep_converted": {"type": "boolean", "description": "Keep converted WAV intermediates."},
                    "ncm_converter": {"type": "string", "description": "Optional ncmdump-compatible converter."},
                    "essentia_top_k": {"type": "integer", "description": "Number of style/tag predictions."},
                }
            ),
            handler=lambda args: recognize_style(
                _required_audio(args, runtime_options, "music.recognize_style"),
                provider=str(args.get("provider") or runtime_options.get("style_provider") or "auto"),
                output_dir=args.get("output_dir") or runtime_options.get("audio_output_dir"),
                recursive=bool(args.get("recursive", runtime_options.get("audio_recursive", False))),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("audio_ncm_converter"),
                essentia_model_type=args.get("essentia_model_type") or runtime_options.get("style_essentia_model_type"),
                essentia_embedding_model_path=args.get("essentia_embedding_model_path") or runtime_options.get("style_essentia_embedding_model_path"),
                essentia_classifier_model_path=args.get("essentia_classifier_model_path") or runtime_options.get("style_essentia_classifier_model_path"),
                essentia_metadata_path=args.get("essentia_metadata_path") or runtime_options.get("style_essentia_metadata_path"),
                essentia_top_k=int(args.get("essentia_top_k", runtime_options.get("style_essentia_top_k", 8))),
                progress=progress,
            ),
            danger_level="low",
        )
    )
    registry.register(
        ToolSpec(
            name="music.separate_stems",
            description="Separate vocals and accompaniment from an audio file or batch directory.",
            parameters=_audio_schema(
                {
                    "provider": {"type": "string", "description": "Separation provider: auto, heuristic, or msst."},
                    "output_dir": {"type": "string", "description": "Optional output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "keep_converted": {"type": "boolean", "description": "Keep converted WAV intermediates."},
                    "model_type": {"type": "string", "description": "MSST model type."},
                    "model_path": {"type": "string", "description": "MSST checkpoint path."},
                    "config_path": {"type": "string", "description": "MSST YAML config path."},
                    "device": {"type": "string", "description": "Inference device."},
                }
            ),
            handler=lambda args: separate_stems(
                _required_audio(args, runtime_options, "music.separate_stems"),
                output_dir=args.get("output_dir") or runtime_options.get("audio_output_dir"),
                provider=str(args.get("provider") or runtime_options.get("separation_provider") or "auto"),
                model_type=args.get("model_type") or runtime_options.get("separation_model_type"),
                model_path=args.get("model_path") or runtime_options.get("separation_model_path"),
                config_path=args.get("config_path") or runtime_options.get("separation_config_path"),
                device=args.get("device") or runtime_options.get("separation_device"),
                use_tta=bool(args.get("use_tta", runtime_options.get("separation_use_tta", False))),
                instrumental_model_type=runtime_options.get("separation_instrumental_model_type"),
                instrumental_model_path=runtime_options.get("separation_instrumental_model_path"),
                instrumental_config_path=runtime_options.get("separation_instrumental_config_path"),
                deharmony_model_type=runtime_options.get("separation_deharmony_model_type"),
                deharmony_model_path=runtime_options.get("separation_deharmony_model_path"),
                deharmony_config_path=runtime_options.get("separation_deharmony_config_path"),
                dereverb_model_type=runtime_options.get("separation_dereverb_model_type"),
                dereverb_model_path=runtime_options.get("separation_dereverb_model_path"),
                dereverb_config_path=runtime_options.get("separation_dereverb_config_path"),
                recursive=bool(args.get("recursive", runtime_options.get("audio_recursive", False))),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("audio_ncm_converter"),
                progress=progress,
            ),
            danger_level="medium",
        )
    )
    registry.register(
        ToolSpec(
            name="music.convert_voice",
            description="Convert a vocal/audio file with a preset or SVCFusion model.",
            parameters=_audio_schema(
                {
                    "preset": {"type": "string", "description": "Voice preset."},
                    "provider": {"type": "string", "description": "Voice provider: auto, placeholder, or svcfusion."},
                    "output": {"type": "string", "description": "Optional output WAV path."},
                    "output_dir": {"type": "string", "description": "Optional batch output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "keep_converted": {"type": "boolean", "description": "Keep converted WAV intermediates."},
                }
            ),
            handler=lambda args: convert_voice(
                _required_audio(args, runtime_options, "music.convert_voice"),
                preset=str(args.get("preset") or runtime_options.get("preset") or "bright"),
                output=args.get("output") or runtime_options.get("output"),
                provider=str(args.get("provider") or runtime_options.get("voice_provider") or "auto"),
                output_dir=args.get("output_dir") or runtime_options.get("audio_output_dir"),
                recursive=bool(args.get("recursive", runtime_options.get("audio_recursive", False))),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("audio_ncm_converter"),
                svcfusion_model_type=runtime_options.get("voice_svcfusion_model_type"),
                svcfusion_model_path=runtime_options.get("voice_svcfusion_model_path"),
                svcfusion_config_path=runtime_options.get("voice_svcfusion_config_path"),
                svcfusion_speaker=runtime_options.get("voice_svcfusion_speaker"),
                svcfusion_device=runtime_options.get("voice_svcfusion_device"),
                svcfusion_source_path=runtime_options.get("voice_svcfusion_source_path"),
                svcfusion_f0_method=str(runtime_options.get("voice_svcfusion_f0_method") or "rmvpe"),
                svcfusion_key_change=float(runtime_options.get("voice_svcfusion_key_change", 0.0)),
                svcfusion_formant_shift_key=float(runtime_options.get("voice_svcfusion_formant_shift_key", 0.0)),
                svcfusion_method=str(runtime_options.get("voice_svcfusion_method") or "auto"),
                svcfusion_threshold=float(runtime_options.get("voice_svcfusion_threshold", -60.0)),
                svcfusion_infer_step=str(runtime_options.get("voice_svcfusion_infer_step") or "auto"),
                svcfusion_t_start=str(runtime_options.get("voice_svcfusion_t_start") or "auto"),
                svcfusion_vocal_register_factor=float(runtime_options.get("voice_svcfusion_vocal_register_factor", 1.0)),
                progress=progress,
            ),
            danger_level="medium",
        )
    )
    registry.register(
        ToolSpec(
            name="music.slice_audio",
            description="Slice audio into vocal/music clips by silence and target length.",
            parameters=_schema(
                {
                    "input": {"type": "string", "description": "Input audio file or directory. Defaults to CLI --audio."},
                    "output_dir": {"type": "string", "description": "Optional output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "min_length_ms": {"type": "integer", "description": "Minimum clip duration."},
                    "max_length_ms": {"type": "integer", "description": "Maximum clip duration."},
                    "keep_converted": {"type": "boolean", "description": "Keep converted WAV intermediates."},
                    "ncm_converter": {"type": "string", "description": "Optional ncmdump-compatible converter."},
                }
            ),
            handler=lambda args: slice_audio(
                args.get("input") or _required_audio(args, runtime_options, "music.slice_audio"),
                output_dir=args.get("output_dir") or runtime_options.get("slice_output_dir") or runtime_options.get("audio_output_dir"),
                recursive=bool(args.get("recursive", runtime_options.get("slice_recursive", False) or runtime_options.get("audio_recursive", False))),
                min_length_ms=int(args.get("min_length_ms", runtime_options.get("slice_min_length_ms", 3000))),
                max_length_ms=int(args.get("max_length_ms", runtime_options.get("slice_max_length_ms", 10000))),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("slice_keep_converted", False) or runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("slice_ncm_converter") or runtime_options.get("audio_ncm_converter"),
                progress=progress,
            ),
            danger_level="medium",
        )
    )
    registry.register(
        ToolSpec(
            name="music.curate_vocal_slices",
            description="Cluster vocal slices and keep the likely target singer cluster.",
            parameters=_schema(
                {
                    "input": {"type": "string", "description": "Input vocal slice file or directory. Defaults to CLI --audio."},
                    "output_dir": {"type": "string", "description": "Optional output directory."},
                    "recursive": {"type": "boolean", "description": "Process directories recursively."},
                    "min_length_ms": {"type": "integer", "description": "Minimum valid slice duration."},
                    "max_length_ms": {"type": "integer", "description": "Maximum valid slice duration."},
                    "distance_threshold": {"type": "number", "description": "Cosine distance clustering threshold."},
                }
            ),
            handler=lambda args: curate_vocal_slices(
                args.get("input") or _required_audio(args, runtime_options, "music.curate_vocal_slices"),
                output_dir=args.get("output_dir") or runtime_options.get("audio_output_dir"),
                recursive=bool(args.get("recursive", runtime_options.get("audio_recursive", False))),
                min_length_ms=int(args.get("min_length_ms", runtime_options.get("curation_min_length_ms", 3000))),
                max_length_ms=int(args.get("max_length_ms", runtime_options.get("curation_max_length_ms", 10000))),
                distance_threshold=float(args.get("distance_threshold", runtime_options.get("curation_distance_threshold", 0.32))),
                embedding_model=str(args.get("embedding_model") or runtime_options.get("curation_embedding_model") or "speechbrain/spkrec-ecapa-voxceleb"),
                model_cache_dir=args.get("model_cache_dir") or runtime_options.get("curation_model_cache_dir"),
                device=args.get("device") or runtime_options.get("curation_device"),
                keep_converted=bool(args.get("keep_converted", runtime_options.get("audio_keep_converted", False))),
                ncm_converter=args.get("ncm_converter") or runtime_options.get("audio_ncm_converter"),
                progress=progress,
            ),
            danger_level="medium",
        )
    )


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _audio_schema(extra_properties: dict[str, Any]) -> dict[str, Any]:
    return _schema(
        {
            "audio": {"type": "string", "description": "Input audio file or directory. Can be omitted when CLI --audio was provided."},
        } | extra_properties
    )


def _required_audio(args: Mapping[str, Any], runtime_options: Mapping[str, Any], tool_name: str) -> Any:
    audio = args.get("audio") or runtime_options.get("audio")
    if not audio:
        raise MusicAgentError(f"Tool {tool_name} requires an audio/input path. Provide --audio or pass an audio argument.")
    return audio
