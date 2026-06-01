"""Simple natural-language router for the CLI MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

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


ROUTE_KEYWORDS = {
    "generate": ("生成", "创作", "写一首", "作曲", "generate", "compose", "make music"),
    "convert_voice": ("换声", "变声", "音色转换", "voice conversion", "convert voice", "change voice"),
    "separate_stems": ("分离", "人声分离", "伴奏", "提取人声", "separate", "stem", "accompaniment", "vocal"),
    "slice_audio": ("切片", "切分", "分割音频", "音频分段", "slice", "slicer", "split audio"),
    "curate_vocal_slices": ("清洗", "筛选", "剔除", "目标歌手", "curate", "filter vocal", "vocal curation"),
    "recognize_style": ("风格", "曲风", "类型", "style", "genre"),
    "analyze": ("分析", "解析", "乐曲分析", "结构", "analyze", "analysis"),
}


def route_request(
    request: str,
    audio: str | Path | None = None,
    duration: float = 8.0,
    preset: str = "bright",
    voice_provider: str = "auto",
    voice_svcfusion_model_type: str | None = None,
    voice_svcfusion_model_path: str | Path | None = None,
    voice_svcfusion_config_path: str | Path | None = None,
    voice_svcfusion_speaker: str | None = None,
    voice_svcfusion_device: str | None = None,
    voice_svcfusion_source_path: str | Path | None = None,
    voice_svcfusion_f0_method: str = "rmvpe",
    voice_svcfusion_key_change: float = 0.0,
    voice_svcfusion_formant_shift_key: float = 0.0,
    voice_svcfusion_method: str = "auto",
    voice_svcfusion_threshold: float = -60.0,
    voice_svcfusion_infer_step: str = "auto",
    voice_svcfusion_t_start: str = "auto",
    voice_svcfusion_vocal_register_factor: float = 1.0,
    output: str | Path | None = None,
    style: str | None = None,
    provider: str = "synth",
    model: str | None = None,
    guidance_scale: float = 3.0,
    seed: int | None = None,
    audio_output_dir: str | Path | None = None,
    audio_recursive: bool = False,
    audio_keep_converted: bool = False,
    audio_ncm_converter: str | None = None,
    slice_output_dir: str | Path | None = None,
    slice_recursive: bool = False,
    slice_min_length_ms: int = 3000,
    slice_max_length_ms: int = 10000,
    slice_keep_converted: bool = False,
    slice_ncm_converter: str | None = None,
    curation_min_length_ms: int = 3000,
    curation_max_length_ms: int = 10000,
    curation_distance_threshold: float = 0.32,
    curation_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
    curation_model_cache_dir: str | Path | None = None,
    curation_device: str | None = "auto",
    separation_provider: str = "auto",
    separation_model_type: str | None = None,
    separation_model_path: str | Path | None = None,
    separation_config_path: str | Path | None = None,
    separation_device: str | None = None,
    separation_use_tta: bool = False,
    separation_instrumental_model_type: str | None = None,
    separation_instrumental_model_path: str | Path | None = None,
    separation_instrumental_config_path: str | Path | None = None,
    separation_deharmony_model_type: str | None = None,
    separation_deharmony_model_path: str | Path | None = None,
    separation_deharmony_config_path: str | Path | None = None,
    separation_dereverb_model_type: str | None = None,
    separation_dereverb_model_path: str | Path | None = None,
    separation_dereverb_config_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Route a natural-language request to one capability and execute it."""
    if not request.strip():
        raise MusicAgentError("Agent request cannot be empty.")

    route, reason = _choose_route(request, audio)
    if route == "generate":
        result = generate_music(
            request,
            duration=duration,
            output=output,
            style=style,
            provider=provider,
            model=model,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    elif route == "recognize_style":
        result = recognize_style(
            _require_audio_for_route(audio, route),
            output_dir=audio_output_dir,
            recursive=audio_recursive,
            keep_converted=audio_keep_converted,
            ncm_converter=audio_ncm_converter,
            progress=progress,
        )
    elif route == "analyze":
        result = analyze_audio(
            _require_audio_for_route(audio, route),
            output_dir=audio_output_dir,
            recursive=audio_recursive,
            keep_converted=audio_keep_converted,
            ncm_converter=audio_ncm_converter,
            progress=progress,
        )
    elif route == "separate_stems":
        result = separate_stems(
            _require_audio_for_route(audio, route),
            output_dir=audio_output_dir,
            provider=separation_provider,
            model_type=separation_model_type,
            model_path=separation_model_path,
            config_path=separation_config_path,
            device=separation_device,
            use_tta=separation_use_tta,
            instrumental_model_type=separation_instrumental_model_type,
            instrumental_model_path=separation_instrumental_model_path,
            instrumental_config_path=separation_instrumental_config_path,
            deharmony_model_type=separation_deharmony_model_type,
            deharmony_model_path=separation_deharmony_model_path,
            deharmony_config_path=separation_deharmony_config_path,
            dereverb_model_type=separation_dereverb_model_type,
            dereverb_model_path=separation_dereverb_model_path,
            dereverb_config_path=separation_dereverb_config_path,
            recursive=audio_recursive,
            keep_converted=audio_keep_converted,
            ncm_converter=audio_ncm_converter,
            progress=progress,
        )
    elif route == "slice_audio":
        result = slice_audio(
            _require_audio_for_route(audio, route),
            output_dir=slice_output_dir or audio_output_dir,
            recursive=slice_recursive or audio_recursive,
            min_length_ms=slice_min_length_ms,
            max_length_ms=slice_max_length_ms,
            keep_converted=slice_keep_converted or audio_keep_converted,
            ncm_converter=slice_ncm_converter or audio_ncm_converter,
            progress=progress,
        )
    elif route == "curate_vocal_slices":
        result = curate_vocal_slices(
            _require_audio_for_route(audio, route),
            output_dir=audio_output_dir,
            recursive=audio_recursive,
            min_length_ms=curation_min_length_ms,
            max_length_ms=curation_max_length_ms,
            distance_threshold=curation_distance_threshold,
            embedding_model=curation_embedding_model,
            model_cache_dir=curation_model_cache_dir,
            device=curation_device,
            keep_converted=audio_keep_converted,
            ncm_converter=audio_ncm_converter,
            progress=progress,
        )
    elif route == "convert_voice":
        result = convert_voice(
            _require_audio_for_route(audio, route),
            preset=preset,
            output=output,
            provider=voice_provider,
            output_dir=audio_output_dir,
            recursive=audio_recursive,
            keep_converted=audio_keep_converted,
            ncm_converter=audio_ncm_converter,
            svcfusion_model_type=voice_svcfusion_model_type,
            svcfusion_model_path=voice_svcfusion_model_path,
            svcfusion_config_path=voice_svcfusion_config_path,
            svcfusion_speaker=voice_svcfusion_speaker,
            svcfusion_device=voice_svcfusion_device,
            svcfusion_source_path=voice_svcfusion_source_path,
            svcfusion_f0_method=voice_svcfusion_f0_method,
            svcfusion_key_change=voice_svcfusion_key_change,
            svcfusion_formant_shift_key=voice_svcfusion_formant_shift_key,
            svcfusion_method=voice_svcfusion_method,
            svcfusion_threshold=voice_svcfusion_threshold,
            svcfusion_infer_step=voice_svcfusion_infer_step,
            svcfusion_t_start=voice_svcfusion_t_start,
            svcfusion_vocal_register_factor=voice_svcfusion_vocal_register_factor,
            progress=progress,
        )
    else:
        raise MusicAgentError(f"Unsupported route: {route}")

    return {
        "capability": "agent",
        "request": request,
        "routed_to": route,
        "route_reason": reason,
        "result": result,
    }


def _choose_route(request: str, audio: str | Path | None) -> tuple[str, str]:
    text = request.lower()
    # Priority matters: "分析这首歌的风格" should answer the style question.
    priority = ["generate", "convert_voice", "separate_stems", "curate_vocal_slices", "slice_audio", "recognize_style", "analyze"]
    for route in priority:
        keyword = _first_match(text, ROUTE_KEYWORDS[route])
        if keyword:
            return route, f"matched keyword '{keyword}'"
    if audio:
        return "analyze", "no explicit keyword; audio was provided so defaulted to analysis"
    return "generate", "no explicit keyword or audio; defaulted to music generation"


def _first_match(text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if keyword.lower() in text:
            return keyword
    return None


def _require_audio_for_route(audio: str | Path | None, route: str) -> str | Path:
    if audio is None:
        raise MusicAgentError(f"Agent routed to '{route}', which requires --audio.")
    return audio
