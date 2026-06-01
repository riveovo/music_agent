"""Command-line entrypoint for the AI music agent MVP."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .agent import route_request
from .capabilities import (
    analyze_audio,
    convert_voice,
    curate_vocal_slices,
    generate_music,
    recognize_style,
    separate_stems,
    slice_audio,
)
from .capabilities.analyze import ANALYSIS_PROVIDERS
from .capabilities.convert_voice import PRESETS, VOICE_CONVERSION_PROVIDERS
from .capabilities.recognize_style import STYLE_RECOGNITION_PROVIDERS
from .errors import MusicAgentError
from .generation import GENERATION_PROVIDERS
from .separation import SUPPORTED_MODEL_TYPES
from .style_recognition import ESSENTIA_STYLE_MODEL_TYPES
from .voice_conversion import SVCFUSION_MODEL_TYPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-agent",
        description="CLI-first MVP for independent music capabilities plus agent routing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate a short MVP music clip.")
    generate.add_argument("--prompt", required=True, help="Music prompt or idea.")
    generate.add_argument("--duration", type=float, default=8.0, help="Duration in seconds.")
    generate.add_argument("--style", help="Optional explicit style hint.")
    generate.add_argument(
        "--provider",
        default="synth",
        choices=sorted(GENERATION_PROVIDERS),
        help="Generation backend. 'synth' is dependency-free; 'musicgen' uses a local model.",
    )
    generate.add_argument("--model", help="Optional provider model name, e.g. facebook/musicgen-small.")
    generate.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="MusicGen classifier-free guidance scale. Higher follows the prompt more closely.",
    )
    generate.add_argument("--seed", type=int, help="Optional random seed for model providers.")
    generate.add_argument("--output", help="Optional output WAV path.")
    generate.set_defaults(handler=_handle_generate)

    recognize = subparsers.add_parser("recognize-style", help="Recognize coarse music style.")
    recognize.add_argument("--audio", required=True, help="Input audio file or directory.")
    recognize.add_argument(
        "--provider",
        default="auto",
        choices=STYLE_RECOGNITION_PROVIDERS,
        help="Style recognition backend. 'essentia' uses local Essentia TensorFlow models.",
    )
    recognize.add_argument("--output-dir", help="Optional output directory.")
    recognize.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    recognize.add_argument("--keep-converted", action="store_true", help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.")
    recognize.add_argument("--ncm-converter", help="Optional ncmdump-compatible executable or command template.")
    recognize.add_argument("--essentia-model-type", choices=ESSENTIA_STYLE_MODEL_TYPES, help="Essentia style model type.")
    recognize.add_argument("--essentia-embedding-model-path", help="Path to the Essentia embedding model .pb.")
    recognize.add_argument("--essentia-classifier-model-path", help="Path to the Essentia genre/style classifier .pb.")
    recognize.add_argument("--essentia-metadata-path", help="Path to the Essentia classifier metadata JSON.")
    recognize.add_argument("--essentia-top-k", type=int, default=8, help="Number of style/tag predictions to return.")
    recognize.set_defaults(handler=_handle_recognize_style)

    analyze = subparsers.add_parser("analyze", help="Analyze audio metadata and loudness.")
    analyze.add_argument("--audio", required=True, help="Input audio file or directory.")
    analyze.add_argument(
        "--provider",
        default="auto",
        choices=ANALYSIS_PROVIDERS,
        help="Analysis backend. 'essentia' extracts BPM, key, chords, sections, and MIR descriptors.",
    )
    analyze.add_argument("--output-dir", help="Optional output directory.")
    analyze.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    analyze.add_argument("--keep-converted", action="store_true", help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.")
    analyze.add_argument("--ncm-converter", help="Optional ncmdump-compatible executable or command template.")
    analyze.add_argument("--essentia-max-sections", type=int, default=12, help="Maximum A/B/C structure sections for Essentia analysis.")
    analyze.set_defaults(handler=_handle_analyze)

    separate = subparsers.add_parser("separate-stems", help="Separate vocals/accompaniment.")
    separate.add_argument("--audio", required=True, help="Input audio file or directory.")
    separate.add_argument("--output-dir", help="Optional output directory.")
    separate.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "heuristic", "msst"],
        help="Separation backend. 'msst' uses a real local RoFormer model; 'heuristic' uses ffmpeg filters.",
    )
    separate.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, help="MSST model type.")
    separate.add_argument("--model-path", help="Path to MSST/RoFormer checkpoint.")
    separate.add_argument("--config-path", help="Path to MSST/RoFormer YAML config.")
    separate.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="MSST inference device. Defaults to auto or MUSIC_AGENT_MSST_DEVICE.",
    )
    separate.add_argument("--use-tta", action="store_true", help="Enable MSST test-time augmentation.")
    separate.add_argument("--instrumental-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for cleaner accompaniment.")
    separate.add_argument("--instrumental-model-path", help="Optional RoFormer checkpoint for cleaner accompaniment.")
    separate.add_argument("--instrumental-config-path", help="Optional RoFormer YAML config for cleaner accompaniment.")
    separate.add_argument("--deharmony-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for vocal deharmony.")
    separate.add_argument("--deharmony-model-path", help="Optional RoFormer checkpoint for vocal deharmony.")
    separate.add_argument("--deharmony-config-path", help="Optional RoFormer YAML config for vocal deharmony.")
    separate.add_argument("--dereverb-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for vocal dereverb/de-echo.")
    separate.add_argument("--dereverb-model-path", help="Optional RoFormer checkpoint for vocal dereverb/de-echo.")
    separate.add_argument("--dereverb-config-path", help="Optional RoFormer YAML config for vocal dereverb/de-echo.")
    separate.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    separate.add_argument("--keep-converted", action="store_true", help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.")
    separate.add_argument("--ncm-converter", help="Optional ncmdump-compatible executable or command template.")
    separate.set_defaults(handler=_handle_separate_stems)

    convert = subparsers.add_parser("convert-voice", help="Convert voice with a placeholder preset or SVCFusion.")
    convert.add_argument("--audio", required=True, help="Input vocal/audio file or directory.")
    convert.add_argument(
        "--provider",
        default="auto",
        choices=VOICE_CONVERSION_PROVIDERS,
        help="Voice conversion backend. 'svcfusion' runs a real local SVC model; 'placeholder' uses ffmpeg filters.",
    )
    convert.add_argument("--preset", default="bright", choices=sorted(PRESETS), help="Voice preset.")
    convert.add_argument("--output", help="Optional output WAV path.")
    convert.add_argument("--output-dir", help="Optional output directory, required for directory batch output control.")
    convert.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    convert.add_argument("--keep-converted", action="store_true", help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.")
    convert.add_argument("--ncm-converter", help="Optional ncmdump-compatible executable or command template.")
    convert.add_argument("--svcfusion-model-type", choices=SVCFUSION_MODEL_TYPES, help="SVCFusion model type.")
    convert.add_argument("--svcfusion-model-path", help="Path to the SVCFusion cascade model checkpoint.")
    convert.add_argument("--svcfusion-config-path", help="Path to the SVCFusion config.yaml.")
    convert.add_argument("--svcfusion-speaker", help="Target speaker name in the SVCFusion model.")
    convert.add_argument(
        "--svcfusion-device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="SVCFusion inference device. Defaults to auto or MUSIC_AGENT_SVCFUSION_DEVICE.",
    )
    convert.add_argument("--svcfusion-source-path", help="Optional path to the checked-out HuanLinOTO/SVCFusion core repo.")
    convert.add_argument("--svcfusion-f0-method", default="rmvpe", help="SVCFusion F0 extractor name.")
    convert.add_argument("--svcfusion-key-change", type=float, default=0.0, help="Pitch shift in semitones.")
    convert.add_argument("--svcfusion-formant-shift-key", type=float, default=0.0, help="Formant shift key.")
    convert.add_argument("--svcfusion-method", default="auto", help="SVCFusion inference method parameter.")
    convert.add_argument("--svcfusion-threshold", type=float, default=-60.0, help="SVCFusion silence/noise threshold.")
    convert.add_argument("--svcfusion-infer-step", default="auto", help="SVCFusion infer_step parameter.")
    convert.add_argument("--svcfusion-t-start", default="auto", help="SVCFusion t_start parameter.")
    convert.add_argument("--svcfusion-vocal-register-factor", type=float, default=1.0, help="SVCFusion vocal register factor.")
    convert.set_defaults(handler=_handle_convert_voice)

    slicer = subparsers.add_parser("slice-audio", help="Slice one audio file or a folder of audio files by silence.")
    slicer.add_argument("--input", required=True, help="Input audio file or directory.")
    slicer.add_argument("--output-dir", help="Optional output directory.")
    slicer.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    slicer.add_argument(
        "--min-length-ms",
        type=int,
        default=3000,
        help="Minimum duration for each sliced clip in milliseconds.",
    )
    slicer.add_argument(
        "--max-length-ms",
        type=int,
        default=10000,
        help="Maximum duration for each sliced clip in milliseconds.",
    )
    slicer.add_argument(
        "--keep-converted",
        action="store_true",
        help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.",
    )
    slicer.add_argument(
        "--ncm-converter",
        help="Optional ncmdump-compatible executable or command template using {input}, {output}, {output_dir}.",
    )
    slicer.set_defaults(handler=_handle_slice_audio)

    curate = subparsers.add_parser(
        "curate-vocal-slices",
        help="Cluster vocal slices and keep the singer cluster with the longest total duration.",
    )
    curate.add_argument("--input", required=True, help="Input vocal slice file or directory.")
    curate.add_argument("--output-dir", help="Optional output directory.")
    curate.add_argument("--recursive", action="store_true", help="Process input directories recursively.")
    curate.add_argument("--min-length-ms", type=int, default=3000, help="Minimum valid slice duration in milliseconds.")
    curate.add_argument("--max-length-ms", type=int, default=10000, help="Maximum valid slice duration in milliseconds.")
    curate.add_argument(
        "--distance-threshold",
        type=float,
        default=0.32,
        help="Cosine distance threshold for agglomerative singer clustering.",
    )
    curate.add_argument(
        "--embedding-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain speaker embedding model name.",
    )
    curate.add_argument("--model-cache-dir", help="Optional local cache directory for the embedding model.")
    curate.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Embedding inference device.",
    )
    curate.add_argument("--keep-converted", action="store_true", help="Keep intermediate WAV files converted from ncm/mp3/flac inputs.")
    curate.add_argument("--ncm-converter", help="Optional ncmdump-compatible executable or command template.")
    curate.set_defaults(handler=_handle_curate_vocal_slices)

    agent = subparsers.add_parser("agent", help="Route a natural-language request to a capability.")
    agent.add_argument("request", help="Natural-language request.")
    agent.add_argument("--audio", help="Optional input audio for analysis/transform requests.")
    agent.add_argument("--duration", type=float, default=8.0, help="Generation duration in seconds.")
    agent.add_argument("--preset", default="bright", choices=sorted(PRESETS), help="Voice preset.")
    agent.add_argument(
        "--voice-provider",
        default="auto",
        choices=VOICE_CONVERSION_PROVIDERS,
        help="Voice conversion backend if the request routes to voice conversion.",
    )
    agent.add_argument("--voice-svcfusion-model-type", choices=SVCFUSION_MODEL_TYPES, help="SVCFusion model type for voice conversion.")
    agent.add_argument("--voice-svcfusion-model-path", help="Path to the SVCFusion cascade model checkpoint.")
    agent.add_argument("--voice-svcfusion-config-path", help="Path to the SVCFusion config.yaml.")
    agent.add_argument("--voice-svcfusion-speaker", help="Target speaker name in the SVCFusion model.")
    agent.add_argument(
        "--voice-svcfusion-device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="SVCFusion inference device if the request routes to voice conversion.",
    )
    agent.add_argument("--voice-svcfusion-source-path", help="Optional path to the checked-out HuanLinOTO/SVCFusion core repo.")
    agent.add_argument("--voice-svcfusion-f0-method", default="rmvpe", help="SVCFusion F0 extractor name.")
    agent.add_argument("--voice-svcfusion-key-change", type=float, default=0.0, help="Pitch shift in semitones for SVCFusion.")
    agent.add_argument("--voice-svcfusion-formant-shift-key", type=float, default=0.0, help="Formant shift key for SVCFusion.")
    agent.add_argument("--voice-svcfusion-method", default="auto", help="SVCFusion inference method parameter.")
    agent.add_argument("--voice-svcfusion-threshold", type=float, default=-60.0, help="SVCFusion silence/noise threshold.")
    agent.add_argument("--voice-svcfusion-infer-step", default="auto", help="SVCFusion infer_step parameter.")
    agent.add_argument("--voice-svcfusion-t-start", default="auto", help="SVCFusion t_start parameter.")
    agent.add_argument("--voice-svcfusion-vocal-register-factor", type=float, default=1.0, help="SVCFusion vocal register factor.")
    agent.add_argument("--style", help="Optional generation style hint.")
    agent.add_argument(
        "--provider",
        default="synth",
        choices=sorted(GENERATION_PROVIDERS),
        help="Generation backend if the request routes to generation.",
    )
    agent.add_argument("--model", help="Optional generation model if the request routes to generation.")
    agent.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="MusicGen guidance scale if the request routes to generation.",
    )
    agent.add_argument("--seed", type=int, help="Optional generation seed if the request routes to generation.")
    agent.add_argument("--output", help="Optional output path for generation or voice conversion.")
    agent.add_argument("--audio-output-dir", help="Optional output directory for routed audio-input capabilities.")
    agent.add_argument("--audio-recursive", action="store_true", help="Recursively process directories for routed audio-input capabilities.")
    agent.add_argument("--audio-keep-converted", action="store_true", help="Keep converted WAV files for routed audio-input capabilities.")
    agent.add_argument("--ncm-converter", help="Optional ncmdump-compatible command for routed audio-input capabilities.")
    agent.add_argument(
        "--analysis-provider",
        default="auto",
        choices=ANALYSIS_PROVIDERS,
        help="Analysis backend if the request routes to music analysis.",
    )
    agent.add_argument("--analysis-essentia-max-sections", type=int, default=12, help="Maximum A/B/C sections for Essentia analysis.")
    agent.add_argument("--slice-output-dir", help="Optional output directory if the request routes to audio slicing.")
    agent.add_argument("--slice-recursive", action="store_true", help="Recursively process directories for audio slicing.")
    agent.add_argument("--slice-min-length-ms", type=int, default=3000, help="Audio slicing minimum clip duration in milliseconds.")
    agent.add_argument("--slice-max-length-ms", type=int, default=10000, help="Audio slicing maximum clip duration in milliseconds.")
    agent.add_argument("--slice-keep-converted", action="store_true", help="Keep converted WAV files for audio slicing.")
    agent.add_argument("--slice-ncm-converter", help="Optional ncmdump-compatible command for NCM audio slicing.")
    agent.add_argument(
        "--style-provider",
        default="auto",
        choices=STYLE_RECOGNITION_PROVIDERS,
        help="Style recognition backend if the request routes to style recognition.",
    )
    agent.add_argument("--style-essentia-model-type", choices=ESSENTIA_STYLE_MODEL_TYPES, help="Essentia style model type.")
    agent.add_argument("--style-essentia-embedding-model-path", help="Path to the Essentia embedding model .pb.")
    agent.add_argument("--style-essentia-classifier-model-path", help="Path to the Essentia genre/style classifier .pb.")
    agent.add_argument("--style-essentia-metadata-path", help="Path to the Essentia classifier metadata JSON.")
    agent.add_argument("--style-essentia-top-k", type=int, default=8, help="Number of style/tag predictions to return.")
    agent.add_argument("--curation-min-length-ms", type=int, default=3000, help="Vocal curation minimum valid slice duration in milliseconds.")
    agent.add_argument("--curation-max-length-ms", type=int, default=10000, help="Vocal curation maximum valid slice duration in milliseconds.")
    agent.add_argument("--curation-distance-threshold", type=float, default=0.32, help="Vocal curation clustering distance threshold.")
    agent.add_argument(
        "--curation-embedding-model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain embedding model for vocal curation.",
    )
    agent.add_argument("--curation-model-cache-dir", help="Optional model cache directory for vocal curation.")
    agent.add_argument(
        "--curation-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Embedding inference device for vocal curation.",
    )
    agent.add_argument(
        "--separation-provider",
        default="auto",
        choices=["auto", "heuristic", "msst"],
        help="Separation backend if the request routes to stem separation.",
    )
    agent.add_argument("--separation-model-type", choices=SUPPORTED_MODEL_TYPES, help="MSST model type for separation.")
    agent.add_argument("--separation-model-path", help="Path to MSST/RoFormer checkpoint for separation.")
    agent.add_argument("--separation-config-path", help="Path to MSST/RoFormer YAML config for separation.")
    agent.add_argument(
        "--separation-device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="MSST inference device if the request routes to stem separation.",
    )
    agent.add_argument("--separation-use-tta", action="store_true", help="Enable MSST TTA for stem separation.")
    agent.add_argument("--separation-instrumental-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for cleaner accompaniment.")
    agent.add_argument("--separation-instrumental-model-path", help="Optional RoFormer checkpoint for cleaner accompaniment.")
    agent.add_argument("--separation-instrumental-config-path", help="Optional RoFormer YAML config for cleaner accompaniment.")
    agent.add_argument("--separation-deharmony-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for vocal deharmony.")
    agent.add_argument("--separation-deharmony-model-path", help="Optional RoFormer checkpoint for vocal deharmony.")
    agent.add_argument("--separation-deharmony-config-path", help="Optional RoFormer YAML config for vocal deharmony.")
    agent.add_argument("--separation-dereverb-model-type", choices=SUPPORTED_MODEL_TYPES, help="Optional RoFormer model type for vocal dereverb/de-echo.")
    agent.add_argument("--separation-dereverb-model-path", help="Optional RoFormer checkpoint for vocal dereverb/de-echo.")
    agent.add_argument("--separation-dereverb-config-path", help="Optional RoFormer YAML config for vocal dereverb/de-echo.")
    agent.set_defaults(handler=_handle_agent)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except MusicAgentError as exc:
        _print_json({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        _print_json({"ok": False, "error": "Interrupted."}, stream=sys.stderr)
        return 130

    _print_json({"ok": True, "data": result})
    return 0


def _handle_generate(args: argparse.Namespace) -> dict[str, object]:
    return generate_music(
        prompt=args.prompt,
        duration=args.duration,
        output=args.output,
        style=args.style,
        provider=args.provider,
        model=args.model,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )


def _handle_recognize_style(args: argparse.Namespace) -> dict[str, object]:
    return recognize_style(
        args.audio,
        provider=args.provider,
        output_dir=args.output_dir,
        recursive=args.recursive,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        essentia_model_type=args.essentia_model_type,
        essentia_embedding_model_path=args.essentia_embedding_model_path,
        essentia_classifier_model_path=args.essentia_classifier_model_path,
        essentia_metadata_path=args.essentia_metadata_path,
        essentia_top_k=args.essentia_top_k,
        progress=_print_progress,
    )


def _handle_analyze(args: argparse.Namespace) -> dict[str, object]:
    return analyze_audio(
        args.audio,
        provider=args.provider,
        output_dir=args.output_dir,
        recursive=args.recursive,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        essentia_max_sections=args.essentia_max_sections,
        progress=_print_progress,
    )


def _handle_separate_stems(args: argparse.Namespace) -> dict[str, object]:
    return separate_stems(
        args.audio,
        output_dir=args.output_dir,
        provider=args.provider,
        model_type=args.model_type,
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        use_tta=args.use_tta,
        instrumental_model_type=args.instrumental_model_type,
        instrumental_model_path=args.instrumental_model_path,
        instrumental_config_path=args.instrumental_config_path,
        deharmony_model_type=args.deharmony_model_type,
        deharmony_model_path=args.deharmony_model_path,
        deharmony_config_path=args.deharmony_config_path,
        dereverb_model_type=args.dereverb_model_type,
        dereverb_model_path=args.dereverb_model_path,
        dereverb_config_path=args.dereverb_config_path,
        recursive=args.recursive,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        progress=_print_progress,
    )


def _handle_convert_voice(args: argparse.Namespace) -> dict[str, object]:
    return convert_voice(
        args.audio,
        preset=args.preset,
        output=args.output,
        provider=args.provider,
        output_dir=args.output_dir,
        recursive=args.recursive,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        svcfusion_model_type=args.svcfusion_model_type,
        svcfusion_model_path=args.svcfusion_model_path,
        svcfusion_config_path=args.svcfusion_config_path,
        svcfusion_speaker=args.svcfusion_speaker,
        svcfusion_device=args.svcfusion_device,
        svcfusion_source_path=args.svcfusion_source_path,
        svcfusion_f0_method=args.svcfusion_f0_method,
        svcfusion_key_change=args.svcfusion_key_change,
        svcfusion_formant_shift_key=args.svcfusion_formant_shift_key,
        svcfusion_method=args.svcfusion_method,
        svcfusion_threshold=args.svcfusion_threshold,
        svcfusion_infer_step=args.svcfusion_infer_step,
        svcfusion_t_start=args.svcfusion_t_start,
        svcfusion_vocal_register_factor=args.svcfusion_vocal_register_factor,
        progress=_print_progress,
    )


def _handle_slice_audio(args: argparse.Namespace) -> dict[str, object]:
    return slice_audio(
        args.input,
        output_dir=args.output_dir,
        recursive=args.recursive,
        min_length_ms=args.min_length_ms,
        max_length_ms=args.max_length_ms,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        progress=_print_progress,
    )


def _handle_curate_vocal_slices(args: argparse.Namespace) -> dict[str, object]:
    return curate_vocal_slices(
        args.input,
        output_dir=args.output_dir,
        recursive=args.recursive,
        min_length_ms=args.min_length_ms,
        max_length_ms=args.max_length_ms,
        distance_threshold=args.distance_threshold,
        embedding_model=args.embedding_model,
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        keep_converted=args.keep_converted,
        ncm_converter=args.ncm_converter,
        progress=_print_progress,
    )


def _handle_agent(args: argparse.Namespace) -> dict[str, object]:
    return route_request(
        request=args.request,
        audio=args.audio,
        duration=args.duration,
        preset=args.preset,
        voice_provider=args.voice_provider,
        voice_svcfusion_model_type=args.voice_svcfusion_model_type,
        voice_svcfusion_model_path=args.voice_svcfusion_model_path,
        voice_svcfusion_config_path=args.voice_svcfusion_config_path,
        voice_svcfusion_speaker=args.voice_svcfusion_speaker,
        voice_svcfusion_device=args.voice_svcfusion_device,
        voice_svcfusion_source_path=args.voice_svcfusion_source_path,
        voice_svcfusion_f0_method=args.voice_svcfusion_f0_method,
        voice_svcfusion_key_change=args.voice_svcfusion_key_change,
        voice_svcfusion_formant_shift_key=args.voice_svcfusion_formant_shift_key,
        voice_svcfusion_method=args.voice_svcfusion_method,
        voice_svcfusion_threshold=args.voice_svcfusion_threshold,
        voice_svcfusion_infer_step=args.voice_svcfusion_infer_step,
        voice_svcfusion_t_start=args.voice_svcfusion_t_start,
        voice_svcfusion_vocal_register_factor=args.voice_svcfusion_vocal_register_factor,
        output=args.output,
        style=args.style,
        provider=args.provider,
        model=args.model,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        audio_output_dir=args.audio_output_dir,
        audio_recursive=args.audio_recursive,
        audio_keep_converted=args.audio_keep_converted,
        audio_ncm_converter=args.ncm_converter,
        analysis_provider=args.analysis_provider,
        analysis_essentia_max_sections=args.analysis_essentia_max_sections,
        slice_output_dir=args.slice_output_dir,
        slice_recursive=args.slice_recursive,
        slice_min_length_ms=args.slice_min_length_ms,
        slice_max_length_ms=args.slice_max_length_ms,
        slice_keep_converted=args.slice_keep_converted,
        slice_ncm_converter=args.slice_ncm_converter,
        style_provider=args.style_provider,
        style_essentia_model_type=args.style_essentia_model_type,
        style_essentia_embedding_model_path=args.style_essentia_embedding_model_path,
        style_essentia_classifier_model_path=args.style_essentia_classifier_model_path,
        style_essentia_metadata_path=args.style_essentia_metadata_path,
        style_essentia_top_k=args.style_essentia_top_k,
        curation_min_length_ms=args.curation_min_length_ms,
        curation_max_length_ms=args.curation_max_length_ms,
        curation_distance_threshold=args.curation_distance_threshold,
        curation_embedding_model=args.curation_embedding_model,
        curation_model_cache_dir=args.curation_model_cache_dir,
        curation_device=args.curation_device,
        separation_provider=args.separation_provider,
        separation_model_type=args.separation_model_type,
        separation_model_path=args.separation_model_path,
        separation_config_path=args.separation_config_path,
        separation_device=args.separation_device,
        separation_use_tta=args.separation_use_tta,
        separation_instrumental_model_type=args.separation_instrumental_model_type,
        separation_instrumental_model_path=args.separation_instrumental_model_path,
        separation_instrumental_config_path=args.separation_instrumental_config_path,
        separation_deharmony_model_type=args.separation_deharmony_model_type,
        separation_deharmony_model_path=args.separation_deharmony_model_path,
        separation_deharmony_config_path=args.separation_deharmony_config_path,
        separation_dereverb_model_type=args.separation_dereverb_model_type,
        separation_dereverb_model_path=args.separation_dereverb_model_path,
        separation_dereverb_config_path=args.separation_dereverb_config_path,
        progress=_print_progress,
    )


def _print_json(payload: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stream)


def _print_progress(message: str) -> None:
    print(f"[music-agent] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
