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
    generate_music,
    recognize_style,
    separate_stems,
)
from .capabilities.convert_voice import PRESETS
from .errors import MusicAgentError
from .generation import GENERATION_PROVIDERS


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
    recognize.add_argument("--audio", required=True, help="Input audio file.")
    recognize.set_defaults(handler=_handle_recognize_style)

    analyze = subparsers.add_parser("analyze", help="Analyze audio metadata and loudness.")
    analyze.add_argument("--audio", required=True, help="Input audio file.")
    analyze.set_defaults(handler=_handle_analyze)

    separate = subparsers.add_parser("separate-stems", help="Separate vocals/accompaniment heuristically.")
    separate.add_argument("--audio", required=True, help="Input audio file.")
    separate.add_argument("--output-dir", help="Optional output directory.")
    separate.set_defaults(handler=_handle_separate_stems)

    convert = subparsers.add_parser("convert-voice", help="Apply a placeholder voice conversion preset.")
    convert.add_argument("--audio", required=True, help="Input vocal/audio file.")
    convert.add_argument("--preset", default="bright", choices=sorted(PRESETS), help="Voice preset.")
    convert.add_argument("--output", help="Optional output WAV path.")
    convert.set_defaults(handler=_handle_convert_voice)

    agent = subparsers.add_parser("agent", help="Route a natural-language request to a capability.")
    agent.add_argument("request", help="Natural-language request.")
    agent.add_argument("--audio", help="Optional input audio for analysis/transform requests.")
    agent.add_argument("--duration", type=float, default=8.0, help="Generation duration in seconds.")
    agent.add_argument("--preset", default="bright", choices=sorted(PRESETS), help="Voice preset.")
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
    return recognize_style(args.audio)


def _handle_analyze(args: argparse.Namespace) -> dict[str, object]:
    return analyze_audio(args.audio)


def _handle_separate_stems(args: argparse.Namespace) -> dict[str, object]:
    return separate_stems(args.audio, output_dir=args.output_dir)


def _handle_convert_voice(args: argparse.Namespace) -> dict[str, object]:
    return convert_voice(args.audio, preset=args.preset, output=args.output)


def _handle_agent(args: argparse.Namespace) -> dict[str, object]:
    return route_request(
        request=args.request,
        audio=args.audio,
        duration=args.duration,
        preset=args.preset,
        output=args.output,
        style=args.style,
        provider=args.provider,
        model=args.model,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )


def _print_json(payload: dict[str, Any], stream: Any = sys.stdout) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stream)


if __name__ == "__main__":
    raise SystemExit(main())
