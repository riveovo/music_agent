"""Simple natural-language router for the CLI MVP."""

from __future__ import annotations

from pathlib import Path

from ..capabilities import (
    analyze_audio,
    convert_voice,
    generate_music,
    recognize_style,
    separate_stems,
)
from ..errors import MusicAgentError


ROUTE_KEYWORDS = {
    "generate": ("生成", "创作", "写一首", "作曲", "generate", "compose", "make music"),
    "convert_voice": ("换声", "变声", "音色转换", "voice conversion", "convert voice", "change voice"),
    "separate_stems": ("分离", "人声分离", "伴奏", "提取人声", "separate", "stem", "accompaniment", "vocal"),
    "recognize_style": ("风格", "曲风", "类型", "style", "genre"),
    "analyze": ("分析", "解析", "乐曲分析", "结构", "analyze", "analysis"),
}


def route_request(
    request: str,
    audio: str | Path | None = None,
    duration: float = 8.0,
    preset: str = "bright",
    output: str | Path | None = None,
    style: str | None = None,
    provider: str = "synth",
    model: str | None = None,
    guidance_scale: float = 3.0,
    seed: int | None = None,
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
        result = recognize_style(_require_audio_for_route(audio, route))
    elif route == "analyze":
        result = analyze_audio(_require_audio_for_route(audio, route))
    elif route == "separate_stems":
        result = separate_stems(_require_audio_for_route(audio, route))
    elif route == "convert_voice":
        result = convert_voice(_require_audio_for_route(audio, route), preset=preset, output=output)
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
    priority = ["generate", "convert_voice", "separate_stems", "recognize_style", "analyze"]
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
