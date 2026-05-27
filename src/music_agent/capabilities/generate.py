"""Music generation capability facade."""

from __future__ import annotations

from pathlib import Path

from ..generation import GenerationRequest, get_generator
from ..generation.synth import detect_style


def generate_music(
    prompt: str,
    duration: float = 8.0,
    output: str | Path | None = None,
    style: str | None = None,
    provider: str = "synth",
    model: str | None = None,
    guidance_scale: float = 3.0,
    seed: int | None = None,
) -> dict[str, object]:
    """Generate music through the selected provider."""
    output_path = Path(output).expanduser() if output else None
    request = GenerationRequest(
        prompt=prompt,
        duration=duration,
        output=output_path,
        style=style,
        model=model,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return get_generator(provider).generate(request)

