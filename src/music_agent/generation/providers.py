"""Provider registry for music generation."""

from __future__ import annotations

from ..errors import MusicAgentError
from .base import MusicGenerator
from .musicgen_local import LocalMusicGenGenerator
from .synth import SynthGenerator


GENERATION_PROVIDERS = {
    "synth": SynthGenerator,
    "musicgen": LocalMusicGenGenerator,
}


def get_generator(provider: str) -> MusicGenerator:
    key = provider.strip().lower()
    generator_class = GENERATION_PROVIDERS.get(key)
    if generator_class is None:
        available = ", ".join(sorted(GENERATION_PROVIDERS))
        raise MusicAgentError(f"Unknown generation provider '{provider}'. Available providers: {available}.")
    return generator_class()

