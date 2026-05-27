"""Independent music capability adapters."""

from .analyze import analyze_audio
from .convert_voice import convert_voice
from .generate import generate_music
from .recognize_style import recognize_style
from .separate_stems import separate_stems

__all__ = [
    "analyze_audio",
    "convert_voice",
    "generate_music",
    "recognize_style",
    "separate_stems",
]

