"""Independent music capability adapters."""

from .analyze import analyze_audio
from .convert_voice import convert_voice
from .curate_vocal_slices import curate_vocal_slices
from .generate import generate_music
from .recognize_style import recognize_style
from .separate_stems import separate_stems
from .slice_audio import slice_audio

__all__ = [
    "analyze_audio",
    "convert_voice",
    "curate_vocal_slices",
    "generate_music",
    "recognize_style",
    "separate_stems",
    "slice_audio",
]
