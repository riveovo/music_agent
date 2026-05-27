"""Shared types for music generation providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-agnostic generation request."""

    prompt: str
    duration: float
    output: Path | None = None
    style: str | None = None
    model: str | None = None
    guidance_scale: float = 3.0
    seed: int | None = None


class MusicGenerator(Protocol):
    """Interface implemented by all generation providers."""

    name: str

    def generate(self, request: GenerationRequest) -> dict[str, object]:
        """Generate music and return a serializable result payload."""

