"""Path helpers for runtime artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def ensure_output_dir(kind: str) -> Path:
    """Create and return the output directory for a capability."""
    output_dir = OUTPUT_ROOT / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str, fallback: str = "music") -> str:
    """Make a short ASCII-ish slug while keeping readable CJK characters."""
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug or fallback)[:48]
