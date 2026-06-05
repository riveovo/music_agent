"""Safe artifact registration and lookup for the web API."""

from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from ..errors import MusicAgentError
from ..paths import OUTPUT_ROOT


ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_ARTIFACT_EXTENSIONS = {
    ".aac",
    ".csv",
    ".flac",
    ".json",
    ".m4a",
    ".mp3",
    ".ncm",
    ".ogg",
    ".txt",
    ".wav",
}


@dataclass(frozen=True)
class Artifact:
    """A file that may be served through the web API."""

    id: str
    path: Path
    name: str
    kind: str
    mime_type: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "url": f"/api/artifacts/{self.id}",
        }


class ArtifactRegistry:
    """Maps local output files to opaque web artifact ids."""

    def __init__(self, *, root: str | Path = OUTPUT_ROOT) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, Artifact] = {}
        self._path_to_id: dict[Path, str] = {}

    def register_path(self, path: str | Path) -> Artifact:
        resolved = self._resolve_safe_path(path)
        if not resolved.is_file():
            raise MusicAgentError(f"Artifact path is not a file: {path}")
        existing_id = self._path_to_id.get(resolved)
        if existing_id is not None:
            return self._artifacts[existing_id]

        artifact_id = uuid4().hex
        artifact = Artifact(
            id=artifact_id,
            path=resolved,
            name=resolved.name,
            kind=_kind_for_path(resolved),
            mime_type=_mime_for_path(resolved),
            size_bytes=resolved.stat().st_size,
        )
        self._artifacts[artifact_id] = artifact
        self._path_to_id[resolved] = artifact_id
        return artifact

    def register_from_result(self, value: Any, *, max_files: int = 200) -> list[Artifact]:
        artifacts: list[Artifact] = []
        seen_ids: set[str] = set()

        def add(path: Path) -> None:
            if len(artifacts) >= max_files:
                return
            try:
                artifact = self.register_path(path)
            except MusicAgentError:
                return
            if artifact.id not in seen_ids:
                seen_ids.add(artifact.id)
                artifacts.append(artifact)

        def visit(item: Any) -> None:
            if len(artifacts) >= max_files:
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
                return
            if isinstance(item, (list, tuple, set)):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, str) or not _looks_like_path(item):
                return
            candidate = Path(item).expanduser()
            if candidate.exists() and candidate.is_dir():
                for child in sorted(candidate.rglob("*")):
                    if child.suffix.lower() in DEFAULT_ARTIFACT_EXTENSIONS and child.is_file():
                        add(child)
                return
            if candidate.exists() and candidate.is_file():
                add(candidate)

        visit(value)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        if not ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise MusicAgentError("Invalid artifact id.")
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise MusicAgentError("Unknown artifact id.") from exc

    def upload_path(self, *, session_id: str | None, filename: str) -> Path:
        bucket = session_id if session_id and ARTIFACT_ID_RE.fullmatch(session_id) else "global"
        stem = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", Path(filename).stem).strip("._-")
        suffix = Path(filename).suffix.lower()
        name = f"{stem or 'audio'}_{uuid4().hex[:8]}{suffix}"
        target_dir = self.root / "uploads" / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / name

    def _resolve_safe_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_relative_to(self.root):
            raise MusicAgentError("Artifact is outside the configured output root.")
        return resolved


def _looks_like_path(value: str) -> bool:
    if "\x00" in value or "\n" in value or len(value) > 512:
        return False
    if "://" in value:
        return False
    suffix = Path(value).suffix.lower()
    if suffix in DEFAULT_ARTIFACT_EXTENSIONS:
        return True
    return "/" in value or "\\" in value


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".aac", ".flac", ".m4a", ".mp3", ".ncm", ".ogg", ".wav"}:
        return "audio"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "table"
    return "file"


def _mime_for_path(path: Path) -> str:
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"
