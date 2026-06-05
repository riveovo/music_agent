import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.errors import MusicAgentError
from music_agent.web.artifacts import ArtifactRegistry


class WebArtifactTests(unittest.TestCase):
    def test_registers_only_files_inside_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            inside = root / "clip.wav"
            inside.write_bytes(b"RIFF")
            outside = Path(tmp) / "secret.wav"
            outside.write_bytes(b"RIFF")
            registry = ArtifactRegistry(root=root)

            artifact = registry.register_path(inside)

            self.assertEqual(artifact.kind, "audio")
            self.assertEqual(registry.get(artifact.id).path, inside.resolve())
            with self.assertRaises(MusicAgentError):
                registry.register_path(outside)
            with self.assertRaises(MusicAgentError):
                registry.get("../secret")

    def test_register_from_result_discovers_nested_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            nested = root / "generate"
            nested.mkdir(parents=True)
            audio = nested / "song.wav"
            sidecar = nested / "song.json"
            audio.write_bytes(b"RIFF")
            sidecar.write_text("{}", encoding="utf-8")
            registry = ArtifactRegistry(root=root)

            artifacts = registry.register_from_result({"result": {"output_dir": str(nested)}})

            self.assertEqual({artifact.name for artifact in artifacts}, {"song.wav", "song.json"})


if __name__ == "__main__":
    unittest.main()
