from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.agent.router import _choose_route, route_request
from music_agent.errors import MusicAgentError


class AgentRouterTests(unittest.TestCase):
    def test_router_prioritizes_style_over_analysis(self) -> None:
        route, reason = _choose_route("分析这首歌的风格", audio="song.wav")

        self.assertEqual(route, "recognize_style")
        self.assertIn("风格", reason)

    def test_router_defaults_to_generation_without_audio(self) -> None:
        route, _ = _choose_route("做一段适合开场的音乐", audio=None)

        self.assertEqual(route, "generate")

    def test_agent_requires_audio_for_style_request(self) -> None:
        with self.assertRaisesRegex(MusicAgentError, "requires --audio"):
            route_request("识别这首歌的风格")

    def test_agent_can_generate(self) -> None:
        with _temp_dir() as tmp_path:
            output = tmp_path / "clip.wav"
            result = route_request("生成一段轻快电子音乐", duration=1, output=output)

            self.assertEqual(result["routed_to"], "generate")
            self.assertEqual(result["result"]["provider"], "synth")
            self.assertTrue(output.exists())


class _temp_dir:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()
