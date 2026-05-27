from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.capabilities import (
    analyze_audio,
    convert_voice,
    generate_music,
    recognize_style,
    separate_stems,
)
from music_agent.errors import MusicAgentError


class CapabilityTests(unittest.TestCase):
    def test_generate_music_creates_wav_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "generated_electronic_test.wav"

            result = generate_music("轻快电子音乐", duration=1, output=output)

            self.assertEqual(result["capability"], "generate")
            self.assertEqual(result["provider"], "synth")
            self.assertEqual(result["style"], "electronic")
            self.assertTrue(output.exists())
            self.assertTrue(Path(str(result["result_json"])).exists())

    def test_generate_rejects_invalid_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(MusicAgentError, "duration"):
                generate_music("test", duration=0, output=Path(tmp) / "bad.wav")

    def test_generate_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(MusicAgentError, "Unknown generation provider"):
            generate_music("test", duration=1, provider="missing")

    def test_musicgen_provider_caps_duration_before_loading_dependencies(self) -> None:
        with self.assertRaisesRegex(MusicAgentError, "MusicGen generation is capped"):
            generate_music("test", duration=31, provider="musicgen")

    def test_analyze_and_recognize_generated_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "generated_electronic_test.wav"
            generate_music("electronic test", duration=1, output=audio)

            analysis = analyze_audio(audio)
            style = recognize_style(audio)

            self.assertEqual(analysis["capability"], "analyze")
            self.assertAlmostEqual(float(analysis["duration_seconds"]), 1, delta=0.1)
            self.assertEqual(style["style"], "electronic")
            self.assertEqual(style["quality"], "heuristic_mvp")

    def test_separate_and_convert_voice_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("pop vocal placeholder", duration=1, output=audio)

            separated = separate_stems(audio, output_dir=tmp_path / "stems")
            converted = convert_voice(audio, preset="bright", output=tmp_path / "converted.wav")

            self.assertTrue(Path(separated["stems"]["vocals"]).exists())
            self.assertTrue(Path(separated["stems"]["accompaniment"]).exists())
            self.assertEqual(separated["quality"], "heuristic_mvp")
            self.assertTrue(Path(str(converted["output_audio"])).exists())
            self.assertEqual(converted["quality"], "placeholder_mvp")

    def test_convert_voice_rejects_unknown_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("test", duration=1, output=audio)

            with self.assertRaisesRegex(MusicAgentError, "Unknown voice preset"):
                convert_voice(audio, preset="unknown")
