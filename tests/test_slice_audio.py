from pathlib import Path
import math
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.audio_inputs import build_ncm_converter_command
from music_agent.capabilities.slice_audio import slice_audio
from music_agent.errors import MusicAgentError


def _require_slice_deps(testcase: unittest.TestCase) -> None:
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        testcase.skipTest(f"Missing audio slicing dependencies: {exc}")


def _write_test_wav(path: Path, *, sample_rate: int = 16000) -> None:
    segments = [
        ("tone", 0.35),
        ("silence", 0.25),
        ("tone", 0.35),
    ]
    samples: list[int] = []
    for kind, duration in segments:
        frame_count = round(sample_rate * duration)
        for index in range(frame_count):
            if kind == "silence":
                value = 0
            else:
                value = round(0.35 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            samples.append(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _write_continuous_wav(path: Path, *, duration: float = 1.2, sample_rate: int = 16000) -> None:
    frame_count = round(sample_rate * duration)
    samples = [
        round(0.35 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(frame_count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


class SliceAudioTests(unittest.TestCase):
    def test_slice_audio_creates_chunks_and_json(self) -> None:
        _require_slice_deps(self)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            _write_test_wav(audio)

            messages: list[str] = []
            result = slice_audio(
                audio,
                output_dir=tmp_path / "slices",
                min_length_ms=200,
                max_length_ms=500,
                progress=messages.append,
            )

            self.assertEqual(result["capability"], "slice_audio")
            self.assertEqual(result["mode"], "single")
            self.assertGreaterEqual(result["chunk_count"], 2)
            self.assertTrue(Path(str(result["result_json"])).exists())
            for chunk in result["chunks"]:
                self.assertTrue(Path(str(chunk["audio"])).exists())
                self.assertGreater(float(chunk["duration_seconds"]), 0)
            self.assertTrue(any("Audio slicing: detecting silent regions" in message for message in messages))
            self.assertIn("Audio slicing: wrote result JSON", messages)
            self.assertEqual(result["parameters"], {"min_length_ms": 200, "max_length_ms": 500})
            self.assertIn("db_threshold", result["effective_parameters"])
            self.assertIn("min_interval_ms", result["effective_parameters"])

    def test_slice_audio_splits_long_continuous_audio_by_max_length(self) -> None:
        _require_slice_deps(self)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "continuous.wav"
            _write_continuous_wav(audio, duration=1.2)

            result = slice_audio(
                audio,
                output_dir=tmp_path / "slices",
                min_length_ms=300,
                max_length_ms=500,
            )

            self.assertGreaterEqual(result["chunk_count"], 3)
            for chunk in result["chunks"]:
                self.assertLessEqual(float(chunk["duration_seconds"]), 0.5)
                self.assertGreaterEqual(float(chunk["duration_seconds"]), 0.29)

    def test_slice_audio_batch_processes_directory(self) -> None:
        _require_slice_deps(self)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_test_wav(tmp_path / "a.wav")
            _write_test_wav(tmp_path / "nested" / "b.wav")
            (tmp_path / "notes.txt").write_text("skip me", encoding="utf-8")

            result = slice_audio(
                tmp_path,
                output_dir=tmp_path / "out",
                recursive=True,
                min_length_ms=200,
                max_length_ms=500,
            )

            self.assertEqual(result["mode"], "batch")
            self.assertEqual(result["files_found"], 2)
            self.assertEqual(result["files_processed"], 2)
            self.assertEqual(result["files_failed"], 0)
            self.assertEqual(result["files_skipped"], 1)
            self.assertTrue(Path(str(result["result_json"])).exists())

    def test_mp3_input_uses_ffmpeg_conversion_path(self) -> None:
        _require_slice_deps(self)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_wav = tmp_path / "source.wav"
            fake_mp3 = tmp_path / "input.mp3"
            _write_test_wav(source_wav)
            fake_mp3.write_bytes(b"not really mp3")

            def fake_convert(ffmpeg: Path, source: Path, target: Path) -> None:
                target.write_bytes(source_wav.read_bytes())

            with mock.patch("music_agent.audio_inputs.find_executable", return_value=Path("/fake/ffmpeg")):
                with mock.patch("music_agent.audio_inputs.convert_with_ffmpeg", side_effect=fake_convert):
                    result = slice_audio(
                        fake_mp3,
                        output_dir=tmp_path / "slices",
                        min_length_ms=200,
                        max_length_ms=500,
                        keep_converted=True,
                    )

            self.assertTrue(result["conversion"]["required"])
            self.assertTrue(result["conversion"]["kept"])
            self.assertTrue(Path(str(result["conversion"]["converted_audio"])).exists())
            self.assertGreaterEqual(result["chunk_count"], 2)

    def test_ncm_without_converter_or_ncmdump_errors_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.ncm"
            audio.write_bytes(b"fake")

            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch("music_agent.audio_inputs.find_executable", return_value=None):
                    with self.assertRaisesRegex(MusicAgentError, "ncmdump"):
                        slice_audio(audio, output_dir=tmp_path / "slices")

    def test_ncm_converter_template_expands_paths(self) -> None:
        command = build_ncm_converter_command(
            "tool --in {input} --out {output} --dir {output_dir}",
            Path("/tmp/a song.ncm"),
            Path("/tmp/out"),
            Path("/tmp/out/a.wav"),
        )

        self.assertEqual(
            command,
            ["tool", "--in", "/tmp/a song.ncm", "--out", "/tmp/out/a.wav", "--dir", "/tmp/out"],
        )


if __name__ == "__main__":
    unittest.main()
