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

from music_agent.capabilities.curate_vocal_slices import curate_vocal_slices


def _write_wav(path: Path, *, duration: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration * sample_rate)
    samples = [
        round(0.25 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
        for index in range(frame_count)
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _fake_embedding(model: object, audio_path: Path) -> object:
    import numpy as np

    if "target" in audio_path.stem:
        return np.array([1.0, 0.0], dtype="float32")
    return np.array([0.0, 1.0], dtype="float32")


class VocalCurationTests(unittest.TestCase):
    def test_curate_vocal_slices_selects_longest_duration_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "slices"
            _write_wav(input_dir / "target_01.wav", duration=0.45)
            _write_wav(input_dir / "target_02.wav", duration=0.40)
            _write_wav(input_dir / "target_03.wav", duration=0.35)
            _write_wav(input_dir / "guest_01.wav", duration=0.45)
            _write_wav(input_dir / "guest_02.wav", duration=0.30)

            with mock.patch("music_agent.capabilities.curate_vocal_slices._load_embedding_model", return_value=object()):
                with mock.patch("music_agent.capabilities.curate_vocal_slices._extract_embedding", side_effect=_fake_embedding):
                    result = curate_vocal_slices(
                        input_dir,
                        output_dir=tmp_path / "curated",
                        min_length_ms=100,
                        max_length_ms=1000,
                        distance_threshold=0.2,
                    )

            self.assertEqual(result["selection"], "longest_total_duration_cluster")
            self.assertEqual(result["decisions"]["accepted"], 3)
            self.assertEqual(result["decisions"]["rejected"], 2)
            self.assertEqual(result["decisions"]["review"], 0)
            self.assertTrue(Path(result["outputs"]["clusters_csv"]).exists())
            for item in result["items"]:
                self.assertTrue(Path(str(item["output_audio"])).exists())
                if "target" in Path(str(item["audio"])).stem:
                    self.assertEqual(item["decision"], "accepted")
                else:
                    self.assertEqual(item["decision"], "rejected")

    def test_curate_vocal_slices_routes_invalid_duration_to_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "slices"
            _write_wav(input_dir / "target_01.wav", duration=0.40)
            _write_wav(input_dir / "target_02.wav", duration=0.35)
            _write_wav(input_dir / "short_guest.wav", duration=0.03)

            with mock.patch("music_agent.capabilities.curate_vocal_slices._load_embedding_model", return_value=object()):
                with mock.patch("music_agent.capabilities.curate_vocal_slices._extract_embedding", side_effect=_fake_embedding):
                    result = curate_vocal_slices(
                        input_dir,
                        output_dir=tmp_path / "curated",
                        min_length_ms=100,
                        max_length_ms=1000,
                        distance_threshold=0.2,
                    )

            review_items = [item for item in result["items"] if item["decision"] == "review"]
            self.assertEqual(len(review_items), 1)
            self.assertEqual(review_items[0]["reason"], "shorter_than_min_length")


if __name__ == "__main__":
    unittest.main()
