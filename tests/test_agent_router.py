from pathlib import Path
import sys
import unittest
from unittest import mock

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

    def test_agent_passes_slice_audio_args(self) -> None:
        with mock.patch("music_agent.agent.router.slice_audio", return_value={"capability": "slice_audio"}) as mocked:
            result = route_request(
                "把这批音频切片",
                audio="datasets/raw",
                slice_output_dir="outputs/slices",
                slice_recursive=True,
                slice_min_length_ms=3000,
                slice_max_length_ms=10000,
                slice_keep_converted=True,
                slice_ncm_converter="/opt/homebrew/bin/ncmdump",
            )

        self.assertEqual(result["routed_to"], "slice_audio")
        mocked.assert_called_once_with(
            "datasets/raw",
            output_dir="outputs/slices",
            recursive=True,
            min_length_ms=3000,
            max_length_ms=10000,
            keep_converted=True,
            ncm_converter="/opt/homebrew/bin/ncmdump",
            progress=None,
        )

    def test_agent_passes_vocal_curation_args(self) -> None:
        with mock.patch("music_agent.agent.router.curate_vocal_slices", return_value={"capability": "curate_vocal_slices"}) as mocked:
            result = route_request(
                "清洗目标歌手切片",
                audio="outputs/slices/singer",
                audio_output_dir="datasets/curated/singer",
                audio_recursive=True,
                curation_min_length_ms=3000,
                curation_max_length_ms=10000,
                curation_distance_threshold=0.28,
                curation_embedding_model="speechbrain/spkrec-ecapa-voxceleb",
                curation_model_cache_dir="models/speechbrain/cache",
                curation_device="cpu",
            )

        self.assertEqual(result["routed_to"], "curate_vocal_slices")
        mocked.assert_called_once_with(
            "outputs/slices/singer",
            output_dir="datasets/curated/singer",
            recursive=True,
            min_length_ms=3000,
            max_length_ms=10000,
            distance_threshold=0.28,
            embedding_model="speechbrain/spkrec-ecapa-voxceleb",
            model_cache_dir="models/speechbrain/cache",
            device="cpu",
            keep_converted=False,
            ncm_converter=None,
            progress=None,
        )

    def test_agent_passes_separation_provider_args(self) -> None:
        with mock.patch("music_agent.agent.router.separate_stems", return_value={"provider": "msst"}) as mocked:
            result = route_request(
                "分离人声和伴奏",
                audio="song.wav",
                separation_provider="msst",
                separation_model_type="bs_roformer",
                separation_model_path="model.ckpt",
                separation_config_path="model.yaml",
                separation_device="cpu",
                separation_use_tta=True,
            )

        self.assertEqual(result["routed_to"], "separate_stems")
        self.assertEqual(result["result"]["provider"], "msst")
        mocked.assert_called_once_with(
            "song.wav",
            output_dir=None,
            provider="msst",
            model_type="bs_roformer",
            model_path="model.ckpt",
            config_path="model.yaml",
            device="cpu",
            use_tta=True,
            instrumental_model_type=None,
            instrumental_model_path=None,
            instrumental_config_path=None,
            deharmony_model_type=None,
            deharmony_model_path=None,
            deharmony_config_path=None,
            dereverb_model_type=None,
            dereverb_model_path=None,
            dereverb_config_path=None,
            recursive=False,
            keep_converted=False,
            ncm_converter=None,
            progress=None,
        )

    def test_agent_passes_voice_conversion_provider_args(self) -> None:
        with mock.patch("music_agent.agent.router.convert_voice", return_value={"provider": "svcfusion"}) as mocked:
            result = route_request(
                "把这段人声换声",
                audio="vocals.wav",
                output="converted.wav",
                voice_provider="svcfusion",
                voice_svcfusion_model_type="ddsp6_1",
                voice_svcfusion_model_path="model.pt",
                voice_svcfusion_config_path="config.yaml",
                voice_svcfusion_speaker="target",
                voice_svcfusion_device="cpu",
                voice_svcfusion_source_path="/repo/SVCFusion",
                voice_svcfusion_key_change=2.0,
                voice_svcfusion_threshold=-55.0,
            )

        self.assertEqual(result["routed_to"], "convert_voice")
        self.assertEqual(result["result"]["provider"], "svcfusion")
        mocked.assert_called_once_with(
            "vocals.wav",
            preset="bright",
            output="converted.wav",
            provider="svcfusion",
            output_dir=None,
            recursive=False,
            keep_converted=False,
            ncm_converter=None,
            svcfusion_model_type="ddsp6_1",
            svcfusion_model_path="model.pt",
            svcfusion_config_path="config.yaml",
            svcfusion_speaker="target",
            svcfusion_device="cpu",
            svcfusion_source_path="/repo/SVCFusion",
            svcfusion_f0_method="rmvpe",
            svcfusion_key_change=2.0,
            svcfusion_formant_shift_key=0.0,
            svcfusion_method="auto",
            svcfusion_threshold=-55.0,
            svcfusion_infer_step="auto",
            svcfusion_t_start="auto",
            svcfusion_vocal_register_factor=1.0,
            progress=None,
        )


class _temp_dir:
    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()
