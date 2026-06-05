import sys
import unittest
from pathlib import Path
from io import StringIO
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.cli import _handle_chat, build_parser


class CliTests(unittest.TestCase):
    def test_curate_vocal_slices_accepts_args(self) -> None:
        args = build_parser().parse_args(
            [
                "curate-vocal-slices",
                "--input",
                "outputs/slices/singer",
                "--output-dir",
                "datasets/curated/singer",
                "--recursive",
                "--min-length-ms",
                "3000",
                "--max-length-ms",
                "10000",
                "--distance-threshold",
                "0.28",
                "--embedding-model",
                "speechbrain/spkrec-ecapa-voxceleb",
                "--model-cache-dir",
                "models/speechbrain/cache",
                "--device",
                "cpu",
                "--keep-converted",
                "--ncm-converter",
                "/opt/homebrew/bin/ncmdump",
            ]
        )

        self.assertEqual(args.input, "outputs/slices/singer")
        self.assertEqual(args.output_dir, "datasets/curated/singer")
        self.assertTrue(args.recursive)
        self.assertEqual(args.min_length_ms, 3000)
        self.assertEqual(args.max_length_ms, 10000)
        self.assertEqual(args.distance_threshold, 0.28)
        self.assertEqual(args.embedding_model, "speechbrain/spkrec-ecapa-voxceleb")
        self.assertEqual(args.model_cache_dir, "models/speechbrain/cache")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.keep_converted)
        self.assertEqual(args.ncm_converter, "/opt/homebrew/bin/ncmdump")

    def test_audio_input_commands_accept_batch_conversion_args(self) -> None:
        for command in ("analyze", "recognize-style", "separate-stems", "convert-voice"):
            with self.subTest(command=command):
                args = build_parser().parse_args(
                    [
                        command,
                        "--audio",
                        "datasets/raw",
                        "--output-dir",
                        "outputs/batch",
                        "--recursive",
                        "--keep-converted",
                        "--ncm-converter",
                        "/opt/homebrew/bin/ncmdump",
                    ]
                )

                self.assertEqual(args.audio, "datasets/raw")
                self.assertEqual(args.output_dir, "outputs/batch")
                self.assertTrue(args.recursive)
                self.assertTrue(args.keep_converted)
                self.assertEqual(args.ncm_converter, "/opt/homebrew/bin/ncmdump")

    def test_analyze_accepts_essentia_args(self) -> None:
        args = build_parser().parse_args(
            [
                "analyze",
                "--audio",
                "song.wav",
                "--provider",
                "essentia",
                "--essentia-max-sections",
                "10",
            ]
        )

        self.assertEqual(args.provider, "essentia")
        self.assertEqual(args.essentia_max_sections, 10)

    def test_recognize_style_accepts_essentia_args(self) -> None:
        args = build_parser().parse_args(
            [
                "recognize-style",
                "--audio",
                "song.wav",
                "--provider",
                "essentia",
                "--essentia-model-type",
                "discogs519_maest_30s",
                "--essentia-embedding-model-path",
                "embedding.pb",
                "--essentia-classifier-model-path",
                "classifier.pb",
                "--essentia-metadata-path",
                "metadata.json",
                "--essentia-top-k",
                "12",
            ]
        )

        self.assertEqual(args.provider, "essentia")
        self.assertEqual(args.essentia_model_type, "discogs519_maest_30s")
        self.assertEqual(args.essentia_embedding_model_path, "embedding.pb")
        self.assertEqual(args.essentia_classifier_model_path, "classifier.pb")
        self.assertEqual(args.essentia_metadata_path, "metadata.json")
        self.assertEqual(args.essentia_top_k, 12)

    def test_slice_audio_accepts_batch_and_conversion_args(self) -> None:
        args = build_parser().parse_args(
            [
                "slice-audio",
                "--input",
                "datasets/raw",
                "--output-dir",
                "outputs/slices",
                "--recursive",
                "--min-length-ms",
                "3000",
                "--max-length-ms",
                "10000",
                "--keep-converted",
                "--ncm-converter",
                "/opt/homebrew/bin/ncmdump",
            ]
        )

        self.assertEqual(args.input, "datasets/raw")
        self.assertEqual(args.output_dir, "outputs/slices")
        self.assertTrue(args.recursive)
        self.assertEqual(args.min_length_ms, 3000)
        self.assertEqual(args.max_length_ms, 10000)
        self.assertTrue(args.keep_converted)
        self.assertEqual(args.ncm_converter, "/opt/homebrew/bin/ncmdump")

    def test_separate_stems_accepts_msst_args(self) -> None:
        args = build_parser().parse_args(
            [
                "separate-stems",
                "--audio",
                "song.wav",
                "--provider",
                "msst",
                "--model-type",
                "bs_roformer",
                "--model-path",
                "model.ckpt",
                "--config-path",
                "model.yaml",
                "--device",
                "cpu",
                "--use-tta",
                "--instrumental-model-type",
                "mel_band_roformer",
                "--instrumental-model-path",
                "inst.ckpt",
                "--instrumental-config-path",
                "inst.yaml",
                "--deharmony-model-type",
                "mel_band_roformer",
                "--deharmony-model-path",
                "deharmony.ckpt",
                "--deharmony-config-path",
                "deharmony.yaml",
                "--dereverb-model-type",
                "mel_band_roformer",
                "--dereverb-model-path",
                "dereverb.ckpt",
                "--dereverb-config-path",
                "dereverb.yaml",
            ]
        )

        self.assertEqual(args.provider, "msst")
        self.assertEqual(args.model_type, "bs_roformer")
        self.assertEqual(args.model_path, "model.ckpt")
        self.assertEqual(args.config_path, "model.yaml")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.use_tta)
        self.assertEqual(args.instrumental_model_type, "mel_band_roformer")
        self.assertEqual(args.instrumental_model_path, "inst.ckpt")
        self.assertEqual(args.instrumental_config_path, "inst.yaml")
        self.assertEqual(args.deharmony_model_type, "mel_band_roformer")
        self.assertEqual(args.deharmony_model_path, "deharmony.ckpt")
        self.assertEqual(args.deharmony_config_path, "deharmony.yaml")
        self.assertEqual(args.dereverb_model_type, "mel_band_roformer")
        self.assertEqual(args.dereverb_model_path, "dereverb.ckpt")
        self.assertEqual(args.dereverb_config_path, "dereverb.yaml")

    def test_convert_voice_accepts_svcfusion_args(self) -> None:
        args = build_parser().parse_args(
            [
                "convert-voice",
                "--audio",
                "vocals.wav",
                "--provider",
                "svcfusion",
                "--output",
                "converted.wav",
                "--svcfusion-model-type",
                "ddsp6_1",
                "--svcfusion-model-path",
                "model.pt",
                "--svcfusion-config-path",
                "config.yaml",
                "--svcfusion-speaker",
                "target",
                "--svcfusion-device",
                "cpu",
                "--svcfusion-source-path",
                "/repo/SVCFusion",
                "--svcfusion-f0-method",
                "rmvpe",
                "--svcfusion-key-change",
                "2",
                "--svcfusion-formant-shift-key",
                "1",
                "--svcfusion-threshold",
                "-55",
            ]
        )

        self.assertEqual(args.provider, "svcfusion")
        self.assertEqual(args.svcfusion_model_type, "ddsp6_1")
        self.assertEqual(args.svcfusion_model_path, "model.pt")
        self.assertEqual(args.svcfusion_config_path, "config.yaml")
        self.assertEqual(args.svcfusion_speaker, "target")
        self.assertEqual(args.svcfusion_device, "cpu")
        self.assertEqual(args.svcfusion_source_path, "/repo/SVCFusion")
        self.assertEqual(args.svcfusion_key_change, 2.0)
        self.assertEqual(args.svcfusion_formant_shift_key, 1.0)
        self.assertEqual(args.svcfusion_threshold, -55.0)

    def test_agent_accepts_separation_args_without_colliding_with_generation_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "agent",
                "分离人声",
                "--audio",
                "song.wav",
                "--provider",
                "synth",
                "--separation-provider",
                "msst",
                "--separation-model-type",
                "mel_band_roformer",
                "--separation-model-path",
                "model.ckpt",
                "--separation-config-path",
                "model.yaml",
                "--separation-deharmony-model-path",
                "deharmony.ckpt",
                "--separation-deharmony-config-path",
                "deharmony.yaml",
            ]
        )

        self.assertEqual(args.provider, "synth")
        self.assertEqual(args.separation_provider, "msst")
        self.assertEqual(args.separation_model_type, "mel_band_roformer")
        self.assertEqual(args.separation_model_path, "model.ckpt")
        self.assertEqual(args.separation_config_path, "model.yaml")
        self.assertEqual(args.separation_deharmony_model_path, "deharmony.ckpt")
        self.assertEqual(args.separation_deharmony_config_path, "deharmony.yaml")

    def test_agent_accepts_voice_provider_args_without_colliding_with_generation_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "agent",
                "把这段人声换成目标音色",
                "--audio",
                "vocals.wav",
                "--provider",
                "synth",
                "--voice-provider",
                "svcfusion",
                "--voice-svcfusion-model-path",
                "model.pt",
                "--voice-svcfusion-config-path",
                "config.yaml",
                "--voice-svcfusion-speaker",
                "target",
            ]
        )

        self.assertEqual(args.provider, "synth")
        self.assertEqual(args.voice_provider, "svcfusion")
        self.assertEqual(args.voice_svcfusion_model_path, "model.pt")
        self.assertEqual(args.voice_svcfusion_config_path, "config.yaml")
        self.assertEqual(args.voice_svcfusion_speaker, "target")

    def test_agent_accepts_style_provider_args_without_colliding_with_generation_style(self) -> None:
        args = build_parser().parse_args(
            [
                "agent",
                "识别这首歌的风格",
                "--audio",
                "song.wav",
                "--style",
                "lofi",
                "--style-provider",
                "essentia",
                "--style-essentia-embedding-model-path",
                "embedding.pb",
                "--style-essentia-classifier-model-path",
                "classifier.pb",
                "--style-essentia-metadata-path",
                "metadata.json",
            ]
        )

        self.assertEqual(args.style, "lofi")
        self.assertEqual(args.style_provider, "essentia")
        self.assertEqual(args.style_essentia_embedding_model_path, "embedding.pb")
        self.assertEqual(args.style_essentia_classifier_model_path, "classifier.pb")
        self.assertEqual(args.style_essentia_metadata_path, "metadata.json")

    def test_agent_accepts_analysis_provider_args(self) -> None:
        args = build_parser().parse_args(
            [
                "agent",
                "分析这首歌",
                "--audio",
                "song.wav",
                "--analysis-provider",
                "essentia",
                "--analysis-essentia-max-sections",
                "9",
            ]
        )

        self.assertEqual(args.analysis_provider, "essentia")
        self.assertEqual(args.analysis_essentia_max_sections, 9)

    def test_agent_accepts_react_engine_args(self) -> None:
        args = build_parser().parse_args(
            [
                "agent",
                "生成一段音乐",
                "--agent-engine",
                "openai",
                "--openai-model",
                "gpt-test",
                "--skills-path",
                ".agents/skills",
                "--tools-path",
                ".agents/tools",
                "--max-steps",
                "4",
            ]
        )

        self.assertEqual(args.agent_engine, "openai")
        self.assertEqual(args.openai_model, "gpt-test")
        self.assertEqual(args.skills_path, ".agents/skills")
        self.assertEqual(args.tools_path, ".agents/tools")
        self.assertEqual(args.max_steps, 4)

    def test_chat_accepts_interactive_session_args(self) -> None:
        args = build_parser().parse_args(
            [
                "chat",
                "--agent-engine",
                "keyword",
                "--audio",
                "song.wav",
                "--analysis-provider",
                "essentia",
                "--max-steps",
                "3",
            ]
        )

        self.assertEqual(args.agent_engine, "keyword")
        self.assertEqual(args.audio, "song.wav")
        self.assertEqual(args.analysis_provider, "essentia")
        self.assertEqual(args.max_steps, 3)

    def test_web_accepts_service_and_agent_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "web",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--agent-engine",
                "keyword",
                "--openai-model",
                "gpt-test",
                "--skills-path",
                ".agents/skills",
                "--tools-path",
                ".agents/tools",
                "--max-steps",
                "5",
                "--audio",
                "song.wav",
            ]
        )

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        self.assertEqual(args.agent_engine, "keyword")
        self.assertEqual(args.openai_model, "gpt-test")
        self.assertEqual(args.skills_path, ".agents/skills")
        self.assertEqual(args.tools_path, ".agents/tools")
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(args.audio, "song.wav")

    def test_chat_keyword_session_reads_multiple_turns(self) -> None:
        args = build_parser().parse_args(["chat", "--agent-engine", "keyword", "--audio", "song.wav"])

        with mock.patch("sys.stdin", StringIO("分析这首歌\n/exit\n")):
            with mock.patch("sys.stdout", StringIO()):
                with mock.patch("sys.stderr", StringIO()):
                    with mock.patch("music_agent.cli.route_request", return_value={"engine": "keyword"}) as mocked:
                        result = _handle_chat(args)

        self.assertEqual(result["capability"], "chat")
        self.assertEqual(result["engine"], "keyword")
        self.assertEqual(result["turns"], 1)
        mocked.assert_called_once()
