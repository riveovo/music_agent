from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

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
from music_agent.music_analysis import EssentiaAnalysisOutput
from music_agent.separation.msst import MSSTSeparationOutput, MSSTStageOutput, demix
from music_agent.style_recognition import EssentiaStyleOutput
from music_agent.voice_conversion import SVCFusionOutput


def _require_tools(testcase: unittest.TestCase, *tools: str) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        testcase.skipTest(f"Missing required tool(s): {', '.join(missing)}")


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
        _require_tools(self, "ffmpeg", "ffprobe")
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
        _require_tools(self, "ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("pop vocal placeholder", duration=1, output=audio)

            separated = separate_stems(audio, output_dir=tmp_path / "stems", provider="heuristic")
            converted = convert_voice(audio, preset="bright", provider="placeholder", output=tmp_path / "converted.wav")

            self.assertTrue(Path(separated["stems"]["vocals"]).exists())
            self.assertTrue(Path(separated["stems"]["accompaniment"]).exists())
            self.assertEqual(separated["quality"], "heuristic_mvp")
            self.assertTrue(Path(str(converted["output_audio"])).exists())
            self.assertEqual(converted["quality"], "placeholder_mvp")

    def test_analyze_audio_batch_processes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            generate_music("batch one", duration=1, output=input_dir / "one.wav")
            generate_music("batch two", duration=1, output=input_dir / "two.wav")
            (input_dir / "skip.txt").write_text("skip", encoding="utf-8")
            metadata = {
                "streams": [
                    {
                        "codec_type": "audio",
                        "duration": "1.0",
                        "channels": 1,
                        "sample_rate": "22050",
                        "codec_name": "pcm_s16le",
                    }
                ],
                "format": {"format_name": "wav", "bit_rate": "352800", "size": "1000"},
            }

            with mock.patch("music_agent.capabilities.analyze.ffprobe_json", return_value=metadata):
                with mock.patch("music_agent.capabilities.analyze._measure_loudness", return_value={"mean_db": -18.0, "max_db": -3.0}):
                    result = analyze_audio(input_dir, output_dir=tmp_path / "analysis", recursive=True)

            self.assertEqual(result["mode"], "batch")
            self.assertEqual(result["files_found"], 2)
            self.assertEqual(result["files_processed"], 2)
            self.assertEqual(result["files_skipped"], 3)
            self.assertTrue(Path(str(result["result_json"])).exists())

    def test_analyze_audio_essentia_provider_uses_backend_and_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            audio.write_bytes(b"fake wav")
            metadata = {
                "streams": [
                    {
                        "codec_type": "audio",
                        "duration": "64.0",
                        "channels": 2,
                        "sample_rate": "44100",
                        "codec_name": "pcm_s16le",
                    }
                ],
                "format": {"format_name": "wav", "bit_rate": "1411200", "size": "1000"},
            }
            messages: list[str] = []

            fake_output = EssentiaAnalysisOutput(
                tempo={"bpm": 128.0, "confidence": 0.82, "method": "essentia_music_extractor"},
                meter={"beats": [0.0, 0.5, 1.0, 1.5], "beats_count": 4, "downbeats": [0.0]},
                tonal={"key": "A", "scale": "minor", "key_strength": 0.7, "profile": "edma", "alternatives": []},
                chords={
                    "key": "A",
                    "scale": "minor",
                    "histogram": {"Am": 0.5},
                    "sequence": [{"start_seconds": 0.0, "end_seconds": 2.0, "chord": "Am"}],
                },
                spectral={"spectral_centroid_mean": 2200.0},
                sections=[
                    {"start_seconds": 0.0, "end_seconds": 32.0, "label": "A"},
                    {"start_seconds": 32.0, "end_seconds": 64.0, "label": "B"},
                ],
                descriptors={"rhythm": {"rhythm.danceability": 1.1}},
                extractor_version="test",
            )

            def fake_backend(
                audio_path: Path,
                config: object = None,
                progress: object = None,
            ) -> EssentiaAnalysisOutput:
                if progress is not None:
                    progress("fake Essentia analysis progress")
                return fake_output

            with mock.patch("music_agent.capabilities.analyze.ffprobe_json", return_value=metadata):
                with mock.patch("music_agent.capabilities.analyze._measure_loudness", return_value={"mean_db": -11.0, "max_db": -1.0}):
                    with mock.patch("music_agent.capabilities.analyze.analyze_with_essentia", side_effect=fake_backend):
                        result = analyze_audio(
                            audio,
                            output_dir=tmp_path / "analysis",
                            provider="essentia",
                            essentia_max_sections=8,
                            progress=messages.append,
                        )

            self.assertEqual(result["provider"], "essentia")
            self.assertEqual(result["quality"], "essentia_music_analysis")
            self.assertEqual(result["tempo"]["bpm"], 128.0)
            self.assertEqual(result["tonal"]["key"], "A")
            self.assertEqual(result["sections"][0]["label"], "A")
            self.assertEqual(result["summary"]["musical"]["structure"], "A-B")
            self.assertTrue(Path(str(result["result_json"])).exists())
            self.assertIn("fake Essentia analysis progress", messages)

    def test_analyze_audio_rejects_unknown_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "input.wav"
            audio.write_bytes(b"fake wav")

            with self.assertRaisesRegex(MusicAgentError, "Unknown analysis provider"):
                analyze_audio(audio, provider="missing")

    def test_real_essentia_analysis_integration_from_env(self) -> None:
        audio = os.getenv("MUSIC_AGENT_TEST_ANALYSIS_AUDIO")
        if not audio:
            self.skipTest("Set MUSIC_AGENT_TEST_ANALYSIS_AUDIO to run the real Essentia analysis integration test.")

        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_audio(
                audio,
                output_dir=Path(tmp) / "analysis",
                provider="essentia",
            )

            self.assertEqual(result["provider"], "essentia")
            self.assertEqual(result["quality"], "essentia_music_analysis")
            self.assertIn("tempo", result)
            self.assertIn("sections", result)

    def test_convert_voice_batch_processes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            generate_music("batch one", duration=1, output=input_dir / "one.wav")
            generate_music("batch two", duration=1, output=input_dir / "two.wav")

            def fake_run_tool(args: list[str]) -> object:
                Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(args[-1]).write_bytes(b"converted")
                return object()

            with mock.patch("music_agent.capabilities.convert_voice.require_tool", return_value="/fake/ffmpeg"):
                with mock.patch("music_agent.capabilities.convert_voice.run_tool", side_effect=fake_run_tool):
                    result = convert_voice(
                        input_dir,
                        provider="placeholder",
                        output_dir=tmp_path / "converted",
                        recursive=True,
                    )

            self.assertEqual(result["mode"], "batch")
            self.assertEqual(result["files_found"], 2)
            self.assertEqual(result["files_processed"], 2)
            for item in result["results"]:
                self.assertTrue(Path(str(item["output_audio"])).exists())

    def test_separate_stems_batch_processes_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            input_dir.mkdir()
            generate_music("batch one", duration=1, output=input_dir / "one.wav")
            generate_music("batch two", duration=1, output=input_dir / "two.wav")

            def fake_run_tool(args: list[str]) -> object:
                Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(args[-1]).write_bytes(b"stem")
                return object()

            with mock.patch("music_agent.capabilities.separate_stems.require_tool", return_value="/fake/ffmpeg"):
                with mock.patch("music_agent.capabilities.separate_stems.run_tool", side_effect=fake_run_tool):
                    result = separate_stems(
                        input_dir,
                        output_dir=tmp_path / "stems",
                        provider="heuristic",
                        recursive=True,
                    )

            self.assertEqual(result["mode"], "batch")
            self.assertEqual(result["files_found"], 2)
            self.assertEqual(result["files_processed"], 2)
            for item in result["results"]:
                self.assertTrue(Path(str(item["stems"]["vocals"])).exists())
                self.assertTrue(Path(str(item["stems"]["accompaniment"])).exists())

    def test_convert_voice_rejects_unknown_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("test", duration=1, output=audio)

            with self.assertRaisesRegex(MusicAgentError, "Unknown voice preset"):
                convert_voice(audio, preset="unknown", provider="placeholder")

    def test_convert_voice_svcfusion_provider_uses_backend_and_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            model = tmp_path / "model.pt"
            config = tmp_path / "config.yaml"
            output = tmp_path / "converted.wav"
            model.write_bytes(b"fake")
            config.write_text("model: fake\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)
            messages: list[str] = []

            def fake_backend(
                input_audio: Path,
                output_audio: Path,
                resolved_config: object,
                progress: object = None,
            ) -> SVCFusionOutput:
                if progress is not None:
                    progress("fake SVCFusion progress")
                Path(output_audio).write_bytes(b"converted")
                return SVCFusionOutput(
                    output_audio=Path(output_audio),
                    model_type=resolved_config.model_type,
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                    speaker=resolved_config.speaker,
                    device="cpu",
                    source_path=resolved_config.source_path,
                    parameters={"f0_method": resolved_config.f0_method},
                )

            with mock.patch("music_agent.capabilities.convert_voice.convert_with_svcfusion", side_effect=fake_backend):
                result = convert_voice(
                    audio,
                    provider="svcfusion",
                    output=output,
                    svcfusion_model_path=model,
                    svcfusion_config_path=config,
                    svcfusion_speaker="target",
                    svcfusion_source_path=tmp_path,
                    progress=messages.append,
                )

            self.assertEqual(result["provider"], "svcfusion")
            self.assertEqual(result["quality"], "svcfusion_ddsp6_1")
            self.assertEqual(result["svcfusion"]["speaker"], "target")
            self.assertTrue(output.exists())
            self.assertTrue(Path(str(result["result_json"])).exists())
            self.assertIn("fake SVCFusion progress", messages)

    def test_convert_voice_svcfusion_provider_requires_model_and_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("test", duration=1, output=audio)

            with self.assertRaisesRegex(MusicAgentError, "model_path and speaker"):
                convert_voice(audio, provider="svcfusion")

    def test_recognize_style_essentia_provider_uses_backend_and_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            embedding = tmp_path / "embedding.pb"
            classifier = tmp_path / "classifier.pb"
            metadata = tmp_path / "metadata.json"
            audio.write_bytes(b"fake wav")
            embedding.write_bytes(b"fake")
            classifier.write_bytes(b"fake")
            metadata.write_text('{"classes": ["Electronic---House", "Rock---Indie Rock"]}', encoding="utf-8")
            messages: list[str] = []

            fake_analysis = {
                "audio": str(audio),
                "duration_seconds": 42.0,
                "channels": 2,
                "loudness": {"mean_db": -12.0},
                "conversion": {"required": False},
            }

            def fake_backend(
                audio_path: Path,
                resolved_config: object,
                progress: object = None,
            ) -> EssentiaStyleOutput:
                if progress is not None:
                    progress("fake Essentia progress")
                return EssentiaStyleOutput(
                    style="electronic",
                    confidence=0.91,
                    top_styles=[{"style": "electronic", "score": 0.91, "evidence": ["Electronic---House"]}],
                    raw_tags=[{"label": "Electronic---House", "score": 0.91}],
                    segments=[{"index": 1, "start_seconds": 0.0, "end_seconds": 30.0}],
                    labels_count=2,
                    model_type=resolved_config.model_type,
                    embedding_model_path=resolved_config.embedding_model_path,
                    classifier_model_path=resolved_config.classifier_model_path,
                    metadata_path=resolved_config.metadata_path,
                )

            with mock.patch("music_agent.capabilities.recognize_style.analyze_audio", return_value=fake_analysis):
                with mock.patch("music_agent.capabilities.recognize_style.recognize_style_with_essentia", side_effect=fake_backend):
                    result = recognize_style(
                        audio,
                        output_dir=tmp_path / "style",
                        provider="essentia",
                        essentia_embedding_model_path=embedding,
                        essentia_classifier_model_path=classifier,
                        essentia_metadata_path=metadata,
                        progress=messages.append,
                    )

            self.assertEqual(result["provider"], "essentia")
            self.assertEqual(result["quality"], "essentia_discogs_maest")
            self.assertEqual(result["style"], "electronic")
            self.assertEqual(result["confidence"], 0.91)
            self.assertEqual(result["top_styles"][0]["style"], "electronic")
            self.assertTrue(Path(str(result["result_json"])).exists())
            self.assertIn("fake Essentia progress", messages)

    def test_recognize_style_essentia_provider_requires_complete_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "input.wav"
            audio.write_bytes(b"fake wav")

            with self.assertRaisesRegex(MusicAgentError, "embedding_model_path, classifier_model_path, and metadata_path"):
                recognize_style(audio, provider="essentia")

    def test_real_essentia_style_integration_from_env(self) -> None:
        audio = os.getenv("MUSIC_AGENT_TEST_STYLE_AUDIO")
        embedding = os.getenv("MUSIC_AGENT_TEST_STYLE_ESSENTIA_EMBEDDING_MODEL_PATH")
        classifier = os.getenv("MUSIC_AGENT_TEST_STYLE_ESSENTIA_CLASSIFIER_MODEL_PATH")
        metadata = os.getenv("MUSIC_AGENT_TEST_STYLE_ESSENTIA_METADATA_PATH")
        if not all([audio, embedding, classifier, metadata]):
            self.skipTest("Set MUSIC_AGENT_TEST_STYLE_* env vars to run the real Essentia style integration test.")

        with tempfile.TemporaryDirectory() as tmp:
            result = recognize_style(
                audio,
                output_dir=Path(tmp) / "style",
                provider="essentia",
                essentia_embedding_model_path=embedding,
                essentia_classifier_model_path=classifier,
                essentia_metadata_path=metadata,
            )

            self.assertEqual(result["provider"], "essentia")
            self.assertEqual(result["quality"], "essentia_discogs_maest")
            self.assertTrue(result["raw_tags"])

    def test_msst_provider_requires_complete_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            generate_music("test", duration=1, output=audio)

            with self.assertRaisesRegex(MusicAgentError, "model_type, model_path, and config_path"):
                separate_stems(audio, output_dir=tmp_path / "stems", provider="msst")

    def test_msst_provider_rejects_missing_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            config = tmp_path / "model.yaml"
            config.write_text("audio: {}\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)

            with self.assertRaisesRegex(MusicAgentError, "MSST model file does not exist"):
                separate_stems(
                    audio,
                    output_dir=tmp_path / "stems",
                    provider="msst",
                    model_type="bs_roformer",
                    model_path=tmp_path / "missing.ckpt",
                    config_path=config,
                )

    def test_msst_provider_uses_backend_and_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            model = tmp_path / "model.ckpt"
            config = tmp_path / "model.yaml"
            model.write_bytes(b"fake")
            config.write_text("audio: {}\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)

            def fake_backend(
                audio_path: Path,
                output_dir: Path,
                resolved_config: object,
                progress: object = None,
            ) -> MSSTSeparationOutput:
                output_dir.mkdir(parents=True, exist_ok=True)
                vocals = output_dir / "vocals.wav"
                accompaniment = output_dir / "accompaniment.wav"
                vocals.write_bytes(b"vocals")
                accompaniment.write_bytes(b"accompaniment")
                return MSSTSeparationOutput(
                    vocals=vocals,
                    accompaniment=accompaniment,
                    sample_rate=44100,
                    device="cpu",
                    model_type="bs_roformer",
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                    source_stems=("vocals", "instrumental"),
                )

            with mock.patch("music_agent.capabilities.separate_stems.separate_with_msst", side_effect=fake_backend):
                result = separate_stems(
                    audio,
                    output_dir=tmp_path / "stems",
                    provider="msst",
                    model_type="bs_roformer",
                    model_path=model,
                    config_path=config,
                )

            self.assertEqual(result["provider"], "msst")
            self.assertEqual(result["quality"], "msst_roformer")
            self.assertEqual(result["source_stems"], ["instrumental", "vocals"])
            self.assertTrue(Path(result["stems"]["vocals"]).exists())
            self.assertTrue(Path(result["stems"]["accompaniment"]).exists())
            self.assertTrue(Path(str(result["result_json"])).exists())

    def test_auto_provider_uses_env_msst_config_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            model = tmp_path / "model.ckpt"
            config = tmp_path / "model.yaml"
            model.write_bytes(b"fake")
            config.write_text("audio: {}\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)

            def fake_backend(
                audio_path: Path,
                output_dir: Path,
                resolved_config: object,
                progress: object = None,
            ) -> MSSTSeparationOutput:
                output_dir.mkdir(parents=True, exist_ok=True)
                vocals = output_dir / "vocals.wav"
                accompaniment = output_dir / "accompaniment.wav"
                vocals.write_bytes(b"vocals")
                accompaniment.write_bytes(b"accompaniment")
                return MSSTSeparationOutput(
                    vocals=vocals,
                    accompaniment=accompaniment,
                    sample_rate=44100,
                    device="cpu",
                    model_type=resolved_config.model_type,
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                    source_stems=("vocals",),
                )

            env = {
                "MUSIC_AGENT_MSST_MODEL_TYPE": "bs_roformer",
                "MUSIC_AGENT_MSST_MODEL_PATH": str(model),
                "MUSIC_AGENT_MSST_CONFIG_PATH": str(config),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("music_agent.capabilities.separate_stems.separate_with_msst", side_effect=fake_backend):
                    result = separate_stems(audio, output_dir=tmp_path / "stems")

            self.assertEqual(result["provider"], "msst")

    def test_msst_provider_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            model = tmp_path / "model.ckpt"
            config = tmp_path / "model.yaml"
            model.write_bytes(b"fake")
            config.write_text("audio: {}\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)
            messages: list[str] = []

            def fake_backend(
                audio_path: Path,
                output_dir: Path,
                resolved_config: object,
                progress: object = None,
            ) -> MSSTSeparationOutput:
                if progress is not None:
                    progress("fake backend progress")
                output_dir.mkdir(parents=True, exist_ok=True)
                vocals = output_dir / "vocals.wav"
                accompaniment = output_dir / "accompaniment.wav"
                vocals.write_bytes(b"vocals")
                accompaniment.write_bytes(b"accompaniment")
                return MSSTSeparationOutput(
                    vocals=vocals,
                    accompaniment=accompaniment,
                    sample_rate=44100,
                    device="cpu",
                    model_type=resolved_config.model_type,
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                    source_stems=("vocals",),
                )

            with mock.patch("music_agent.capabilities.separate_stems.separate_with_msst", side_effect=fake_backend):
                separate_stems(
                    audio,
                    output_dir=tmp_path / "stems",
                    provider="msst",
                    model_type="bs_roformer",
                    model_path=model,
                    config_path=config,
                    progress=messages.append,
                )

            self.assertIn("fake backend progress", messages)
            self.assertIn("Separation: wrote result JSON", messages)

    def test_msst_provider_runs_optional_refinement_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "input.wav"
            model = tmp_path / "model.ckpt"
            config = tmp_path / "model.yaml"
            instrumental_model = tmp_path / "instrumental.ckpt"
            instrumental_config = tmp_path / "instrumental.yaml"
            deharmony_model = tmp_path / "deharmony.ckpt"
            deharmony_config = tmp_path / "deharmony.yaml"
            dereverb_model = tmp_path / "dereverb.ckpt"
            dereverb_config = tmp_path / "dereverb.yaml"
            for path in (model, instrumental_model, deharmony_model, dereverb_model):
                path.write_bytes(b"fake")
            for path in (config, instrumental_config, deharmony_config, dereverb_config):
                path.write_text("audio: {}\n", encoding="utf-8")
            generate_music("test", duration=1, output=audio)
            stage_names: list[str] = []

            def fake_primary(
                audio_path: Path,
                output_dir: Path,
                resolved_config: object,
                progress: object = None,
            ) -> MSSTSeparationOutput:
                output_dir.mkdir(parents=True, exist_ok=True)
                vocals = output_dir / "vocals.wav"
                accompaniment = output_dir / "accompaniment.wav"
                vocals.write_bytes(b"primary vocals")
                accompaniment.write_bytes(b"primary accompaniment")
                return MSSTSeparationOutput(
                    vocals=vocals,
                    accompaniment=accompaniment,
                    sample_rate=44100,
                    device="cpu",
                    model_type=resolved_config.model_type,
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                    source_stems=("vocals", "instrumental"),
                )

            def fake_stage(
                audio_path: Path,
                output_audio: Path,
                resolved_config: object,
                preferred_stems: tuple[str, ...],
                stage_name: str,
                progress: object = None,
            ) -> MSSTStageOutput:
                stage_names.append(stage_name)
                output_audio.parent.mkdir(parents=True, exist_ok=True)
                output_audio.write_bytes(stage_name.encode("utf-8"))
                selected = preferred_stems[0]
                return MSSTStageOutput(
                    output_audio=output_audio,
                    selected_stem=selected,
                    source_stems=(selected, "other"),
                    sample_rate=44100,
                    device="cpu",
                    model_type=resolved_config.model_type,
                    model_path=resolved_config.model_path,
                    config_path=resolved_config.config_path,
                )

            with mock.patch("music_agent.capabilities.separate_stems.separate_with_msst", side_effect=fake_primary):
                with mock.patch("music_agent.capabilities.separate_stems.run_msst_stage", side_effect=fake_stage):
                    result = separate_stems(
                        audio,
                        output_dir=tmp_path / "stems",
                        provider="msst",
                        model_type="bs_roformer",
                        model_path=model,
                        config_path=config,
                        instrumental_model_type="mel_band_roformer",
                        instrumental_model_path=instrumental_model,
                        instrumental_config_path=instrumental_config,
                        deharmony_model_type="mel_band_roformer",
                        deharmony_model_path=deharmony_model,
                        deharmony_config_path=deharmony_config,
                        dereverb_model_type="mel_band_roformer",
                        dereverb_model_path=dereverb_model,
                        dereverb_config_path=dereverb_config,
                    )

            self.assertEqual(stage_names, ["instrumental refinement", "deharmony", "dereverb"])
            self.assertEqual([stage["stage"] for stage in result["postprocess"]], ["instrumental_refinement", "deharmony", "dereverb"])
            self.assertTrue(Path(result["artifacts"]["primary_vocals"]).exists())
            self.assertTrue(Path(result["artifacts"]["deharmonized_vocals"]).exists())
            self.assertTrue(Path(result["artifacts"]["dereverbed_vocals"]).exists())
            self.assertEqual(Path(result["stems"]["vocals"]).read_bytes(), b"dereverb")
            self.assertEqual(Path(result["stems"]["accompaniment"]).read_bytes(), b"instrumental refinement")

    def test_demix_with_fake_roformer_model(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"Missing torch/numpy for demix unit test: {exc}")

        class Node(dict):
            def __getattr__(self, key: str) -> object:
                return self[key]

        class FakeModel(torch.nn.Module):
            def forward(self, value: object) -> object:
                return value.unsqueeze(1) * 0.25

        config = Node(
            audio=Node(chunk_size=32),
            inference=Node(batch_size=2, num_overlap=2),
            training=Node(instruments=["vocals", "instrumental"], target_instrument="vocals", use_amp=False),
        )
        mix = np.ones((2, 64), dtype=np.float32)

        result = demix(
            config,
            FakeModel(),
            mix,
            "cpu",
            model_type="bs_roformer",
            torch_module=torch,
            np_module=np,
        )

        self.assertEqual(set(result), {"vocals"})
        self.assertEqual(result["vocals"].shape, mix.shape)
        self.assertTrue(np.allclose(result["vocals"], 0.25))

    def test_real_msst_integration_from_env(self) -> None:
        audio = os.getenv("MUSIC_AGENT_TEST_MSST_AUDIO")
        model_type = os.getenv("MUSIC_AGENT_TEST_MSST_MODEL_TYPE")
        model_path = os.getenv("MUSIC_AGENT_TEST_MSST_MODEL_PATH")
        config_path = os.getenv("MUSIC_AGENT_TEST_MSST_CONFIG_PATH")
        if not all([audio, model_type, model_path, config_path]):
            self.skipTest("Set MUSIC_AGENT_TEST_MSST_* env vars to run the real MSST integration test.")

        with tempfile.TemporaryDirectory() as tmp:
            result = separate_stems(
                audio,
                output_dir=Path(tmp) / "stems",
                provider="msst",
                model_type=model_type,
                model_path=model_path,
                config_path=config_path,
                device=os.getenv("MUSIC_AGENT_TEST_MSST_DEVICE", "auto"),
            )

            self.assertEqual(result["provider"], "msst")
            self.assertTrue(Path(result["stems"]["vocals"]).exists())
            self.assertTrue(Path(result["stems"]["accompaniment"]).exists())
