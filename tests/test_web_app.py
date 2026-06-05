import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from fastapi.testclient import TestClient

    from music_agent.web.app import create_app
    from music_agent.web.sessions import WebConfig
except Exception:  # pragma: no cover - optional web dependency gate.
    TestClient = None
    create_app = None
    WebConfig = None


@unittest.skipIf(TestClient is None, "FastAPI web dependencies are not installed.")
class WebAppTests(unittest.TestCase):
    def test_session_message_and_artifact_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            generated = output_root / "generate" / "clip.wav"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"RIFF")
            app = create_app(WebConfig(agent_engine="keyword", output_root=output_root))
            client = TestClient(app)

            session = client.post("/api/sessions", json={"agent_engine": "keyword"}).json()["session"]
            with mock.patch(
                "music_agent.web.sessions.route_request",
                return_value={"engine": "keyword", "routed_to": "generate", "result": {"output_audio": str(generated)}},
            ):
                response = client.post(f"/api/sessions/{session['id']}/messages", json={"content": "生成一段音乐"})

            payload = response.json()
            self.assertTrue(payload["ok"])
            artifacts = payload["data"]["artifacts"]
            self.assertEqual(artifacts[0]["name"], "clip.wav")
            download = client.get(artifacts[0]["url"])
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.content, b"RIFF")

    def test_stream_message_emits_artifact_and_final_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            generated = output_root / "generate" / "clip.wav"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"RIFF")
            app = create_app(WebConfig(agent_engine="keyword", output_root=output_root))
            client = TestClient(app)
            session = client.post("/api/sessions", json={"agent_engine": "keyword"}).json()["session"]

            with mock.patch(
                "music_agent.web.sessions.route_request",
                return_value={"engine": "keyword", "routed_to": "generate", "result": {"output_audio": str(generated)}},
            ):
                response = client.post(f"/api/sessions/{session['id']}/messages/stream", json={"content": "生成一段音乐"})

            self.assertEqual(response.status_code, 200)
            self.assertIn("event: artifact", response.text)
            self.assertIn("event: final", response.text)

    def test_upload_audio_validation_and_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            app = create_app(WebConfig(agent_engine="keyword", output_root=output_root))
            client = TestClient(app)
            session = client.post("/api/sessions", json={"agent_engine": "keyword"}).json()["session"]

            rejected = client.post(
                "/api/uploads/audio",
                data={"session_id": session["id"]},
                files={"file": ("notes.txt", b"not audio", "text/plain")},
            )
            self.assertEqual(rejected.status_code, 400)

            uploaded = client.post(
                "/api/uploads/audio",
                data={"session_id": session["id"]},
                files={"file": ("voice.wav", b"RIFF", "audio/wav")},
            )
            payload = uploaded.json()
            self.assertTrue(payload["ok"])
            download = client.get(payload["artifact"]["url"])
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.content, b"RIFF")


if __name__ == "__main__":
    unittest.main()
