import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.agent.react import OpenAIReActUnavailable
from music_agent.web.sessions import SessionManager, WebConfig


class WebSessionTests(unittest.TestCase):
    def test_auto_engine_falls_back_to_keyword_when_openai_unavailable(self) -> None:
        manager = SessionManager(WebConfig(agent_engine="auto"))

        with mock.patch(
            "music_agent.web.sessions.ReActSession",
            side_effect=OpenAIReActUnavailable("OPENAI_API_KEY is not set."),
        ):
            session = manager.create_session()

        self.assertEqual(session.requested_engine, "auto")
        self.assertEqual(session.actual_engine, "keyword")
        self.assertIn("OPENAI_API_KEY", session.fallback_reason or "")

    def test_openai_engine_surfaces_unavailable_error(self) -> None:
        manager = SessionManager(WebConfig(agent_engine="openai"))

        with mock.patch(
            "music_agent.web.sessions.ReActSession",
            side_effect=OpenAIReActUnavailable("SDK missing."),
        ):
            with self.assertRaises(OpenAIReActUnavailable):
                manager.create_session()

    def test_keyword_session_filters_runtime_options_before_routing(self) -> None:
        manager = SessionManager(
            WebConfig(
                agent_engine="keyword",
                runtime_options={"duration": 3.0, "unknown_web_field": "ignored"},
            )
        )
        session = manager.create_session(runtime_options={"provider": "synth"})

        with mock.patch("music_agent.web.sessions.route_request", return_value={"engine": "keyword"}) as routed:
            result = session.ask("生成一段音乐", runtime_updates={"extra": "ignored", "audio": "song.wav"})

        self.assertEqual(result["engine"], "keyword")
        kwargs = routed.call_args.kwargs
        self.assertEqual(kwargs["duration"], 3.0)
        self.assertEqual(kwargs["provider"], "synth")
        self.assertEqual(kwargs["audio"], "song.wav")
        self.assertNotIn("unknown_web_field", kwargs)
        self.assertNotIn("extra", kwargs)


if __name__ == "__main__":
    unittest.main()
