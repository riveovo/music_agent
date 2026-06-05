from pathlib import Path
import json
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.agent.react import OpenAIReActUnavailable, ReActSession, run_react_agent
from music_agent.agent.router import route_request


class ReactAgentTests(unittest.TestCase):
    def test_react_loop_loads_skill_then_calls_external_tool(self) -> None:
        fake_client = _FakeOpenAIClient(
            [
                _function_call("call_1", "list_skills", {}),
                _function_call("call_2", "load_skill", {"name": "music-generation"}),
                _function_call("call_3", "music_generate", {"prompt": "lofi intro", "duration": 1}),
                {"output": [], "output_text": "Generated the clip."},
            ]
        )

        with mock.patch("music_agent.agent.react._openai_client", return_value=fake_client):
            with mock.patch(
                "music_agent.tools.music.generate_music",
                return_value={"capability": "generate", "output_audio": "outputs/generated.wav"},
            ) as mocked_generate:
                result = run_react_agent(
                    "生成一段 lofi intro",
                    {"request": "生成一段 lofi intro", "duration": 1, "provider": "synth"},
                    model="gpt-test",
                    max_steps=4,
                )

        self.assertEqual(result["engine"], "openai_react")
        self.assertEqual(result["skill_used"], "music-generation")
        self.assertEqual(result["result"]["output_audio"], "outputs/generated.wav")
        self.assertEqual(result["final_answer"], "Generated the clip.")
        mocked_generate.assert_called_once()
        self.assertIn("music_generate", fake_client.responses.tools_by_call[2])

    def test_route_request_auto_falls_back_without_openai_config(self) -> None:
        with mock.patch(
            "music_agent.agent.router.run_react_agent",
            side_effect=OpenAIReActUnavailable("missing key"),
        ):
            result = route_request("生成一段轻快电子音乐", duration=1, agent_engine="auto")

        self.assertEqual(result["engine"], "keyword")
        self.assertEqual(result["routed_to"], "generate")
        self.assertIn("fallback_reason", result)

    def test_react_session_preserves_context_between_turns(self) -> None:
        fake_client = _FakeOpenAIClient(
            [
                {"output": [], "output_text": "First answer."},
                {"output": [], "output_text": "Second answer."},
            ]
        )

        with mock.patch("music_agent.agent.react._openai_client", return_value=fake_client):
            session = ReActSession({"audio": "song.wav"}, model="gpt-test")
            first = session.ask("先分析这首歌")
            second = session.ask("再帮我分离人声")

        self.assertEqual(first["turn"], 1)
        self.assertEqual(second["turn"], 2)
        self.assertGreater(len(fake_client.responses.inputs_by_call[1]), len(fake_client.responses.inputs_by_call[0]))


def _function_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "output": [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
        ]
    }


class _FakeOpenAIClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = _FakeResponses(responses)


class _FakeResponses:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = responses
        self.tools_by_call: list[list[str]] = []
        self.inputs_by_call: list[list[object]] = []

    def create(self, **kwargs: object) -> dict[str, object]:
        self.tools_by_call.append([str(tool["name"]) for tool in kwargs["tools"]])
        self.inputs_by_call.append(list(kwargs["input"]))
        return self._responses.pop(0)
