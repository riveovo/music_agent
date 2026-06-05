from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from music_agent.errors import MusicAgentError
from music_agent.skills import SkillRegistry, build_skill_registry, parse_skill_file
from music_agent.tools import build_tool_registry


class SkillToolTests(unittest.TestCase):
    def test_builtin_music_tools_are_registered_with_openai_aliases(self) -> None:
        registry = build_tool_registry({"request": "生成音乐"})

        spec = registry.get("music.generate")

        self.assertEqual(spec.openai_name, "music_generate")
        self.assertIn("music.analyze_audio", registry.names())
        self.assertEqual(registry.get("music_generate").name, "music.generate")

    def test_parse_skill_file_reads_frontmatter_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "demo"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                """---
name: demo-skill
description: Use for demo work.
allowed_tools:
  - demo.tool
required_inputs: ["audio"]
---

# Demo

Call the external tool.
""",
                encoding="utf-8",
            )

            skill = parse_skill_file(skill_file)

        self.assertEqual(skill.name, "demo-skill")
        self.assertEqual(skill.allowed_tools, ("demo.tool",))
        self.assertEqual(skill.required_inputs, ("audio",))
        self.assertIn("Call the external tool", skill.body)

    def test_duplicate_skill_names_raise(self) -> None:
        registry = SkillRegistry()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SKILL.md"
            path.write_text(
                """---
name: duplicate
description: First.
allowed_tools: []
---
Body
""",
                encoding="utf-8",
            )
            skill = parse_skill_file(path)

        registry.register(skill)
        with self.assertRaisesRegex(MusicAgentError, "Duplicate skill name"):
            registry.register(skill)

    def test_missing_allowed_tool_raises(self) -> None:
        tools = build_tool_registry({"request": "test"})
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            skill_dir = skills_root / "bad"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: bad-skill
description: References a missing tool.
allowed_tools:
  - demo.missing
---
Body
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MusicAgentError, "missing external tool"):
                build_skill_registry(tools, skills_path=skills_root)
