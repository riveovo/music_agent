"""Standard SKILL.md discovery for the music agent."""

from .core import (
    SKILLS_PATH_ENV,
    Skill,
    SkillRegistry,
    build_skill_registry,
    parse_skill_file,
)

__all__ = [
    "SKILLS_PATH_ENV",
    "Skill",
    "SkillRegistry",
    "build_skill_registry",
    "parse_skill_file",
]
