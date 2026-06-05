---
name: music-style-recognition
description: Use when the user asks about style, genre, mood, energy, 曲风, 风格, or 类型 of an audio file.
allowed_tools:
  - music.recognize_style
  - music.analyze_audio
required_inputs:
  - audio
---

# Music Style Recognition

Use this skill to answer questions about a track's genre, style, mood, and energy.

1. Prefer `music.recognize_style` for direct style or genre questions.
2. Use `music.analyze_audio` only when the user also asks for metadata, loudness, tempo, or structure.
3. Report style, confidence, mood, energy, evidence, and result artifact paths when available.
