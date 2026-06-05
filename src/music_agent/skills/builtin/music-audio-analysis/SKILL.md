---
name: music-audio-analysis
description: Use when the user asks to analyze audio metadata, loudness, tempo, key, chords, sections, or musical structure.
allowed_tools:
  - music.analyze_audio
required_inputs:
  - audio
---

# Music Audio Analysis

Use this skill for audio inspection and music information retrieval.

1. Call `music.analyze_audio` with the provided audio path or CLI `--audio`.
2. Use provider defaults unless the user explicitly asks for a real MIR analysis, then prefer an Essentia provider if configured.
3. Summarize the most useful returned fields and include artifact paths.
