---
name: music-audio-slicing
description: Use when the user asks to slice, split, segment, 切片, 切分, or 分割 audio into smaller clips.
allowed_tools:
  - music.slice_audio
required_inputs:
  - audio
---

# Music Audio Slicing

Use this skill to create clip slices from audio files or directories.

1. Call `music.slice_audio` with the provided input path or CLI `--audio`.
2. Use CLI defaults for output directory, recursion, min/max length, and conversion behavior.
3. Return slice counts, output directory, and result JSON path.
