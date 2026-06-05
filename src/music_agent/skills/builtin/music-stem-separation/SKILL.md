---
name: music-stem-separation
description: Use when the user asks to separate stems, extract vocals, extract accompaniment, 分离人声, 提取人声, or make instrumental tracks.
allowed_tools:
  - music.separate_stems
required_inputs:
  - audio
---

# Music Stem Separation

Use this skill to split vocals and accompaniment.

1. Call `music.separate_stems` with the provided audio path or CLI `--audio`.
2. Use CLI defaults for provider, MSST model config, output directory, recursion, and conversion behavior.
3. Return the vocal and accompaniment paths and any separation JSON artifact.
