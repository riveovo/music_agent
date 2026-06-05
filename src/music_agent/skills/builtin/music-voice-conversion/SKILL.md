---
name: music-voice-conversion
description: Use when the user asks to convert voice, change voice, 换声, 变声, or apply a vocal preset/model.
allowed_tools:
  - music.convert_voice
required_inputs:
  - audio
---

# Music Voice Conversion

Use this skill to transform a voice or vocal track.

1. Call `music.convert_voice` with the provided audio path or CLI `--audio`.
2. Use CLI defaults for preset, provider, SVCFusion model config, speaker, and output path.
3. Return the converted audio path and result artifact path.
