---
name: music-generation
description: Use when the user wants to generate, compose, make, or sketch a new music clip from natural language.
allowed_tools:
  - music.generate
required_inputs:
  - prompt
---

# Music Generation

Use this skill to turn a natural-language music idea into an audio clip.

1. Use `music.generate` exactly once unless the user asks for multiple variants.
2. Use the user's request as the prompt unless they provide a clearer prompt argument.
3. Prefer CLI defaults for duration, provider, model, style, seed, and output path when present.
4. Return the generated audio path, provider, style, and result JSON path when available.
