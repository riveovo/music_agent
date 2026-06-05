---
name: music-vocal-slice-curation
description: Use when the user asks to curate, clean, filter, cluster, 筛选, 清洗, or keep target singer vocal slices.
allowed_tools:
  - music.curate_vocal_slices
required_inputs:
  - audio
---

# Music Vocal Slice Curation

Use this skill to select the likely target singer cluster from vocal slices.

1. Call `music.curate_vocal_slices` with the provided slice path or CLI `--audio`.
2. Use CLI defaults for output directory, clustering threshold, embedding model, and device.
3. Return accepted, rejected, review counts, output directories, CSV, and result JSON paths.
