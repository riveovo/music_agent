# AI Music Agent MVP

CLI-first MVP for an AI music agent. Each music capability can run on its own,
and the `agent` command routes natural-language requests to those capabilities.

## Requirements

- Python 3.12
- `ffmpeg` and `ffprobe` on `PATH` for audio analysis and transforms

The default runtime uses only the Python standard library plus system `ffmpeg`.
The local MusicGen provider is optional and installs heavier ML dependencies.

## Layout

Source code lives under `src/music_agent/`. For no-install local runs, prefix
commands with `PYTHONPATH=src`. After an editable install, the `music-agent`
console command is also available.

## Commands

```bash
PYTHONPATH=src python3.12 -m music_agent.cli generate --prompt "轻快电子音乐" --duration 5
PYTHONPATH=src python3.12 -m music_agent.cli generate --provider musicgen --prompt "lofi hip hop with warm piano" --duration 10
PYTHONPATH=src python3.12 -m music_agent.cli recognize-style --audio outputs/generate/example.wav
PYTHONPATH=src python3.12 -m music_agent.cli analyze --audio outputs/generate/example.wav
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems --audio outputs/generate/example.wav --provider heuristic
PYTHONPATH=src python3.12 -m music_agent.cli convert-voice --audio outputs/generate/example.wav --preset bright
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio --input outputs/generate/example.wav
PYTHONPATH=src python3.12 -m music_agent.cli agent "分析这首歌的风格" --audio outputs/generate/example.wav
```

All commands print JSON. Audio and JSON artifacts are written under
`outputs/`.

## Shared Audio Input Behavior

Audio-input capabilities accept WAV, MP3, FLAC, and NCM inputs:

- `analyze`
- `recognize-style`
- `separate-stems`
- `convert-voice`
- `slice-audio`

For all of these commands:

- WAV files are processed directly.
- MP3 and FLAC files are converted to temporary WAV files with `ffmpeg`.
- NCM files are decrypted with `ncmdump` first, then converted to WAV with
  `ffmpeg`. The command checks common Homebrew paths such as
  `/opt/homebrew/bin`.
- Use `--keep-converted` to keep intermediate converted WAV files in the output
  directory. By default, converted WAVs are temporary.
- Use `--ncm-converter /path/to/ncmdump` or `MUSIC_AGENT_NCM_CONVERTER` if your
  converter is not discoverable.

Directory batch processing uses the same input rules:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems \
  --audio datasets/raw_songs \
  --output-dir outputs/stems/batch_001 \
  --provider msst \
  --model-type bs_roformer \
  --model-path models/model.ckpt \
  --config-path models/model.yaml \
  --recursive
```

`--recursive` means subfolders are processed too. Batch outputs keep one
subdirectory per source audio file.

## Real Music Generation

The default `--provider synth` is free, local, and dependency-free. It is useful
for testing the Agent pipeline, but it is not a neural music model.

For real local text-to-music generation, install the optional MusicGen provider:

```bash
python3.12 -m pip install -e ".[musicgen-local]"
```

Then run:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli generate \
  --provider musicgen \
  --model facebook/musicgen-small \
  --prompt "80s pop track with bassy drums and synth" \
  --duration 10 \
  --guidance-scale 3
```

Notes:

- First run downloads model weights from Hugging Face.
- `musicgen` is capped at 30 seconds in this CLI.
- CPU works but can be slow; Apple Silicon MPS or CUDA is preferred.
- The model provider is free to run locally, but you are responsible for model
  license terms before commercial use.

## Real Vocal/Accompaniment Separation

The default `separate-stems --provider auto` uses the MSST-compatible backend
only when a complete model configuration is provided. Otherwise it falls back
to the lightweight ffmpeg heuristic.

Install the optional RoFormer separation dependencies:

```bash
python3.12 -m pip install -e ".[separation-msst]"
```

Then provide a compatible BS-RoFormer or MelBand-RoFormer vocal model:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems \
  --audio song.wav \
  --provider msst \
  --model-type bs_roformer \
  --model-path /path/to/model.ckpt \
  --config-path /path/to/model.yaml \
  --device auto
```

You can add optional refinement stages when you have matching models:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems \
  --audio song.wav \
  --provider msst \
  --model-type bs_roformer \
  --model-path /path/to/vocal_model.ckpt \
  --config-path /path/to/vocal_model.yaml \
  --instrumental-model-type mel_band_roformer \
  --instrumental-model-path /path/to/instrumental_model.ckpt \
  --instrumental-config-path /path/to/instrumental_model.yaml \
  --deharmony-model-type mel_band_roformer \
  --deharmony-model-path /path/to/deharmony_model.ckpt \
  --deharmony-config-path /path/to/deharmony_model.yaml \
  --dereverb-model-type mel_band_roformer \
  --dereverb-model-path /path/to/dereverb_model.ckpt \
  --dereverb-config-path /path/to/dereverb_model.yaml
```

- The instrumental stage rewrites `accompaniment.wav` from an accompaniment
  focused model, which can reduce leftover vocal bleed.
- The deharmony stage writes `vocals_deharmonized.wav` and feeds it to the next
  vocal cleanup step.
- The dereverb/de-echo stage writes `vocals_dereverbed.wav`; final
  `vocals.wav` is replaced with the last cleaned vocal output.

You can also set:

```bash
export MUSIC_AGENT_MSST_MODEL_TYPE=bs_roformer
export MUSIC_AGENT_MSST_MODEL_PATH=/path/to/model.ckpt
export MUSIC_AGENT_MSST_CONFIG_PATH=/path/to/model.yaml
export MUSIC_AGENT_MSST_DEVICE=auto
export MUSIC_AGENT_MSST_INSTRUMENTAL_MODEL_TYPE=mel_band_roformer
export MUSIC_AGENT_MSST_INSTRUMENTAL_MODEL_PATH=/path/to/instrumental_model.ckpt
export MUSIC_AGENT_MSST_INSTRUMENTAL_CONFIG_PATH=/path/to/instrumental_model.yaml
export MUSIC_AGENT_MSST_DEHARMONY_MODEL_TYPE=mel_band_roformer
export MUSIC_AGENT_MSST_DEHARMONY_MODEL_PATH=/path/to/deharmony_model.ckpt
export MUSIC_AGENT_MSST_DEHARMONY_CONFIG_PATH=/path/to/deharmony_model.yaml
export MUSIC_AGENT_MSST_DEREVERB_MODEL_TYPE=mel_band_roformer
export MUSIC_AGENT_MSST_DEREVERB_MODEL_PATH=/path/to/dereverb_model.ckpt
export MUSIC_AGENT_MSST_DEREVERB_CONFIG_PATH=/path/to/dereverb_model.yaml
```

The backend writes `vocals.wav`, `accompaniment.wav`, and `separation.json`.
Long-running separation progress is printed to stderr; stdout remains the final
machine-readable JSON result.
Model weights are not downloaded automatically and are not stored in this repo.
The in-project RoFormer inference code is a minimal MSST-WebUI-derived subset;
see `THIRD_PARTY_NOTICES.md` and `MSST_AGPL_LICENSE.txt`.

## Real Voice Conversion

`convert-voice` keeps the old lightweight ffmpeg preset as a fallback, but can
also run a real SVCFusion-compatible DDSP 6.1 model when you provide a model and
target speaker.

Install the local dependencies:

```bash
python3.12 -m pip install -e ".[voice-svcfusion]"
```

This extra contains the common SVCFusion/DDSP inference dependencies used by
the adapter. Some model configs can still require extra encoder-specific
packages from the upstream SVCFusion environment.

Make the SVCFusion core repository importable, either with `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/SVCFusion:$PYTHONPATH
```

or per command:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli convert-voice \
  --provider svcfusion \
  --audio outputs/stems/song/vocals.wav \
  --output outputs/convert_voice/song_target.wav \
  --svcfusion-source-path /path/to/SVCFusion \
  --svcfusion-model-type ddsp6_1 \
  --svcfusion-model-path models/svcfusion/target/model.pt \
  --svcfusion-config-path models/svcfusion/target/config.yaml \
  --svcfusion-speaker target_speaker \
  --svcfusion-device auto
```

Batch conversion uses the same flags:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli convert-voice \
  --provider svcfusion \
  --audio outputs/slices/target_singer \
  --output-dir outputs/convert_voice/target_batch \
  --recursive \
  --svcfusion-source-path /path/to/SVCFusion \
  --svcfusion-model-path models/svcfusion/target/model.pt \
  --svcfusion-config-path models/svcfusion/target/config.yaml \
  --svcfusion-speaker target_speaker
```

Useful environment variables:

```bash
export MUSIC_AGENT_SVCFUSION_MODEL_TYPE=ddsp6_1
export MUSIC_AGENT_SVCFUSION_MODEL_PATH=models/svcfusion/target/model.pt
export MUSIC_AGENT_SVCFUSION_CONFIG_PATH=models/svcfusion/target/config.yaml
export MUSIC_AGENT_SVCFUSION_SPEAKER=target_speaker
export MUSIC_AGENT_SVCFUSION_DEVICE=auto
export MUSIC_AGENT_SVCFUSION_SOURCE_PATH=/path/to/SVCFusion
```

With those set, `--provider auto` will choose SVCFusion. Without a complete
SVCFusion config, `auto` falls back to the placeholder preset.

The upstream SVCFusion public repository says it contains the core code but not
the frontend entrypoint, so this project treats it as an explicitly supplied
external core rather than a vendored dependency. See `THIRD_PARTY_NOTICES.md`.

## Vocal Slicing and Batch Audio Conversion

Install the lightweight slicing dependencies:

```bash
python3.12 -m pip install -e ".[audio-slice]"
```

Slice one WAV/MP3/FLAC/NCM file:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio \
  --input song.mp3 \
  --output-dir outputs/slices/song \
  --min-length-ms 3000 \
  --max-length-ms 10000
```

Batch process a folder:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio \
  --input datasets/raw_songs \
  --output-dir outputs/slices/batch_001 \
  --recursive
```

- WAV files are sliced directly.
- MP3, FLAC, and NCM follow the shared conversion behavior above.
- Use `--keep-converted` when you want to keep the intermediate WAV files next
  to the slice outputs.

The slicing implementation follows openvpi/audio-slicer's RMS silence detection
idea, but the command line only asks for the target clip length range. The
silence threshold and minimum quiet interval are estimated from the audio's RMS
distribution, then long clips are split near their quietest internal point so
outputs stay close to `--min-length-ms` and `--max-length-ms`. Progress is
printed to stderr; stdout remains the final JSON result.

## Vocal Slice Curation

When you have many dry vocal slices from mostly one singer, use
`curate-vocal-slices` to cluster singer embeddings and keep the cluster with
the longest total slice duration.

Install the optional free local curation stack:

```bash
python3.12 -m pip install -e ".[vocal-curation]"
```

Then run:

```bash
PYTHONPATH=src python3.12 -m music_agent.cli curate-vocal-slices \
  --input outputs/slices/target_singer \
  --output-dir datasets/curated/target_singer \
  --min-length-ms 3000 \
  --max-length-ms 10000 \
  --distance-threshold 0.32
```

Outputs:

```text
datasets/curated/target_singer/
  accepted/
  rejected/
  review/
  curation.json
  clusters.csv
```

The default embedding model is `speechbrain/spkrec-ecapa-voxceleb`, a free
Apache-2.0 SpeechBrain ECAPA-TDNN speaker embedding model. First run downloads
it into `models/speechbrain/`, which is ignored by git. Clustering uses
cosine-distance agglomerative clustering, then selects the cluster whose slices
have the largest total duration.

For a stricter split, lower `--distance-threshold`; for a looser split, raise
it slightly. A practical first sweep is `0.28` to `0.36`.

## Tests

```bash
PYTHONPATH=src python3.12 -m pytest
```

The tests are written with the standard `unittest` API, so they can also run
without installing anything:

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests
```

If `pytest` is not installed and you want the exact pytest command, install the
dev extra in a virtual environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```
