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
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems --audio outputs/generate/example.wav
PYTHONPATH=src python3.12 -m music_agent.cli convert-voice --audio outputs/generate/example.wav --preset bright
PYTHONPATH=src python3.12 -m music_agent.cli agent "分析这首歌的风格" --audio outputs/generate/example.wav
```

All commands print JSON. Audio and JSON artifacts are written under
`outputs/`.

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
