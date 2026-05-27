"""Local MusicGen provider backed by Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path

from ..audio import write_json
from ..errors import MusicAgentError
from ..paths import ensure_output_dir, slugify, timestamp
from .base import GenerationRequest
from .synth import detect_style


DEFAULT_MODEL = "facebook/musicgen-small"
DEFAULT_SAMPLE_RATE = 32_000
MAX_DURATION_SECONDS = 30
TOKENS_PER_SECOND = 50


class LocalMusicGenGenerator:
    """Generate music locally using `transformers` MusicGen."""

    name = "musicgen"

    def generate(self, request: GenerationRequest) -> dict[str, object]:
        _validate_request(request)
        deps = _load_dependencies()

        torch = deps["torch"]
        AutoProcessor = deps["AutoProcessor"]
        MusicgenForConditionalGeneration = deps["MusicgenForConditionalGeneration"]
        wavfile = deps["wavfile"]

        model_name = request.model or DEFAULT_MODEL
        output_path = request.output or _default_output_path(request.prompt, model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = _select_device(torch)
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            model.to(device)
        except Exception as exc:
            raise MusicAgentError(
                f"Could not load MusicGen model '{model_name}'. "
                "Check your network for the first download, Hugging Face access, and local disk space."
            ) from exc

        inputs = processor(text=[request.prompt], padding=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        max_new_tokens = max(1, int(request.duration * TOKENS_PER_SECOND))

        generation_kwargs: dict[str, object] = {
            "do_sample": True,
            "guidance_scale": request.guidance_scale,
            "max_new_tokens": max_new_tokens,
        }
        if request.seed is not None:
            torch.manual_seed(request.seed)

        try:
            with torch.no_grad():
                audio_values = model.generate(**inputs, **generation_kwargs)

            sampling_rate = int(getattr(model.config.audio_encoder, "sampling_rate", DEFAULT_SAMPLE_RATE))
            audio_array = audio_values[0, 0].detach().cpu().float().numpy()
            wavfile.write(str(output_path), rate=sampling_rate, data=audio_array)
        except Exception as exc:
            raise MusicAgentError(
                "MusicGen generation failed. Try a shorter --duration, a smaller --model, "
                "or run on a machine with more memory/GPU support."
            ) from exc

        result = {
            "capability": "generate",
            "provider": self.name,
            "quality": "local_musicgen",
            "prompt": request.prompt,
            "style": detect_style(request.prompt, request.style),
            "duration_seconds": round(request.duration, 3),
            "model": model_name,
            "device": device,
            "guidance_scale": request.guidance_scale,
            "max_new_tokens": max_new_tokens,
            "sample_rate": sampling_rate,
            "output_audio": str(output_path),
            "notes": "Generated locally with Hugging Face Transformers MusicGen.",
        }
        result_path = output_path.with_suffix(".json")
        write_json(result_path, result | {"result_json": str(result_path)})
        result["result_json"] = str(result_path)
        return result


def _validate_request(request: GenerationRequest) -> None:
    if request.duration <= 0:
        raise MusicAgentError("--duration must be greater than 0 seconds.")
    if request.duration > MAX_DURATION_SECONDS:
        raise MusicAgentError(
            f"MusicGen generation is capped at {MAX_DURATION_SECONDS} seconds. "
            "Use a shorter --duration or switch to --provider synth for longer mock clips."
        )
    if not request.prompt.strip():
        raise MusicAgentError("--prompt cannot be empty.")


def _load_dependencies() -> dict[str, object]:
    try:
        import torch
        from scipy.io import wavfile
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError as exc:
        raise MusicAgentError(
            "The local MusicGen provider needs optional ML dependencies. "
            "Install them with: python3.12 -m pip install -e '.[musicgen-local]'. "
            "You can keep using the free lightweight fallback with --provider synth."
        ) from exc
    return {
        "torch": torch,
        "wavfile": wavfile,
        "AutoProcessor": AutoProcessor,
        "MusicgenForConditionalGeneration": MusicgenForConditionalGeneration,
    }


def _select_device(torch: object) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_output_path(prompt: str, model_name: str) -> Path:
    model_slug = slugify(model_name.split("/")[-1], fallback="musicgen")
    name = f"generated_musicgen_{model_slug}_{slugify(prompt)}_{timestamp()}.wav"
    return ensure_output_dir("generate") / name
