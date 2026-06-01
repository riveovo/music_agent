"""SVCFusion-compatible voice conversion adapter.

SVCFusion's public repository exposes model classes rather than a stable pip
package or CLI entrypoint, so this module keeps the integration explicit:
callers must provide model files and either make the SVCFusion core importable
or pass a source path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable, Iterator
from uuid import uuid4

from ..errors import MusicAgentError


SVCFUSION_MODEL_TYPES = ("ddsp6_1",)
SVCFUSION_ENV = {
    "model_type": "MUSIC_AGENT_SVCFUSION_MODEL_TYPE",
    "model_path": "MUSIC_AGENT_SVCFUSION_MODEL_PATH",
    "config_path": "MUSIC_AGENT_SVCFUSION_CONFIG_PATH",
    "speaker": "MUSIC_AGENT_SVCFUSION_SPEAKER",
    "device": "MUSIC_AGENT_SVCFUSION_DEVICE",
    "source_path": "MUSIC_AGENT_SVCFUSION_SOURCE_PATH",
}


@dataclass(frozen=True)
class SVCFusionConfig:
    model_type: str
    model_path: Path
    speaker: str
    config_path: Path | None = None
    device: str | None = "auto"
    source_path: Path | None = None
    f0_method: str = "rmvpe"
    key_change: float = 0.0
    formant_shift_key: float = 0.0
    method: str = "auto"
    threshold: float = -60.0
    infer_step: str = "auto"
    t_start: str = "auto"
    vocal_register_factor: float = 1.0


@dataclass(frozen=True)
class SVCFusionOutput:
    output_audio: Path
    model_type: str
    model_path: Path
    config_path: Path | None
    speaker: str
    device: str
    source_path: Path | None
    parameters: dict[str, object]


def has_complete_svcfusion_env() -> bool:
    """Return whether env vars are enough to run SVCFusion conversion."""
    return bool(
        os.getenv(SVCFUSION_ENV["model_path"])
        and os.getenv(SVCFUSION_ENV["speaker"])
    )


def resolve_svcfusion_config(
    *,
    model_type: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    speaker: str | None = None,
    device: str | None = None,
    source_path: str | Path | None = None,
    f0_method: str = "rmvpe",
    key_change: float = 0.0,
    formant_shift_key: float = 0.0,
    method: str = "auto",
    threshold: float = -60.0,
    infer_step: str = "auto",
    t_start: str = "auto",
    vocal_register_factor: float = 1.0,
) -> SVCFusionConfig:
    """Resolve SVCFusion options from explicit args plus environment."""
    resolved_model_type = model_type or os.getenv(SVCFUSION_ENV["model_type"]) or "ddsp6_1"
    if resolved_model_type not in SVCFUSION_MODEL_TYPES:
        supported = ", ".join(SVCFUSION_MODEL_TYPES)
        raise MusicAgentError(
            f"Unsupported SVCFusion model_type '{resolved_model_type}'. Supported types: {supported}."
        )

    resolved_model_path = model_path or os.getenv(SVCFUSION_ENV["model_path"])
    resolved_speaker = speaker or os.getenv(SVCFUSION_ENV["speaker"])
    if not resolved_model_path or not resolved_speaker:
        raise MusicAgentError(
            "SVCFusion voice conversion requires model_path and speaker. "
            "Pass --svcfusion-model-path/--svcfusion-speaker or set "
            f"{SVCFUSION_ENV['model_path']}/{SVCFUSION_ENV['speaker']}."
        )

    model_file = Path(resolved_model_path).expanduser()
    if not model_file.exists():
        raise MusicAgentError(f"SVCFusion model file does not exist: {model_file}")
    if not model_file.is_file():
        raise MusicAgentError(f"SVCFusion model path is not a file: {model_file}")

    explicit_config = config_path or os.getenv(SVCFUSION_ENV["config_path"])
    config_file = Path(explicit_config).expanduser() if explicit_config else model_file.parent / "config.yaml"
    if not config_file.exists():
        raise MusicAgentError(
            f"SVCFusion config file does not exist: {config_file}. "
            "The upstream DDSP 6.1 loader expects a config.yaml next to the model, "
            "or pass --svcfusion-config-path."
        )
    if not config_file.is_file():
        raise MusicAgentError(f"SVCFusion config path is not a file: {config_file}")

    resolved_source_path = source_path or os.getenv(SVCFUSION_ENV["source_path"])
    source = Path(resolved_source_path).expanduser() if resolved_source_path else None
    if source is not None and not source.exists():
        raise MusicAgentError(f"SVCFusion source path does not exist: {source}")

    return SVCFusionConfig(
        model_type=resolved_model_type,
        model_path=model_file,
        config_path=config_file,
        speaker=resolved_speaker,
        device=device or os.getenv(SVCFUSION_ENV["device"]) or "auto",
        source_path=source,
        f0_method=f0_method,
        key_change=key_change,
        formant_shift_key=formant_shift_key,
        method=method,
        threshold=threshold,
        infer_step=infer_step,
        t_start=t_start,
        vocal_register_factor=vocal_register_factor,
    )


def convert_with_svcfusion(
    input_audio: str | Path,
    output_audio: str | Path,
    config: SVCFusionConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> SVCFusionOutput:
    """Run SVCFusion voice conversion and copy the result to output_audio."""
    input_path = Path(input_audio).expanduser()
    output_path = Path(output_audio).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)

    _report(progress, "SVCFusion: importing core modules")
    with _temporary_sys_path(_candidate_source_roots(config.source_path)):
        model_cls = _load_model_class(config.model_type)

        tmp_infer_dir = Path("tmp") / "infer_opt"
        tmp_infer_dir.mkdir(parents=True, exist_ok=True)
        with _staged_model_path(config.model_path, config.config_path) as cascade_model:
            model = model_cls()
            try:
                _report(progress, f"SVCFusion: loading model on {device}")
                speakers = model.load_model(
                    {
                        "device": device,
                        "cascade": str(cascade_model),
                    }
                )
                speaker_list = list(speakers or [])
                if config.speaker not in speaker_list:
                    available = ", ".join(str(item) for item in speaker_list) or "(none)"
                    raise MusicAgentError(
                        f"SVCFusion speaker '{config.speaker}' was not found in the model. "
                        f"Available speakers: {available}."
                    )

                params = _build_infer_params(input_path, config)
                _report(progress, f"SVCFusion: running inference for speaker '{config.speaker}'")
                generated = Path(model.infer(params, progress=_SVCFusionProgress(progress)))
                if not generated.exists():
                    raise MusicAgentError(f"SVCFusion did not produce an output file: {generated}")
                shutil.copy2(generated, output_path)
                _report(progress, f"SVCFusion: wrote {output_path.name}")
            finally:
                unload = getattr(model, "unload_model", None)
                if callable(unload):
                    unload()

    return SVCFusionOutput(
        output_audio=output_path,
        model_type=config.model_type,
        model_path=config.model_path,
        config_path=config.config_path,
        speaker=config.speaker,
        device=device,
        source_path=config.source_path,
        parameters={
            "f0_method": config.f0_method,
            "key_change": config.key_change,
            "formant_shift_key": config.formant_shift_key,
            "method": config.method,
            "threshold": config.threshold,
            "infer_step": config.infer_step,
            "t_start": config.t_start,
            "vocal_register_factor": config.vocal_register_factor,
        },
    )


def _load_model_class(model_type: str) -> type:
    if model_type == "ddsp6_1":
        try:
            module = importlib.import_module("SVCFusion.models.ddsp6_1")
        except ImportError as exc:
            raise MusicAgentError(
                "SVCFusion core modules are not importable. Install or checkout "
                "HuanLinOTO/SVCFusion and either add it to PYTHONPATH or pass "
                "--svcfusion-source-path /path/to/SVCFusion. The public upstream "
                "repository exposes core code, not a standalone pip package."
            ) from exc
        try:
            return getattr(module, "DDSP_6_1Model")
        except AttributeError as exc:
            raise MusicAgentError("SVCFusion core is missing DDSP_6_1Model.") from exc
    raise MusicAgentError(f"Unsupported SVCFusion model_type: {model_type}")


def _build_infer_params(input_path: Path, config: SVCFusionConfig) -> dict[str, object]:
    return {
        "num_formant_shift_key": config.formant_shift_key,
        "f0": config.f0_method,
        "audio": str(input_path),
        "keychange": config.key_change,
        "method": config.method,
        "threshold": config.threshold,
        "infer_step": config.infer_step,
        "t_start": config.t_start,
        "vocal_register_factor": config.vocal_register_factor,
        "spk": config.speaker,
        "hash": uuid4().hex,
    }


def _resolve_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _candidate_source_roots(source_path: Path | None) -> list[Path]:
    if source_path is None:
        return []
    source = source_path.expanduser()
    if (source / "SVCFusion").is_dir():
        return [source]
    if source.name == "SVCFusion" and (source / "models").is_dir():
        return [source.parent]
    return [source]


@contextmanager
def _temporary_sys_path(paths: list[Path]) -> Iterator[None]:
    added: list[str] = []
    for path in paths:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
            added.append(value)
    try:
        yield
    finally:
        for value in added:
            if value in sys.path:
                sys.path.remove(value)


@contextmanager
def _staged_model_path(model_path: Path, config_path: Path | None) -> Iterator[Path]:
    expected_config = model_path.parent / "config.yaml"
    if config_path is None or config_path.resolve() == expected_config.resolve():
        yield model_path
        return

    with tempfile.TemporaryDirectory(prefix="music_agent_svcfusion_") as tmp:
        staged_dir = Path(tmp)
        staged_model = staged_dir / model_path.name
        try:
            staged_model.symlink_to(model_path.resolve())
        except OSError as exc:
            raise MusicAgentError(
                "Could not stage SVCFusion model with a separate config path. "
                "Place config.yaml next to the model file and retry."
            ) from exc
        shutil.copy2(config_path, staged_dir / "config.yaml")
        yield staged_model


class _SVCFusionProgress:
    def __init__(self, callback: Callable[[str], None] | None) -> None:
        self._callback = callback

    def __call__(self, *args: object, **kwargs: object) -> None:
        if self._callback is None:
            return
        if args:
            self._callback(f"SVCFusion: {args[0]}")

    def tqdm(self, iterable: object = None, **kwargs: object) -> object:
        if iterable is None:
            return []
        return iterable


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
