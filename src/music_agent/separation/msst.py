"""MSST-compatible RoFormer source separation backend.

This module keeps the music-agent runtime independent from any local MSST
checkout. The RoFormer model definitions live in this package, while model
weights and YAML configs are supplied explicitly by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from ..errors import MusicAgentError


SUPPORTED_MODEL_TYPES = ("bs_roformer", "mel_band_roformer")
INSTALL_HINT = "Install optional dependencies with: python -m pip install -e '.[separation-msst]'."

ENV_MODEL_TYPE = "MUSIC_AGENT_MSST_MODEL_TYPE"
ENV_MODEL_PATH = "MUSIC_AGENT_MSST_MODEL_PATH"
ENV_CONFIG_PATH = "MUSIC_AGENT_MSST_CONFIG_PATH"
ENV_DEVICE = "MUSIC_AGENT_MSST_DEVICE"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class MSSTSeparationConfig:
    """Resolved settings for an MSST-style RoFormer separation run."""

    model_type: str
    model_path: Path
    config_path: Path
    device: str = "auto"
    use_tta: bool = False


@dataclass(frozen=True)
class MSSTSeparationOutput:
    """Files and metadata produced by an MSST separation run."""

    vocals: Path
    accompaniment: Path
    sample_rate: int
    device: str
    model_type: str
    model_path: Path
    config_path: Path
    source_stems: tuple[str, ...]


@dataclass(frozen=True)
class MSSTStageOutput:
    """Selected stem produced by one RoFormer processing stage."""

    output_audio: Path
    selected_stem: str
    source_stems: tuple[str, ...]
    sample_rate: int
    device: str
    model_type: str
    model_path: Path
    config_path: Path


def resolve_msst_config(
    *,
    model_type: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | None = None,
    use_tta: bool = False,
    require_complete: bool = False,
    env_prefix: str | None = None,
    label: str = "MSST separation",
) -> MSSTSeparationConfig | None:
    """Resolve explicit/env MSST settings.

    Returns None only when no MSST settings are present and the caller did not
    require a complete config. Partial settings are treated as user intent and
    therefore raise a recoverable error.
    """
    env_model_type = _stage_env_name(env_prefix, "MODEL_TYPE")
    env_model_path = _stage_env_name(env_prefix, "MODEL_PATH")
    env_config_path = _stage_env_name(env_prefix, "CONFIG_PATH")
    env_device = _stage_env_name(env_prefix, "DEVICE")

    raw_model_type = _coalesce(model_type, os.getenv(env_model_type), os.getenv(ENV_MODEL_TYPE) if env_prefix is None else None)
    raw_model_path = _coalesce(model_path, os.getenv(env_model_path), os.getenv(ENV_MODEL_PATH) if env_prefix is None else None)
    raw_config_path = _coalesce(config_path, os.getenv(env_config_path), os.getenv(ENV_CONFIG_PATH) if env_prefix is None else None)
    raw_device = _coalesce(device, os.getenv(env_device), os.getenv(ENV_DEVICE), "auto")

    has_any = any(value not in (None, "") for value in (raw_model_type, raw_model_path, raw_config_path))
    if not has_any and not require_complete:
        return None

    missing = [
        name
        for name, value in (
            ("model_type", raw_model_type),
            ("model_path", raw_model_path),
            ("config_path", raw_config_path),
        )
        if value in (None, "")
    ]
    if missing:
        env_hint = ", ".join([env_model_type, env_model_path, env_config_path])
        raise MusicAgentError(
            f"{label} requires model_type, model_path, and config_path. "
            f"Missing: {', '.join(missing)}. You can pass CLI args or set {env_hint}."
        )

    resolved_model_type = str(raw_model_type).strip().lower()
    if resolved_model_type not in SUPPORTED_MODEL_TYPES:
        available = ", ".join(SUPPORTED_MODEL_TYPES)
        raise MusicAgentError(
            f"Unsupported {label} model_type '{raw_model_type}'. "
            f"This project currently supports: {available}."
        )

    resolved_model_path = _resolve_existing_file(raw_model_path, "MSST model")
    resolved_config_path = _resolve_existing_file(raw_config_path, "MSST config")
    resolved_device = str(raw_device or "auto").strip().lower()
    if resolved_device not in {"auto", "cpu", "cuda", "mps"}:
        raise MusicAgentError("MSST device must be one of: auto, cpu, cuda, mps.")

    return MSSTSeparationConfig(
        model_type=resolved_model_type,
        model_path=resolved_model_path,
        config_path=resolved_config_path,
        device=resolved_device,
        use_tta=use_tta,
    )


def resolve_msst_stage_config(
    stage: str,
    *,
    model_type: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | None = None,
    use_tta: bool = False,
    require_complete: bool = False,
) -> MSSTSeparationConfig | None:
    """Resolve optional MSST settings for a named post-processing stage."""
    return resolve_msst_config(
        model_type=model_type,
        model_path=model_path,
        config_path=config_path,
        device=device,
        use_tta=use_tta,
        require_complete=require_complete,
        env_prefix=stage,
        label=f"MSST {stage.lower()} stage",
    )


def separate_with_msst(
    audio_path: str | Path,
    output_dir: str | Path,
    config: MSSTSeparationConfig,
    progress: ProgressCallback | None = None,
) -> MSSTSeparationOutput:
    """Run MSST-compatible RoFormer separation and write canonical stem files."""
    _report(progress, "MSST: loading Python dependencies")
    deps = _load_runtime_dependencies()
    np = deps["np"]
    librosa = deps["librosa"]
    sf = deps["sf"]

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    _report(progress, "MSST: loading model configuration and checkpoint")
    separator = MSSTRoFormerSeparator(config, deps=deps, progress=progress)
    sample_rate = int(_cfg_get(separator.model_config.audio, "sample_rate", 44100))
    try:
        _report(progress, f"MSST: loading audio at {sample_rate} Hz")
        mix, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    except Exception as exc:
        raise MusicAgentError(f"Could not load audio for MSST separation: {audio_path}") from exc

    try:
        _report(progress, "MSST: running source separation")
        stems = separator.separate(mix)
    finally:
        separator.del_cache()

    vocals = _stem_by_name(stems, ("vocals", "vocal"))
    if vocals is None:
        found = ", ".join(sorted(stems)) or "none"
        raise MusicAgentError(f"MSST model did not produce a vocals stem. Stems found: {found}.")

    accompaniment = _stem_by_name(stems, ("instrumental", "accompaniment", "other", "no_vocals"))
    if accompaniment is None:
        _report(progress, "MSST: deriving accompaniment from mixture minus vocals")
        accompaniment = _subtract_stem(_mix_to_samples_channels(np.asarray(mix)), np.asarray(vocals), np)

    vocals_path = target_dir / "vocals.wav"
    accompaniment_path = target_dir / "accompaniment.wav"
    _report(progress, "MSST: writing stem audio files")
    sf.write(str(vocals_path), _sanitize_audio(vocals, np), sample_rate, subtype="FLOAT")
    sf.write(str(accompaniment_path), _sanitize_audio(accompaniment, np), sample_rate, subtype="FLOAT")
    _report(progress, "MSST: separation complete")

    return MSSTSeparationOutput(
        vocals=vocals_path,
        accompaniment=accompaniment_path,
        sample_rate=sample_rate,
        device=separator.device,
        model_type=config.model_type,
        model_path=config.model_path,
        config_path=config.config_path,
        source_stems=tuple(sorted(stems)),
    )


def run_msst_stage(
    audio_path: str | Path,
    output_audio: str | Path,
    config: MSSTSeparationConfig,
    *,
    preferred_stems: tuple[str, ...],
    stage_name: str,
    progress: ProgressCallback | None = None,
) -> MSSTStageOutput:
    """Run a RoFormer stage on one audio file and write one selected stem."""
    deps = _load_runtime_dependencies()
    librosa = deps["librosa"]
    np = deps["np"]
    sf = deps["sf"]

    _report(progress, f"MSST {stage_name}: loading model")
    separator = MSSTRoFormerSeparator(config, deps=deps, progress=progress)
    sample_rate = int(_cfg_get(separator.model_config.audio, "sample_rate", 44100))
    try:
        _report(progress, f"MSST {stage_name}: loading audio at {sample_rate} Hz")
        mix, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)
    except Exception as exc:
        raise MusicAgentError(f"Could not load audio for MSST {stage_name}: {audio_path}") from exc

    try:
        _report(progress, f"MSST {stage_name}: running model")
        stems = separator.separate(mix, stage_name=stage_name)
    finally:
        separator.del_cache()

    selected_name, selected_audio = _select_stem(stems, preferred_stems)
    output_path = Path(output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _report(progress, f"MSST {stage_name}: writing {selected_name} to {output_path.name}")
    sf.write(str(output_path), _sanitize_audio(np.asarray(selected_audio), np), sample_rate, subtype="FLOAT")

    return MSSTStageOutput(
        output_audio=output_path,
        selected_stem=selected_name,
        source_stems=tuple(sorted(stems)),
        sample_rate=sample_rate,
        device=separator.device,
        model_type=config.model_type,
        model_path=config.model_path,
        config_path=config.config_path,
    )


class MSSTRoFormerSeparator:
    """Small in-project equivalent of the MSST RoFormer inference path."""

    def __init__(
        self,
        config: MSSTSeparationConfig,
        deps: dict[str, Any] | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.run_config = config
        self.deps = deps or _load_runtime_dependencies()
        self.progress = progress
        self.torch = self.deps["torch"]
        self.np = self.deps["np"]
        self.device = _select_device(self.torch, config.device)
        _report(self.progress, f"MSST: using device {self.device}")
        self.model_config = _load_yaml_config(config.config_path, self.deps)
        self.model = self._load_model()

    def separate(self, mix: Any, *, stage_name: str = "main pass") -> dict[str, Any]:
        """Separate an audio array shaped as channels x samples."""
        np = self.np
        model_cfg = _cfg_get(self.model_config, "model", {})
        training_cfg = self.model_config.training
        inference_cfg = self.model_config.inference

        is_stereo = bool(_cfg_get(model_cfg, "stereo", True))
        mix = np.asarray(mix, dtype=np.float32)
        if is_stereo and mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)
        elif is_stereo and mix.ndim != 1 and mix.shape[0] > 2:
            mix = np.stack([np.mean(mix, axis=0), np.mean(mix, axis=0)], axis=0)
        elif not is_stereo and mix.ndim != 1:
            mix = np.mean(mix, axis=0)

        instruments = list(_cfg_get(training_cfg, "instruments", []))
        target_instrument = _cfg_get(training_cfg, "target_instrument", None)
        if target_instrument is not None:
            instruments_for_model = [target_instrument]
        else:
            instruments_for_model = instruments

        mix_orig = mix.copy()
        norm_params = None
        if bool(_cfg_get(inference_cfg, "normalize", False)):
            mix, norm_params = self._normalize_audio(mix)

        waveforms = demix(
            self.model_config,
            self.model,
            mix,
            self.device,
            model_type=self.run_config.model_type,
            torch_module=self.torch,
            np_module=np,
            progress=self.progress,
            stage=stage_name,
        )
        if self.run_config.use_tta:
            _report(self.progress, "MSST: applying test-time augmentation")
            waveforms = self._apply_tta(mix, waveforms)

        results = {}
        for instr in instruments_for_model:
            estimates = waveforms[instr]
            if norm_params is not None:
                estimates = estimates * norm_params["std"] + norm_params["mean"]
            results[instr] = estimates.T

        if target_instrument is not None:
            other_instruments = [instr for instr in instruments if instr != target_instrument]
            if other_instruments:
                target = results[target_instrument].T
                other = _subtract_stem(mix_orig.T, target.T, np)
                results[other_instruments[0]] = other

        return results

    def del_cache(self) -> None:
        """Release accelerator memory when possible."""
        gc.collect()
        if "mps" in self.device and hasattr(self.torch, "mps"):
            self.torch.mps.empty_cache()
        if "cuda" in self.device and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _load_model(self) -> Any:
        torch = self.torch
        _report(self.progress, f"MSST: building {self.run_config.model_type} model")
        model = _build_model(self.run_config.model_type, self.model_config)
        _report(self.progress, f"MSST: loading checkpoint {self.run_config.model_path}")
        state_dict = _load_state_dict(self.run_config.model_path, self.device, deps=self.deps)
        if isinstance(state_dict, dict) and "state" in state_dict:
            state_dict = state_dict["state"]
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _strip_module_prefix(state_dict)
        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise MusicAgentError(
                "Could not load MSST checkpoint into the selected RoFormer model. "
                "Check that --model-type matches the YAML config and checkpoint."
            ) from exc
        model = model.to(self.device)
        model.eval()
        torch.backends.cudnn.benchmark = True
        _report(self.progress, "MSST: model ready")
        return model

    def _normalize_audio(self, audio: Any) -> tuple[Any, dict[str, float]]:
        mono = audio.mean(0) if audio.ndim > 1 else audio
        mean = float(mono.mean())
        std = float(mono.std()) or 1.0
        return (audio - mean) / std, {"mean": mean, "std": std}

    def _apply_tta(self, mix: Any, waveforms_orig: dict[str, Any]) -> dict[str, Any]:
        track_proc_list = [mix[..., ::-1].copy(), -1.0 * mix.copy()]
        for index, augmented_mix in enumerate(track_proc_list):
            waveforms = demix(
                self.model_config,
                self.model,
                augmented_mix,
                self.device,
                model_type=self.run_config.model_type,
                torch_module=self.torch,
                np_module=self.np,
                progress=self.progress,
                stage=f"TTA pass {index + 1}/{len(track_proc_list)}",
            )
            for stem_name, waveform in waveforms.items():
                if index == 0:
                    waveforms_orig[stem_name] += waveform[..., ::-1].copy()
                else:
                    waveforms_orig[stem_name] -= waveform
        for stem_name in waveforms_orig:
            waveforms_orig[stem_name] /= len(track_proc_list) + 1
        return waveforms_orig


def demix(
    config: Any,
    model: Any,
    mix: Any,
    device: str,
    *,
    model_type: str,
    torch_module: Any | None = None,
    np_module: Any | None = None,
    progress: ProgressCallback | None = None,
    stage: str = "separation",
) -> dict[str, Any]:
    """Chunked overlap-add demix loop adapted for RoFormer inference."""
    torch, np, nn = _torch_numpy(torch_module=torch_module, np_module=np_module)
    mix = torch.tensor(mix, dtype=torch.float32)
    if mix.ndim == 1:
        mix = mix.unsqueeze(0)

    audio_cfg = config.audio
    inference_cfg = config.inference
    training_cfg = config.training
    chunk_size = int(_cfg_get(audio_cfg, "chunk_size"))
    num_overlap = int(_cfg_get(inference_cfg, "num_overlap", 4))
    batch_size = int(_cfg_get(inference_cfg, "batch_size", 1))
    step = max(1, int(chunk_size // num_overlap))
    fade_size = int(chunk_size // 10)
    border = chunk_size - step
    length_init = mix.shape[-1]

    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    window_start, window_middle, window_finish = _build_windows(torch, chunk_size, fade_size)
    target_instrument = _cfg_get(training_cfg, "target_instrument", None)
    instruments = list(_cfg_get(training_cfg, "instruments", []))
    source_count = 1 if target_instrument is not None else len(instruments)
    req_shape = (source_count,) + tuple(mix.shape)

    result = torch.zeros(req_shape, dtype=torch.float32)
    counter = torch.zeros(req_shape, dtype=torch.float32)
    batch_data = []
    batch_locations = []
    index = 0
    amp_enabled = bool(_cfg_get(training_cfg, "use_amp", True)) and str(device).startswith("cuda")
    total_chunks = max(1, (mix.shape[-1] + step - 1) // step)
    processed_chunks = 0
    last_progress = 0.0
    _report(progress, f"MSST: {stage} chunks 0/{total_chunks} (0%)")

    with torch.amp.autocast("cuda", enabled=amp_enabled):
        with torch.inference_mode():
            while index < mix.shape[-1]:
                part = mix[:, index : index + chunk_size].to(device)
                length = part.shape[-1]
                if length < chunk_size:
                    if length > chunk_size // 2 + 1:
                        part = nn.functional.pad(part, (0, chunk_size - length), mode="reflect")
                    else:
                        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode="constant", value=0)
                batch_data.append(part)
                batch_locations.append((index, length))
                index += step

                if len(batch_data) >= batch_size or index >= mix.shape[-1]:
                    prediction = model(torch.stack(batch_data, dim=0))
                    for batch_index, (start, length) in enumerate(batch_locations):
                        window = window_middle
                        if start == 0:
                            window = window_start
                        elif index >= mix.shape[-1]:
                            window = window_finish
                        result[..., start : start + length] += prediction[batch_index][..., :length].cpu() * window[..., :length]
                        counter[..., start : start + length] += window[..., :length]
                    processed_chunks += len(batch_locations)
                    now = monotonic()
                    if processed_chunks >= total_chunks or processed_chunks == len(batch_locations) or now - last_progress >= 5:
                        percent = min(100, int((processed_chunks / total_chunks) * 100))
                        _report(progress, f"MSST: {stage} chunks {processed_chunks}/{total_chunks} ({percent}%)")
                        last_progress = now
                    batch_data = []
                    batch_locations = []

    estimated_sources = result / torch.clamp(counter, min=1e-8)
    estimated_sources = estimated_sources.cpu().numpy()
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]

    names = [target_instrument] if target_instrument is not None else instruments
    return {name: value for name, value in zip(names, estimated_sources)}


def _build_model(model_type: str, config: Any) -> Any:
    try:
        if model_type == "bs_roformer":
            from .msst_roformer import BSRoformer

            return BSRoformer(**dict(config.model))
        if model_type == "mel_band_roformer":
            from .msst_roformer import MelBandRoformer

            return MelBandRoformer(**dict(config.model))
    except ImportError as exc:
        raise MusicAgentError(f"MSST RoFormer dependencies are not installed. {INSTALL_HINT}") from exc
    raise MusicAgentError(f"Unsupported MSST model_type '{model_type}'.")


def _load_yaml_config(path: Path, deps: dict[str, Any]) -> Any:
    yaml = deps["yaml"]
    ConfigDict = deps["ConfigDict"]
    try:
        with path.open() as handle:
            return ConfigDict(yaml.load(handle, Loader=yaml.FullLoader))
    except Exception as exc:
        raise MusicAgentError(f"Could not read MSST YAML config: {path}") from exc


def _load_state_dict(path: Path, device: str, deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    if path.suffix == ".safetensors":
        load_file = deps["load_file"]
        return load_file(str(path), device=device)
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)
    except Exception as exc:
        raise MusicAgentError(f"Could not load MSST checkpoint: {path}") from exc


def _load_runtime_dependencies() -> dict[str, Any]:
    try:
        import librosa
        from ml_collections import ConfigDict
        import numpy as np
        import soundfile as sf
        import torch
        import yaml
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MusicAgentError(f"MSST separation dependencies are not installed. {INSTALL_HINT}") from exc
    return {
        "ConfigDict": ConfigDict,
        "librosa": librosa,
        "load_file": load_file,
        "np": np,
        "sf": sf,
        "torch": torch,
        "yaml": yaml,
    }


def _torch_numpy(*, torch_module: Any | None = None, np_module: Any | None = None) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise MusicAgentError(f"MSST separation dependencies are not installed. {INSTALL_HINT}") from exc
    return torch_module or torch, np_module or np, nn


def _select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise MusicAgentError("MSST device 'cuda' was requested, but CUDA is not available.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise MusicAgentError("MSST device 'mps' was requested, but Apple MPS is not available.")
    return requested


def _build_windows(torch: Any, chunk_size: int, fade_size: int) -> tuple[Any, Any, Any]:
    window_start = torch.ones(chunk_size)
    window_middle = torch.ones(chunk_size)
    window_finish = torch.ones(chunk_size)
    if fade_size > 0:
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_start[-fade_size:] *= fadeout
        window_finish[:fade_size] *= fadein
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein
    return window_start, window_middle, window_finish


def _strip_module_prefix(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        return state_dict
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _stem_by_name(stems: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    lookup = {key.lower(): value for key, value in stems.items()}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def _select_stem(stems: dict[str, Any], names: tuple[str, ...]) -> tuple[str, Any]:
    lookup = {key.lower(): (key, value) for key, value in stems.items()}
    for name in names:
        match = lookup.get(name.lower())
        if match is not None:
            return match
    found = ", ".join(sorted(stems)) or "none"
    wanted = ", ".join(names)
    raise MusicAgentError(f"MSST stage did not produce an expected stem. Wanted one of: {wanted}. Found: {found}.")


def _mix_to_samples_channels(mix: Any) -> Any:
    if mix.ndim == 1:
        return mix[:, None]
    return mix.T


def _subtract_stem(mix: Any, stem: Any, np: Any) -> Any:
    mix = _as_samples_channels(mix, np)
    stem = _as_samples_channels(stem, np)
    channels = max(mix.shape[1], stem.shape[1])
    mix = _match_channels(mix, channels, np)
    stem = _match_channels(stem, channels, np)
    length = min(mix.shape[0], stem.shape[0])
    return mix[:length] - stem[:length]


def _as_samples_channels(audio: Any, np: Any) -> Any:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio[:, None]
    return audio


def _match_channels(audio: Any, channels: int, np: Any) -> Any:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, channels, axis=1)
    return audio[:, :channels]


def _sanitize_audio(audio: Any, np: Any) -> Any:
    audio = _as_samples_channels(audio, np)
    np.nan_to_num(audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(obj, key, default)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _stage_env_name(env_prefix: str | None, key: str) -> str:
    if env_prefix is None:
        return f"MUSIC_AGENT_MSST_{key}"
    normalized = env_prefix.strip().upper().replace("-", "_")
    return f"MUSIC_AGENT_MSST_{normalized}_{key}"


def _resolve_existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise MusicAgentError(f"{label} file does not exist: {path}")
    if not path.is_file():
        raise MusicAgentError(f"{label} path is not a file: {path}")
    return path.resolve()


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
