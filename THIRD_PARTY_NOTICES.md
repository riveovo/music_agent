# Third-Party Notices

## MSST-WebUI RoFormer Inference Components

This project includes a minimal, modified subset of the RoFormer source
separation model code from MSST-WebUI for local vocal/accompaniment separation.

- Source project: https://github.com/SUC-DriverOld/MSST-WebUI
- Upstream purpose: WebUI and inference tools for Music Source Separation
  Training models.
- Included files: `src/music_agent/separation/msst_roformer/*`
- License: GNU Affero General Public License v3.0

The full AGPL-3.0 license text is included in `MSST_AGPL_LICENSE.txt`.

## openvpi/audio-slicer RMS Silence Slicing

This project includes an adapted RMS-based silence detection slicer inspired by
openvpi/audio-slicer.

- Source project: https://github.com/openvpi/audio-slicer
- Upstream purpose: Python audio slicing with silence detection.
- Included/adapted file: `src/music_agent/capabilities/slice_audio.py`
- License: MIT License

The OpenVPI MIT license text is included in
`OPENVPI_AUDIO_SLICER_LICENSE.txt`.

## HuanLinOTO/SVCFusion Adapter

This project includes an adapter that can call an externally provided
SVCFusion core checkout for voice conversion. No SVCFusion source files or
model weights are vendored into this repository.

- Source project: https://github.com/HuanLinOTO/SVCFusion
- Upstream purpose: Singing voice conversion core components.
- Local adapter file: `src/music_agent/voice_conversion/svcfusion.py`
- Upstream license file: `LISENCE` in the SVCFusion repository, GPL-3.0 text.

The public SVCFusion repository describes itself as core code without the
frontend entrypoint, so users must make that core code importable explicitly
when running the adapter.
