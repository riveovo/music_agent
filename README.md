# AI Music Agent

AI Music Agent 是一个面向音乐生产、音频处理和数据准备的本地优先 Agent 工具箱。你可以用自然语言让 Agent 调度能力，也可以直接使用命令行子命令完成生成、分析、风格识别、人声分离、变声、音频切片和人声切片清洗。

项目的核心设计是：

- **自然语言入口**：`agent` 命令使用 OpenAI Responses API 的 ReAct 循环，在配置可用时自动选择并调用 skill。
- **能力独立可用**：每个音乐能力都有独立 CLI 子命令，适合脚本、批处理和自动化流水线。
- **Skill + 外部 Tool 扩展**：Skill 只描述“什么时候用、怎么做、允许用哪些工具”；真正执行由外部 tool 完成，方便后续接入新的模型、服务或内部工具。
- **本地优先**：默认运行路径尽量轻量；重型 ML 能力通过可选 extra 安装，模型权重不自动写入仓库。
- **中文友好**：自然语言请求、命令示例和文档都支持中文使用场景。

## 环境要求

- Python 3.12
- `ffmpeg` 和 `ffprobe`，用于音频分析、转换和部分处理能力
- 可选：`ncmdump`，用于处理 NCM 音频
- 可选：OpenAI API Key，用于 ReAct Agent

默认安装只依赖 Python 标准库和系统音频工具。MusicGen、Essentia、MSST、SVCFusion、SpeechBrain 等能力都通过可选依赖安装。

## 快速开始

未安装包时，可以在项目根目录用 `PYTHONPATH=src` 运行：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli generate \
  --prompt "轻快电子音乐" \
  --duration 5
```

也可以安装为可编辑包：

```bash
python3.12 -m pip install -e .
music-agent generate --prompt "轻快电子音乐" --duration 5
```

所有命令默认输出 JSON。音频和结果 JSON 会写入 `outputs/`。

## 常用命令

```bash
# 生成一段本地合成音乐
PYTHONPATH=src python3.12 -m music_agent.cli generate \
  --prompt "轻快电子音乐" \
  --duration 5

# 使用本地 MusicGen 生成音乐
PYTHONPATH=src python3.12 -m music_agent.cli generate \
  --provider musicgen \
  --prompt "lofi hip hop with warm piano" \
  --duration 10

# 分析音频元数据、响度和可选音乐结构
PYTHONPATH=src python3.12 -m music_agent.cli analyze \
  --audio outputs/generate/example.wav

# 识别曲风和能量情绪
PYTHONPATH=src python3.12 -m music_agent.cli recognize-style \
  --audio outputs/generate/example.wav

# 分离人声和伴奏
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems \
  --audio outputs/generate/example.wav \
  --provider heuristic

# 变声或音色转换
PYTHONPATH=src python3.12 -m music_agent.cli convert-voice \
  --audio outputs/generate/example.wav \
  --preset bright

# 按静音和长度范围切片
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio \
  --input outputs/generate/example.wav

# 用自然语言调用 Agent
PYTHONPATH=src python3.12 -m music_agent.cli agent \
  "分析这首歌的风格" \
  --audio outputs/generate/example.wav

# 启动持续对话模式
PYTHONPATH=src python3.12 -m music_agent.cli chat \
  --audio outputs/generate/example.wav

# 启动本地 Web 服务和 ChatGPT 式界面
PYTHONPATH=src python3.12 -m music_agent.cli web \
  --agent-engine auto
```

## 自然语言 Agent

`agent` 默认使用 `--agent-engine auto`：

- 如果安装了 OpenAI SDK extra 且设置了 `OPENAI_API_KEY`，会运行 OpenAI Responses API ReAct 循环。
- 如果 OpenAI 配置不可用，会自动 fallback 到旧的关键词路由，保留离线可用性。
- 如果你想强制使用某个模式，可以指定 `--agent-engine openai` 或 `--agent-engine keyword`。

安装 OpenAI Agent 依赖：

```bash
python3.12 -m pip install -e ".[agent-openai]"
export OPENAI_API_KEY="..."
```

使用 OpenAI ReAct Agent：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli agent \
  "把这首歌做人声分离，然后告诉我输出文件在哪里" \
  --audio song.wav \
  --agent-engine openai
```

启动持续对话模式：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli chat \
  --audio song.wav \
  --agent-engine auto
```

进入后可以连续输入请求：

```text
分析这首歌的风格
再帮我分离人声
把人声切成 3 到 10 秒的片段
/exit
```

`chat` 会复用同一个进程、同一套 skill/tool registry。OpenAI ReAct 模式下还会保留前文上下文，适合“再帮我……”“基于刚才结果……”这类连续工作流。`auto` 模式如果无法使用 OpenAI，会 fallback 到关键词路由；此时仍是持续进程，但没有大模型上下文理解。

交互命令：

- `/exit` 或 `/quit`：退出会话。
- `/clear`：清空 OpenAI 对话上下文，但保留当前配置和默认音频。
- `/audio PATH`：切换当前会话的默认音频路径。
- `/help`：显示交互命令。

可用参数：

- `--openai-model`：指定 ReAct Agent 模型；默认读取 `MUSIC_AGENT_OPENAI_MODEL`，否则使用项目默认模型。
- `--max-steps`：限制每轮模型和工具往返轮数。
- `--skills-path`：加载外部 `*/SKILL.md` skill 目录。
- `--tools-path`：加载外部 tool JSON 配置目录。

## Web 服务和 Web UI

Web 入口复用同一套 Agent、skill 和外部 tool 架构，适合用浏览器进行持续对话、上传音频、观察工具调用过程，并直接播放或下载输出产物。

安装 Web 依赖：

```bash
python3.12 -m pip install -e ".[web,agent-openai]"
```

启动服务：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli web \
  --host 127.0.0.1 \
  --port 8765 \
  --agent-engine auto
```

打开 `http://127.0.0.1:8765` 即可使用 Web UI。默认是本机单用户模式：浏览器会话对应后端内存里的一个 Agent session；OpenAI ReAct 模式会保留上下文，`auto` 在 OpenAI SDK 或 API key 不可用时会 fallback 到关键词路由。

Web API 提供：

- `GET /api/health`
- `GET /api/skills`
- `GET /api/tools`
- `POST /api/sessions`
- `POST /api/sessions/{session_id}/messages`
- `POST /api/sessions/{session_id}/messages/stream`
- `POST /api/sessions/{session_id}/clear`
- `POST /api/uploads/audio`
- `GET /api/artifacts/{artifact_id}`

前端使用 Vite + React + TypeScript。第一次开发前安装依赖：

```bash
npm --prefix web install
```

构建生产 UI：

```bash
npm --prefix web run build
```

构建后 FastAPI 会自动托管 `web/dist`。开发模式可以同时运行后端和 Vite：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli web
npm --prefix web run dev
```

Web 层只通过 artifact id 暴露 `outputs/` 下的产物文件，不把任意本地路径直接开放给浏览器。上传音频会写入 `outputs/uploads/{session_id}/`，随后作为当前对话的默认音频输入传给 Agent。

## Skill 和外部 Tool

项目采用两层扩展模型：

- **Skill**：标准 `SKILL.md` 文件夹，用来描述触发场景、执行步骤、输入要求和允许使用的外部工具。
- **Tool**：真正执行动作的外部工具，可以绑定 Python 函数，也可以为后续 MCP、HTTP、CLI provider 预留接口。

内置工具包括：

- `music.generate`
- `music.analyze_audio`
- `music.recognize_style`
- `music.separate_stems`
- `music.convert_voice`
- `music.slice_audio`
- `music.curate_vocal_slices`

一个最小 skill 示例：

```yaml
---
name: my-music-skill
description: Use when the user wants a custom audio workflow.
allowed_tools:
  - music.analyze_audio
required_inputs:
  - audio
---

# My Music Skill

Use `music.analyze_audio` to inspect the provided track and summarize the result.
```

外部 skill 默认从 `.agents/skills/*/SKILL.md` 发现，也可以通过 `--skills-path` 或 `MUSIC_AGENT_SKILLS_PATH` 指定。Skill 本身不会直接执行代码；Agent 必须通过 `allowed_tools` 声明的外部工具执行。

外部 tool 可以放在 `.agents/tools/*.json`，也可以通过 `--tools-path` 或 `MUSIC_AGENT_TOOLS_PATH` 指定。当前版本会执行 `python` provider；`mcp`、`http`、`cli` provider 已保留结构，调用时会返回明确的未实现提示。

Python tool 配置示例：

```json
{
  "name": "custom.my_tool",
  "description": "Run my custom music workflow.",
  "provider": "python",
  "module": "my_package.tools",
  "callable": "run_my_tool",
  "parameters": {
    "type": "object",
    "properties": {
      "audio": {
        "type": "string",
        "description": "Input audio path."
      }
    },
    "required": ["audio"],
    "additionalProperties": false
  }
}
```

对应 Python 函数接收 JSON schema 中的参数作为关键字参数，并返回可 JSON 序列化的结果。

## 音频输入规则

以下命令支持 WAV、MP3、FLAC 和 NCM 输入：

- `analyze`
- `recognize-style`
- `separate-stems`
- `convert-voice`
- `slice-audio`

统一规则：

- WAV 直接处理。
- MP3 和 FLAC 会通过 `ffmpeg` 临时转成 WAV。
- NCM 会先用 `ncmdump` 解密，再用 `ffmpeg` 转成 WAV。程序会检查常见 Homebrew 路径，例如 `/opt/homebrew/bin`。
- 默认不保留中间 WAV。需要保留时使用 `--keep-converted`。
- 如果 `ncmdump` 不在可发现路径，使用 `--ncm-converter /path/to/ncmdump` 或 `MUSIC_AGENT_NCM_CONVERTER`。

目录批处理使用同一套输入规则：

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

`--recursive` 表示递归处理子目录。批处理输出会按源音频文件保留独立子目录。

## 音乐生成

默认 `--provider synth` 是本地、免费、无额外依赖的合成器，适合测试流水线和快速占位。它不是神经网络音乐生成模型。

如需本地文本生成音乐，安装 MusicGen 依赖：

```bash
python3.12 -m pip install -e ".[musicgen-local]"
```

运行：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli generate \
  --provider musicgen \
  --model facebook/musicgen-small \
  --prompt "80s pop track with bassy drums and synth" \
  --duration 10 \
  --guidance-scale 3
```

注意：

- 首次运行会从 Hugging Face 下载模型权重。
- 当前 CLI 将 `musicgen` 输出限制在 30 秒以内。
- CPU 可运行但较慢，Apple Silicon MPS 或 CUDA 更合适。
- 请自行确认模型许可是否支持你的使用场景。

## 音乐分析

默认 `analyze --provider auto` 使用轻量元数据和响度分析。需要 BPM、节拍、调性、和弦、频谱特征和 A/B/C 段落结构时，可以使用 Essentia：

```bash
python3.12 -m pip install -e ".[analysis-essentia]"
```

```bash
PYTHONPATH=src python3.12 -m music_agent.cli analyze \
  --audio song.wav \
  --provider essentia \
  --essentia-max-sections 12
```

批处理：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli analyze \
  --audio datasets/raw_songs \
  --output-dir outputs/analysis/batch_001 \
  --recursive \
  --provider essentia
```

`sections` 输出中的 A/B/C 是重复音乐材料的结构标签，不保证等同于主歌、副歌等语义名称。

也可以让 `auto` 默认选择 Essentia：

```bash
export MUSIC_AGENT_ANALYSIS_PROVIDER=essentia
```

Essentia 是 AGPL-3.0，商业使用请确认上游许可。

## 曲风识别

默认 `recognize-style --provider auto` 只有在完整 Essentia 模型配置可用时才使用 Essentia，否则使用轻量启发式识别。

安装依赖：

```bash
python3.12 -m pip install -e ".[style-essentia]"
```

将模型文件放到 `models/style/essentia/`：

- Embedding model: `discogs-maest-30s-pw-519l-2.pb`
- Classification head: `genre_discogs519-discogs-maest-30s-pw-519l-1.pb`
- Metadata: `genre_discogs519-discogs-maest-30s-pw-519l-1.json`

运行：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli recognize-style \
  --audio song.wav \
  --provider essentia \
  --essentia-model-type discogs519_maest_30s \
  --essentia-embedding-model-path models/style/essentia/discogs-maest-30s-pw-519l-2.pb \
  --essentia-classifier-model-path models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.pb \
  --essentia-metadata-path models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.json
```

批处理：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli recognize-style \
  --audio datasets/raw_songs \
  --output-dir outputs/style/batch_001 \
  --recursive \
  --provider essentia \
  --essentia-embedding-model-path models/style/essentia/discogs-maest-30s-pw-519l-2.pb \
  --essentia-classifier-model-path models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.pb \
  --essentia-metadata-path models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.json
```

环境变量：

```bash
export MUSIC_AGENT_STYLE_ESSENTIA_MODEL_TYPE=discogs519_maest_30s
export MUSIC_AGENT_STYLE_ESSENTIA_EMBEDDING_MODEL_PATH=models/style/essentia/discogs-maest-30s-pw-519l-2.pb
export MUSIC_AGENT_STYLE_ESSENTIA_CLASSIFIER_MODEL_PATH=models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.pb
export MUSIC_AGENT_STYLE_ESSENTIA_METADATA_PATH=models/style/essentia/genre_discogs519-discogs-maest-30s-pw-519l-1.json
```

Essentia 后端会分析多个内部 30 秒窗口，聚合输出 `style`、`top_styles`、Discogs 原始标签、证据窗口和置信度。MTG TensorFlow 模型常见许可为非商业 Creative Commons，请按具体模型元数据确认。

## 人声和伴奏分离

默认 `separate-stems --provider auto` 会在完整 MSST 配置可用时使用 RoFormer 后端，否则回退到轻量 ffmpeg 启发式方案。

安装依赖：

```bash
python3.12 -m pip install -e ".[separation-msst]"
```

提供兼容的 BS-RoFormer 或 MelBand-RoFormer 人声模型：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli separate-stems \
  --audio song.wav \
  --provider msst \
  --model-type bs_roformer \
  --model-path /path/to/model.ckpt \
  --config-path /path/to/model.yaml \
  --device auto
```

可选清理阶段：

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

- instrumental 阶段会重写 `accompaniment.wav`，用于降低残留人声。
- deharmony 阶段会写出 `vocals_deharmonized.wav`。
- dereverb/de-echo 阶段会写出 `vocals_dereverbed.wav`，最终 `vocals.wav` 会替换为最后一个清理结果。

环境变量：

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

后端会写出 `vocals.wav`、`accompaniment.wav` 和 `separation.json`。长任务进度打印到 stderr，stdout 始终保留最终机器可读 JSON。模型权重不会自动下载，也不会存入仓库。项目内 RoFormer 推理代码是 MSST-WebUI 派生的最小子集，详见 `THIRD_PARTY_NOTICES.md` 和 `MSST_AGPL_LICENSE.txt`。

## 变声和音色转换

`convert-voice` 保留轻量 ffmpeg preset，也可以在提供模型和目标说话人时运行 SVCFusion 兼容的 DDSP 6.1 模型。

安装依赖：

```bash
python3.12 -m pip install -e ".[voice-svcfusion]"
```

让 SVCFusion core 仓库可导入：

```bash
export PYTHONPATH=/path/to/SVCFusion:$PYTHONPATH
```

或在命令中指定：

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

批处理：

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

常用环境变量：

```bash
export MUSIC_AGENT_SVCFUSION_MODEL_TYPE=ddsp6_1
export MUSIC_AGENT_SVCFUSION_MODEL_PATH=models/svcfusion/target/model.pt
export MUSIC_AGENT_SVCFUSION_CONFIG_PATH=models/svcfusion/target/config.yaml
export MUSIC_AGENT_SVCFUSION_SPEAKER=target_speaker
export MUSIC_AGENT_SVCFUSION_DEVICE=auto
export MUSIC_AGENT_SVCFUSION_SOURCE_PATH=/path/to/SVCFusion
```

配置完整时，`--provider auto` 会选择 SVCFusion；否则回退到 placeholder preset。项目不会 vendoring SVCFusion core，请按上游说明自行准备。

## 音频切片

安装轻量切片依赖：

```bash
python3.12 -m pip install -e ".[audio-slice]"
```

切一个文件：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio \
  --input song.mp3 \
  --output-dir outputs/slices/song \
  --min-length-ms 3000 \
  --max-length-ms 10000
```

批处理目录：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli slice-audio \
  --input datasets/raw_songs \
  --output-dir outputs/slices/batch_001 \
  --recursive
```

切片实现参考 openvpi/audio-slicer 的 RMS 静音检测思路。命令只要求目标长度范围，静音阈值和最小静音间隔会从音频 RMS 分布估计；过长片段会在内部最安静位置附近切开，让输出更接近 `--min-length-ms` 和 `--max-length-ms`。

## 人声切片清洗

当你有一批主要来自同一歌手的干声切片时，可以用 `curate-vocal-slices` 聚类说话人 embedding，保留总时长最长的歌手簇。

安装依赖：

```bash
python3.12 -m pip install -e ".[vocal-curation]"
```

运行：

```bash
PYTHONPATH=src python3.12 -m music_agent.cli curate-vocal-slices \
  --input outputs/slices/target_singer \
  --output-dir datasets/curated/target_singer \
  --min-length-ms 3000 \
  --max-length-ms 10000 \
  --distance-threshold 0.32
```

输出结构：

```text
datasets/curated/target_singer/
  accepted/
  rejected/
  review/
  curation.json
  clusters.csv
```

默认 embedding 模型是 `speechbrain/spkrec-ecapa-voxceleb`，Apache-2.0 许可。首次运行会下载到 `models/speechbrain/`，该目录已被 git 忽略。阈值越低越严格，越高越宽松，建议先在 `0.28` 到 `0.36` 之间试。

## 测试

```bash
PYTHONPATH=src python3.12 -m pytest
```

测试使用标准 `unittest` API，也可以不安装 pytest 直接运行：

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests
```

如果需要完整开发环境：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,web]"
python -m pytest
npm --prefix web install
npm --prefix web run build
```

未安装 `.[web]` 时，Web API 的 FastAPI TestClient 测试会自动跳过；session 和 artifact 的核心单元测试仍会运行。
