# Bilibili 英文视频自动中文配音

这个项目用于把 Bilibili 英文视频按 BV 号端到端转换为带中文字幕和中文配音的视频。流程包括下载视频、获取英文字幕或本地 ASR、DeepSeek 翻译、术语表约束、本地 CosyVoice TTS、质量检查和 MKV 封装。

## 环境

Python/TTS 环境使用本地 conda 环境：

```bash
/home/sil/workspace/conda_envs/veddo-tts/bin/python
```

首次配置或复现环境：

```bash
./setup_veddo_tts.sh
```

脚本默认依赖：

- `yt-dlp` 下载 Bilibili 视频并读取 Chrome 登录 cookies
- `openai` SDK 调用 DeepSeek
- `openai-whisper` 在没有英文字幕时本地 ASR
- `CosyVoice-300M-SFT` 本地中文 TTS
- `ffmpeg` 封装音视频和字幕

## DeepSeek Key

不要提交 API key。程序按以下顺序读取 key：

1. 环境变量 `DEEPSEEK_API_KEY`
2. 项目根目录 `deepseek_api_key.txt`

`deepseek_api_key.txt` 已在 `.gitignore` 中忽略。

## 术语表

术语表分两层：

- 全局术语表：`glossary.json`
- 视频局部术语表：`output/<BV>/glossary.json`

局部术语表会覆盖全局术语表中的同名条目。两个文件都是 JSON object，例如：

```json
{
  "quaternion": "四元数",
  "Gimbal lock": "万向节锁"
}
```

## 使用

完整转换：

```bash
./auto_bilibili_dub.sh BV1BbUSB4EGN --allow-asr
```

如果 Bilibili 提供英文字幕，程序会优先使用字幕。没有英文字幕时，只有加上 `--allow-asr` 才会用本地 Whisper 从原音频识别英文。

只生成/验证中文字幕，不跑 TTS：

```bash
./auto_bilibili_dub.sh BV1BbUSB4EGN --skip-download --allow-asr --skip-tts
```

常用参数：

```bash
--translation-batch-size 8
--translation-workers 3
--asr-polish-workers 3
--tts-workers 2
--tts-devices 0,1
--speaker 中文女
--max-chars-per-sec 10
--max-duration-ratio 1.15
--deepseek-model deepseek-v4-pro
```

速度相关：

- DeepSeek ASR 纠错和翻译会按批次并发调用 API。
- CosyVoice TTS 会按字幕 chunk 分组并行合成，默认使用 `--tts-devices 0,1`。
- 如果只有一张 GPU，运行时改成 `--tts-workers 1 --tts-devices 0`。
- 已生成的英文字幕、ASR 纠错、翻译、TTS 片段都会缓存，失败后可以续跑。

智能处理：

- `--allow-asr` 启用 Whisper 后，程序会先合并英文碎片，再用 DeepSeek 修正 ASR 错词、重复词和断句。
- 翻译阶段继续使用 DeepSeek，并强制套用全局/局部术语表。
- 如需跳过英文 ASR 纠错，可加 `--no-asr-polish`。

## 输出

默认输出到：

```text
output/<BV>/
  <BV>.source.mp4
  <BV>.en.srt
  <BV>.zh-CN.srt
  <BV>.zh.wav
  <BV>.zh.mkv
  glossary.json
  asr_polish_cache.json
  quality_report.json
  translation_cache.json
```

`output/` 是生成物目录，已被 `.gitignore` 忽略。

## 质量门禁

程序会生成 `quality_report.json` 并检查：

- 中文字幕字速是否过高
- 英文中出现的术语是否按术语表出现在中文中
- TTS 生成音频是否需要过度压缩才能塞进原视频时间轴

如果质量检查失败，程序会停止，不会强行封装最终 MKV。

## 文件说明

- `auto_bilibili_dub.py`：端到端主程序
- `auto_bilibili_dub.sh`：固定 `veddo-tts` 环境和 CosyVoice 路径的启动脚本
- `download_bilibili_hd.sh`：仅下载 Bilibili 高清视频的辅助脚本
- `setup_veddo_tts.sh`：环境配置/复现脚本
- `glossary.json`：项目级全局动态术语表
