#!/usr/bin/env python3
"""End-to-end Bilibili English video -> Chinese subtitles + Chinese dub.

The default path is intentionally conservative: it accepts only subtitles that
yt-dlp exposes as normal subtitles, not automatic captions. Use --allow-asr to
fall back to local Whisper transcription when no English subtitle is available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
import srt
import soundfile as sf
from yt_dlp import YoutubeDL


ROOT = Path(__file__).resolve().parent
DEFAULT_FFMPEG = shutil.which("ffmpeg") or "/home/sil/workspace/conda_envs/veddo-tts/bin/ffmpeg"
DEFAULT_FFPROBE = shutil.which("ffprobe") or "/home/sil/workspace/conda_envs/veddo-tts/bin/ffprobe"
DEFAULT_COSYVOICE = Path("/home/sil/workspace/CosyVoice")
DEFAULT_COSYVOICE_MODEL = DEFAULT_COSYVOICE / "pretrained_models/CosyVoice-300M-SFT"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_GLOSSARY = ROOT / "glossary.json"
DEFAULT_DEEPSEEK_KEY_FILE = ROOT / "deepseek_api_key.txt"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
TRANSLATION_CACHE_VERSION = "deepseek-v1"


@dataclass
class DubChunk:
    index: int
    start: float
    end: float
    text: str
    sub_indexes: list[int]

    @property
    def slot(self) -> float:
        return max(0.2, self.end - self.start)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def output_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def ffprobe_duration(path: Path) -> float:
    return float(
        output_text(
            [
                DEFAULT_FFPROBE,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    )


def td(seconds: float) -> dt.timedelta:
    return dt.timedelta(seconds=max(0.0, seconds))


def validate_video_id(video_id: str) -> str:
    if not re.fullmatch(r"(BV[0-9A-Za-z]+|av[0-9]+)", video_id):
        raise SystemExit(f"视频编号格式不正确: {video_id}")
    return video_id


def bilibili_url(video_id: str) -> str:
    return f"https://www.bilibili.com/video/{video_id}/"


def load_one_glossary(glossary_path: Path) -> dict[str, str]:
    if not glossary_path.exists():
        return {}
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    if not isinstance(glossary, dict):
        raise ValueError(f"术语表必须是 JSON object: {glossary_path}")
    return {str(k): str(v) for k, v in glossary.items() if str(k).strip()}


def ensure_local_glossary(local_glossary_path: Path) -> None:
    if not local_glossary_path.exists():
        local_glossary_path.write_text("{}\n", encoding="utf-8")


def load_glossaries(global_glossary_path: Path, local_glossary_path: Path | None) -> tuple[dict[str, str], str]:
    global_glossary = load_one_glossary(global_glossary_path)
    local_glossary = load_one_glossary(local_glossary_path) if local_glossary_path else {}
    glossary = {**global_glossary, **local_glossary}
    raw = json.dumps(
        {
            "global": global_glossary,
            "local": local_glossary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] if glossary else "no-glossary"
    return glossary, digest


def normalize_chinese_spacing(text: str) -> str:
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+([，。！？；：、])", r"\1", text)
    text = re.sub(r"([（《])\s+(?=[\u4e00-\u9fff])", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+([）》])", r"\1", text)
    return text.strip()


def apply_glossary(translated: str, source_text: str, glossary: dict[str, str]) -> str:
    text = translated
    source_lower = source_text.lower()
    for source, target in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(source.lower())}(?![A-Za-z0-9])", source_lower):
            text = re.sub(re.escape(source), target, text, flags=re.I)
            if target not in text:
                text = f"{text}（{target}）"
    return normalize_chinese_spacing(text)


def read_deepseek_api_key(key_file: Path) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"找不到 DeepSeek API key：请设置 DEEPSEEK_API_KEY 或写入 {key_file}")
    return key


def extract_json_object(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"DeepSeek 返回不是 JSON：{text[:500]}")


def deepseek_translate_batch(
    client: OpenAI,
    *,
    model: str,
    batch: list[dict],
    glossary: dict[str, str],
    reasoning_effort: str,
    retries: int = 3,
) -> dict[int, str]:
    glossary_lines = "\n".join(f"- {source} => {target}" for source, target in sorted(glossary.items()))
    payload = {
        "glossary": glossary,
        "subtitles": batch,
    }
    system_prompt = (
        "你是专业英文视频本地化译者，负责把英文字幕翻译成自然、准确、适合中文配音的简体中文。\n"
        "要求：\n"
        "1. 严格保留输入 id，不要增删字幕条目。\n"
        "2. 译文必须忠实英文原意，避免逐词硬译，中文要口语自然，适合 TTS 朗读。\n"
        "3. 专名、人名、频道名、作品名不要臆译；不确定时保留英文或常见译名。\n"
        "4. 必须执行术语表；英文出现术语时，中文译文必须包含对应中文术语。\n"
        "5. 不要解释，不要输出 Markdown，只输出 JSON object。\n"
        'JSON 格式：{"translations":[{"id":1,"text":"中文译文"}]}'
    )
    if glossary_lines:
        system_prompt += "\n\n术语表：\n" + glossary_lines
    user_prompt = "请翻译以下字幕 JSON：\n" + json.dumps(payload, ensure_ascii=False)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                reasoning_effort=reasoning_effort,
                extra_body={"thinking": {"type": "enabled"}},
            )
            content = response.choices[0].message.content or ""
            data = extract_json_object(content)
            translations = data.get("translations")
            if not isinstance(translations, list):
                raise ValueError(f"DeepSeek JSON 缺少 translations: {content[:500]}")
            result: dict[int, str] = {}
            for item in translations:
                result[int(item["id"])] = normalize_chinese_spacing(str(item["text"]))
            missing = {entry["id"] for entry in batch} - set(result)
            if missing:
                raise ValueError(f"DeepSeek 返回缺少字幕 id: {sorted(missing)}")
            return result
        except Exception as exc:
            last_error = exc
            print(f"[translate] DeepSeek batch retry {attempt}/{retries}: {exc}", flush=True)
    raise RuntimeError(f"DeepSeek 翻译失败: {last_error}") from last_error


def subtitle_text(sub: srt.Subtitle) -> str:
    return " ".join(sub.content.replace("\n", " ").split())


def subtitle_langs(info: dict) -> list[str]:
    return sorted((info.get("subtitles") or {}).keys())


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return str(value)


def download_video_and_subs(video_id: str, out_dir: Path, *, force: bool, cookies_from_browser: str) -> tuple[Path, Path | None, dict]:
    url = bilibili_url(video_id)
    video_path = out_dir / f"{video_id}.source.mp4"
    info_path = out_dir / f"{video_id}.info.json"
    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{video_id}.source.%(ext)s"),
        "writesubtitles": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "srt/best",
        "convertsubtitles": "srt",
        "writeautomaticsub": False,
        "cookiefile": None,
        "quiet": False,
        "no_warnings": False,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if video_path.exists() and not force:
        ydl_opts["skip_download"] = True
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    info_path.write_text(json.dumps(json_safe(info), ensure_ascii=False, indent=2), encoding="utf-8")

    candidates = sorted(out_dir.glob(f"{video_id}.source.*.srt"))
    en_srt = next((p for p in candidates if re.search(r"\.en(-US|-GB)?\.srt$", p.name)), None)
    if en_srt:
        fixed = out_dir / f"{video_id}.en.srt"
        if fixed != en_srt:
            fixed.write_text(en_srt.read_text(encoding="utf-8"), encoding="utf-8")
        return video_path, fixed, info
    return video_path, None, info


def make_asr_srt(video_path: Path, out_srt: Path, whisper_model: str) -> None:
    import whisper

    model = whisper.load_model(whisper_model, device="cuda")
    result = model.transcribe(str(video_path), language="en", task="transcribe", fp16=True, verbose=False)
    subs = []
    for index, seg in enumerate(result.get("segments", []), start=1):
        text = " ".join(str(seg.get("text", "")).split())
        if text:
            subs.append(srt.Subtitle(index=index, start=td(seg["start"]), end=td(seg["end"]), content=text))
    if not subs:
        raise RuntimeError("Whisper 没有识别出英文字幕")
    out_srt.write_text(srt.compose(subs), encoding="utf-8")


def translate_srt(
    en_srt: Path,
    zh_srt: Path,
    cache_path: Path,
    global_glossary_path: Path,
    local_glossary_path: Path | None,
    *,
    deepseek_model: str,
    deepseek_key_file: Path,
    deepseek_base_url: str,
    deepseek_reasoning_effort: str,
    batch_size: int,
) -> list[srt.Subtitle]:
    subs = list(srt.parse(en_srt.read_text(encoding="utf-8")))
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    glossary, digest = load_glossaries(global_glossary_path, local_glossary_path)
    client = OpenAI(api_key=read_deepseek_api_key(deepseek_key_file), base_url=deepseek_base_url)

    pending: list[dict] = []
    key_by_id: dict[int, str] = {}
    source_by_id: dict[int, str] = {}
    original_text_by_id = {sub.index: subtitle_text(sub) for sub in subs}

    def flush() -> None:
        if not pending:
            return
        print(f"[translate] DeepSeek batch {pending[0]['id']}-{pending[-1]['id']} / {len(subs)}", flush=True)
        translated = deepseek_translate_batch(
            client,
            model=deepseek_model,
            batch=pending,
            glossary=glossary,
            reasoning_effort=deepseek_reasoning_effort,
        )
        for item_id, zh_text in translated.items():
            key = key_by_id[item_id]
            cache[key] = apply_glossary(zh_text, source_by_id[item_id], glossary)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.clear()

    for sub in subs:
        text = original_text_by_id[sub.index]
        if not text:
            continue
        key = f"{TRANSLATION_CACHE_VERSION}:{deepseek_model}:{digest}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}:{text}"
        if key in cache:
            continue
        entry = {
            "id": sub.index,
            "start": str(sub.start),
            "end": str(sub.end),
            "text": text,
        }
        pending.append(entry)
        key_by_id[sub.index] = key
        source_by_id[sub.index] = text
        if len(pending) >= batch_size:
            flush()
    flush()

    for sub in subs:
        text = original_text_by_id[sub.index]
        if not text:
            continue
        key = f"{TRANSLATION_CACHE_VERSION}:{deepseek_model}:{digest}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}:{text}"
        sub.content = cache[key]

    zh_srt.write_text(srt.compose(subs), encoding="utf-8")
    return subs


def smooth_subtitles(
    subs: list[srt.Subtitle],
    *,
    max_chars_per_sec: float,
    min_duration: float = 1.6,
    max_duration: float = 6.0,
) -> list[srt.Subtitle]:
    smoothed: list[srt.Subtitle] = []
    current: list[srt.Subtitle] = []

    def current_duration() -> float:
        return max(0.1, (current[-1].end - current[0].start).total_seconds())

    def current_cps() -> float:
        text = "".join(subtitle_text(s) for s in current)
        return count_chinese(text) / current_duration()

    def flush() -> None:
        if not current:
            return
        content = "\n".join(subtitle_text(s) for s in current)
        smoothed.append(
            srt.Subtitle(
                index=len(smoothed) + 1,
                start=current[0].start,
                end=current[-1].end,
                content=content,
            )
        )
        current.clear()

    for sub in subs:
        if not current:
            current.append(sub)
            continue
        current.append(sub)
        duration = current_duration()
        if duration >= min_duration and current_cps() <= max_chars_per_sec:
            flush()
        elif duration >= max_duration:
            flush()
    flush()
    return smoothed


def smooth_source_subtitles(
    subs: list[srt.Subtitle],
    *,
    min_duration: float = 2.0,
    max_duration: float = 8.0,
    max_chars: int = 180,
) -> list[srt.Subtitle]:
    smoothed: list[srt.Subtitle] = []
    current: list[srt.Subtitle] = []

    def flush() -> None:
        if not current:
            return
        content = " ".join(subtitle_text(s) for s in current)
        smoothed.append(
            srt.Subtitle(
                index=len(smoothed) + 1,
                start=current[0].start,
                end=current[-1].end,
                content=content,
            )
        )
        current.clear()

    for sub in subs:
        text = subtitle_text(sub)
        if not text:
            continue
        current.append(sub)
        duration = (current[-1].end - current[0].start).total_seconds()
        chars = len(" ".join(subtitle_text(s) for s in current))
        ends_sentence = bool(re.search(r"[.!?。！？]$", text))
        if duration >= min_duration and (ends_sentence or duration >= max_duration or chars >= max_chars):
            flush()
    flush()
    return smoothed


def make_chunks(subs: list[srt.Subtitle], *, min_seconds: float, max_seconds: float, max_chars: int) -> list[DubChunk]:
    chunks: list[DubChunk] = []
    current: list[srt.Subtitle] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(subtitle_text(s) for s in current)
        chunks.append(
            DubChunk(
                index=len(chunks) + 1,
                start=current[0].start.total_seconds(),
                end=current[-1].end.total_seconds(),
                text=text,
                sub_indexes=[s.index for s in current],
            )
        )
        current.clear()

    for sub in subs:
        text = subtitle_text(sub)
        if not text:
            continue
        current.append(sub)
        span = current[-1].end.total_seconds() - current[0].start.total_seconds()
        chars = len("".join(subtitle_text(s) for s in current))
        if span >= min_seconds and (span >= max_seconds or chars >= max_chars or re.search(r"[。！？.!?]$", text)):
            flush()
    flush()
    return chunks


def atempo_chain(factor: float) -> str:
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.6f}")
    return ",".join(parts)


def convert_audio_to_slot(raw: Path, fixed: Path, slot: float, *, max_duration_ratio: float) -> dict:
    raw_duration = ffprobe_duration(raw)
    target = max(0.18, slot * 0.96)
    speed_factor = raw_duration / target if raw_duration > target else 1.0
    status = "ok"
    if raw_duration > slot * max_duration_ratio:
        status = "compressed_over_threshold"
    filters = []
    if speed_factor > 1.01:
        filters.append(atempo_chain(speed_factor))
    filters.extend(["apad", f"atrim=0:{slot:.3f}"])
    run(
        [
            DEFAULT_FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-af",
            ",".join(filters),
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(fixed),
        ]
    )
    return {"raw_duration": raw_duration, "slot": slot, "speed_factor": speed_factor, "status": status}


def write_silence(path: Path, seconds: float, sample_rate: int = 44100) -> None:
    if seconds <= 0.02:
        return
    frames = int(seconds * sample_rate)
    sf.write(path, np.zeros((frames, 2), dtype=np.float32), sample_rate)


def cosyvoice_env() -> None:
    cosy = os.environ.get("COSYVOICE_ROOT", str(DEFAULT_COSYVOICE))
    third = str(Path(cosy) / "third_party/Matcha-TTS")
    sys.path.insert(0, cosy)
    sys.path.insert(0, third)


def load_cosyvoice(model_dir: Path, fp16: bool):
    cosyvoice_env()
    from cosyvoice.cli.cosyvoice import CosyVoice

    return CosyVoice(str(model_dir), load_jit=False, load_trt=False, fp16=fp16)


def synthesize_cosyvoice(cosyvoice, text: str, speaker: str, out_wav: Path, speed: float) -> None:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return
    results = cosyvoice.inference_sft(text, speaker, stream=False, speed=speed, text_frontend=True)
    first = next(results)
    wav = first["tts_speech"].detach().cpu().numpy()
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    sf.write(out_wav, wav, cosyvoice.sample_rate)


def make_dub_audio(
    subs: list[srt.Subtitle],
    out_dir: Path,
    total_duration: float,
    *,
    model_dir: Path,
    speaker: str,
    tts_speed: float,
    max_duration_ratio: float,
    fp16: bool,
    force: bool,
) -> tuple[Path, list[dict]]:
    dub_wav = out_dir / f"{out_dir.name}.zh.wav"
    if dub_wav.exists() and not force:
        return dub_wav, []

    chunks = make_chunks(subs, min_seconds=8.0, max_seconds=28.0, max_chars=180)
    segment_dir = out_dir / "tts_segments"
    segment_dir.mkdir(exist_ok=True)
    cosyvoice = load_cosyvoice(model_dir, fp16=fp16)
    if speaker not in cosyvoice.list_available_spks():
        raise RuntimeError(f"CosyVoice 没有音色 {speaker!r}; 可用: {', '.join(cosyvoice.list_available_spks())}")

    concat_file = segment_dir / "concat.txt"
    parts: list[Path] = []
    report: list[dict] = []
    cursor = 0.0
    for chunk in chunks:
        gap = max(0.0, chunk.start - cursor)
        if gap > 0.02:
            gap_path = segment_dir / f"{chunk.index:04d}_gap.wav"
            write_silence(gap_path, gap)
            parts.append(gap_path)

        raw = segment_dir / f"{chunk.index:04d}_raw.wav"
        fixed = segment_dir / f"{chunk.index:04d}_fixed.wav"
        if force:
            raw.unlink(missing_ok=True)
            fixed.unlink(missing_ok=True)
        print(f"[tts] chunk {chunk.index}/{len(chunks)} slot={chunk.slot:.2f}s", flush=True)
        synthesize_cosyvoice(cosyvoice, chunk.text, speaker, raw, speed=tts_speed)
        item = convert_audio_to_slot(raw, fixed, chunk.slot, max_duration_ratio=max_duration_ratio)
        item.update({"index": chunk.index, "start": chunk.start, "end": chunk.end, "sub_indexes": chunk.sub_indexes})
        report.append(item)
        parts.append(fixed)
        cursor = chunk.end

    tail = max(0.0, total_duration - cursor)
    if tail > 0.02:
        tail_path = segment_dir / "tail.wav"
        write_silence(tail_path, tail)
        parts.append(tail_path)

    concat_file.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in parts) + "\n", encoding="utf-8")
    run(
        [
            DEFAULT_FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-ar",
            "44100",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dub_wav),
        ]
    )
    return dub_wav, report


def count_chinese(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def quality_checks(
    en_subs: list[srt.Subtitle],
    zh_subs: list[srt.Subtitle],
    glossary: dict[str, str],
    *,
    max_chars_per_sec: float,
    tts_report: list[dict],
) -> dict:
    issues = []
    for zh in zh_subs:
        duration = max(0.1, (zh.end - zh.start).total_seconds())
        cps = count_chinese(zh.content) / duration
        if cps > max_chars_per_sec:
            issues.append({"type": "subtitle_too_dense", "index": zh.index, "chars_per_sec": round(cps, 2)})
    en_text_all = "\n".join(subtitle_text(en) for en in en_subs).lower()
    zh_text_all = "\n".join(subtitle_text(zh) for zh in zh_subs)
    for source, target in glossary.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(source.lower())}(?![A-Za-z0-9])", en_text_all):
            if target not in zh_text_all:
                issues.append({"type": "glossary_miss", "source": source, "target": target})
    for item in tts_report:
        if item.get("status") != "ok":
            issues.append({"type": "tts_duration_over_threshold", **item})
    return {"passed": not issues, "issues": issues, "tts_segments": tts_report}


def mux_output(video: Path, en_srt: Path, zh_srt: Path, dub_wav: Path, output: Path) -> None:
    run(
        [
            DEFAULT_FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(dub_wav),
            "-i",
            str(en_srt),
            "-i",
            str(zh_srt),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-map",
            "3:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "srt",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=Original",
            "-metadata:s:a:1",
            "language=chi",
            "-metadata:s:a:1",
            "title=Chinese Dub",
            "-metadata:s:s:0",
            "language=eng",
            "-metadata:s:s:0",
            "title=English",
            "-metadata:s:s:1",
            "language=chi",
            "-metadata:s:s:1",
            "title=Chinese",
            str(output),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bilibili 英文视频自动生成中文字幕、中文配音和 MKV 成片")
    parser.add_argument("video_id", help="Bilibili 视频编号，例如 BV1BbUSB4EGN")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--glossary", type=Path, default=DEFAULT_GLOSSARY, help="项目级全局术语表，默认 ./glossary.json")
    parser.add_argument("--local-glossary", type=Path, default=None, help="视频局部术语表，默认 output/<BV>/glossary.json")
    parser.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--deepseek-key-file", type=Path, default=DEFAULT_DEEPSEEK_KEY_FILE)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--deepseek-reasoning-effort", default="high")
    parser.add_argument("--translation-batch-size", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_COSYVOICE_MODEL)
    parser.add_argument("--speaker", default="中文女")
    parser.add_argument("--cookies-from-browser", default="chrome")
    parser.add_argument("--allow-asr", action="store_true", help="没有官方英文字幕时，用本地 Whisper 从原音频识别英文")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--max-chars-per-sec", type=float, default=10.0)
    parser.add_argument("--max-duration-ratio", type=float, default=1.15)
    parser.add_argument("--tts-speed", type=float, default=1.0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_id = validate_video_id(args.video_id)
    out_dir = args.output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    local_glossary = args.local_glossary or (out_dir / "glossary.json")
    ensure_local_glossary(local_glossary)

    video_path = out_dir / f"{video_id}.source.mp4"
    en_srt = out_dir / f"{video_id}.en.srt"
    if not args.skip_download:
        video_path, downloaded_srt, info = download_video_and_subs(
            video_id, out_dir, force=args.force, cookies_from_browser=args.cookies_from_browser
        )
        if downloaded_srt:
            en_srt = downloaded_srt
        else:
            langs = subtitle_langs(info)
            if not args.allow_asr:
                raise SystemExit(
                    "未找到 yt-dlp 暴露的英文官方字幕；为避免误用第三方字幕，流程已停止。"
                    f" 可用字幕语言: {langs or '无'}。如要本地转写，重新运行并加 --allow-asr。"
                )
    if not video_path.exists():
        raise SystemExit(f"找不到视频文件: {video_path}")
    if not en_srt.exists():
        if not args.allow_asr:
            raise SystemExit(f"找不到英文字幕: {en_srt}")
        make_asr_srt(video_path, en_srt, args.whisper_model)
    en_subs = smooth_source_subtitles(list(srt.parse(en_srt.read_text(encoding="utf-8"))))
    en_srt.write_text(srt.compose(en_subs), encoding="utf-8")

    zh_srt = out_dir / f"{video_id}.zh-CN.srt"
    cache = out_dir / "translation_cache.json"
    zh_subs = translate_srt(
        en_srt,
        zh_srt,
        cache,
        args.glossary,
        local_glossary,
        deepseek_model=args.deepseek_model,
        deepseek_key_file=args.deepseek_key_file,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_reasoning_effort=args.deepseek_reasoning_effort,
        batch_size=max(1, args.translation_batch_size),
    )
    zh_subs = smooth_subtitles(zh_subs, max_chars_per_sec=args.max_chars_per_sec)
    zh_srt.write_text(srt.compose(zh_subs), encoding="utf-8")

    tts_report: list[dict] = []
    dub_wav = out_dir / f"{video_id}.zh.wav"
    if not args.skip_tts:
        total_duration = ffprobe_duration(video_path)
        dub_wav, tts_report = make_dub_audio(
            zh_subs,
            out_dir,
            total_duration,
            model_dir=args.model_dir,
            speaker=args.speaker,
            tts_speed=args.tts_speed,
            max_duration_ratio=args.max_duration_ratio,
            fp16=not args.no_fp16,
            force=args.force,
        )

    glossary, glossary_digest = load_glossaries(args.glossary, local_glossary)
    report = quality_checks(
        en_subs,
        zh_subs,
        glossary,
        max_chars_per_sec=args.max_chars_per_sec,
        tts_report=tts_report,
    )
    report.update(
        {
            "video_id": video_id,
            "source": str(video_path),
            "english_srt": str(en_srt),
            "chinese_srt": str(zh_srt),
            "dub_wav": str(dub_wav),
            "global_glossary": str(args.glossary),
            "local_glossary": str(local_glossary),
            "glossary_digest": glossary_digest,
            "translation_model": args.deepseek_model,
        }
    )
    report_path = out_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(f"质量检查未通过，详情见: {report_path}")

    if args.skip_tts:
        print(f"[done] subtitles only: {zh_srt}")
        return 0
    if not dub_wav.exists():
        raise SystemExit(f"找不到中文配音: {dub_wav}")

    output = out_dir / f"{video_id}.zh.mkv"
    mux_output(video_path, en_srt, zh_srt, dub_wav, output)
    print(f"[done] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
