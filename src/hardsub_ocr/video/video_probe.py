from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from hardsub_ocr.models import VideoInfo


def require_ffmpeg() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg/ffprobe를 찾을 수 없습니다. 설치 후 bin 폴더를 PATH에 추가하세요.")
    return ffmpeg, ffprobe


def probe_video(path: Path) -> VideoInfo:
    _, ffprobe = require_ffmpeg()
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate,codec_name:format=duration", "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"ffprobe 실패({result.returncode}): {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        num, den = (int(x) for x in stream.get("avg_frame_rate", "0/1").split("/"))
        return VideoInfo(int(stream["width"]), int(stream["height"]),
                         float(data["format"]["duration"]), num / den if den else 0.0,
                         stream.get("codec_name", ""))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe 결과에서 영상 정보를 읽지 못했습니다.") from exc

