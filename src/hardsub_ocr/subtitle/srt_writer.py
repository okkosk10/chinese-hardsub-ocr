from __future__ import annotations

from pathlib import Path

from hardsub_ocr.models import SubtitleSegment
from hardsub_ocr.utils.timecode import format_timecode


def render_srt(segments: list[SubtitleSegment]) -> str:
    blocks: list[str] = []
    last_end = 0.0
    for index, segment in enumerate(segments, 1):
        if not segment.text.strip():
            continue
        start = max(segment.start_time, last_end)
        end = max(segment.end_time, start + 0.001)
        blocks.append(f"{index}\n{format_timecode(start, True)} --> {format_timecode(end, True)}\n{segment.text.strip()}")
        last_end = end
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(render_srt(segments), encoding="utf-8")
    temp.replace(path)

