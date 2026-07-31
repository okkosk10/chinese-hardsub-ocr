from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any

from hardsub_ocr.models import SubtitleSegment
from hardsub_ocr.subtitle.candidate_selector import chinese_character_ratio
from hardsub_ocr.subtitle.text_normalizer import comparison_key
from hardsub_ocr.subtitle.text_similarity import similarity
from hardsub_ocr.utils.timecode import parse_timecode

_TIME_LINE = re.compile(r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}[,.]\d{3})")
_CHINESE = re.compile(r"[\u3400-\u9fff]")
_PROTECTED_SHORT_CHINESE = {"爸爸", "好的", "不要", "谢谢"}


def _metrics(segment: SubtitleSegment) -> dict[str, float | int]:
    compact = re.sub(r"\s+", "", segment.text)
    return {
        "duration": max(0.0, segment.end_time - segment.start_time),
        "character_count": len(compact),
        "chinese_character_count": len(_CHINESE.findall(compact)),
        "chinese_character_ratio": chinese_character_ratio(segment.text),
        "confidence": segment.confidence,
    }


def _removed(segment: SubtitleSegment, reason: str) -> dict[str, Any]:
    return {"segment": segment.to_dict(), "reason": reason, "metrics": _metrics(segment)}


def _near_long_chinese(segment: SubtitleSegment, other: SubtitleSegment) -> bool:
    if abs(other.start_time - segment.end_time) > 1.0 and abs(segment.start_time - other.end_time) > 1.0:
        return False
    other_key, current_key = comparison_key(other.text), comparison_key(segment.text)
    if len(_CHINESE.findall(other_key)) < 6:
        return False
    return (similarity(segment.text, other.text) >= 60
            or (current_key and current_key in other_key)
            or (other_key and other_key in current_key))


def noise_reason(segment: SubtitleSegment, neighbors: list[SubtitleSegment]) -> str | None:
    metrics = _metrics(segment)
    compact = re.sub(r"\s+", "", segment.text)
    chinese_count = int(metrics["chinese_character_count"])
    if comparison_key(segment.text) in _PROTECTED_SHORT_CHINESE:
        return None
    if chinese_count == 0 and 1 <= len(compact) <= 3:
        return "standalone_short_non_chinese"
    if (metrics["duration"] <= 0.5 and metrics["chinese_character_ratio"] < 0.35
            and segment.confidence < 0.7):
        return "very_short_low_chinese_low_confidence"
    if metrics["duration"] <= 0.85 and chinese_count == 0 and segment.confidence < 0.7:
        return "transient_non_chinese_low_confidence"
    if (metrics["duration"] <= 0.85 and segment.confidence < 0.75
            and any(_near_long_chinese(segment, neighbor) for neighbor in neighbors)):
        return "transient_noise_near_similar_long_chinese"
    return None


def clean_segments(segments: list[SubtitleSegment], duplicate_threshold: float = 88.0,
                   merge_gap_seconds: float = 1.0) -> tuple[list[SubtitleSegment], list[dict[str, Any]]]:
    kept: list[SubtitleSegment] = []
    removed: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        neighbors = [candidate for pos, candidate in enumerate(segments)
                     if pos != index and (abs(candidate.start_time - segment.end_time) <= 1.0
                                          or abs(segment.start_time - candidate.end_time) <= 1.0)]
        reason = noise_reason(segment, neighbors)
        if reason:
            removed.append(_removed(segment, reason))
            continue
        current = replace(segment, source_frames=list(segment.source_frames))
        if (kept and current.start_time - kept[-1].end_time <= merge_gap_seconds
                and similarity(kept[-1].text, current.text) >= duplicate_threshold):
            kept[-1].end_time = max(kept[-1].end_time, current.end_time)
            kept[-1].last_seen_timestamp = max(kept[-1].last_seen_timestamp, current.last_seen_timestamp)
            kept[-1].source_frames.extend(current.source_frames)
            kept[-1].confidence = max(kept[-1].confidence, current.confidence)
            removed.append(_removed(segment, "merged_line_order_duplicate"))
            continue
        kept.append(current)
    for index, segment in enumerate(kept, 1):
        segment.index = index
    return kept, removed


def parse_srt(path: Path) -> list[SubtitleSegment]:
    text = path.read_text(encoding="utf-8-sig")
    segments: list[SubtitleSegment] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        lines = block.splitlines()
        time_index = next((index for index, line in enumerate(lines) if _TIME_LINE.fullmatch(line.strip())), None)
        if time_index is None:
            continue
        match = _TIME_LINE.fullmatch(lines[time_index].strip())
        assert match
        subtitle = "\n".join(lines[time_index + 1:]).strip()
        if subtitle:
            segments.append(SubtitleSegment(len(segments) + 1, parse_timecode(match.group("start")),
                                            parse_timecode(match.group("end")), subtitle, 1.0))
    return segments


def cleaned_srt_path(path: Path) -> Path:
    return path.with_name(path.stem + ".cleaned.srt")
