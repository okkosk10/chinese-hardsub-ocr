from __future__ import annotations

import re

_TIME_RE = re.compile(r"^(?P<h>\d+):(?P<m>[0-5]\d):(?P<s>[0-5]\d)(?:[\.,](?P<ms>\d{1,3}))?$")


def parse_timecode(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("시간은 음수일 수 없습니다.")
        return float(value)
    text = value.strip()
    try:
        seconds = float(text)
        if seconds < 0:
            raise ValueError
        return seconds
    except ValueError:
        pass
    match = _TIME_RE.fullmatch(text)
    if not match:
        raise ValueError("시간 형식은 HH:MM:SS, HH:MM:SS.mmm 또는 초 단위 숫자입니다.")
    ms = (match.group("ms") or "0").ljust(3, "0")
    return int(match.group("h")) * 3600 + int(match.group("m")) * 60 + int(match.group("s")) + int(ms) / 1000


def format_timecode(seconds: float, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{milliseconds:03d}"

