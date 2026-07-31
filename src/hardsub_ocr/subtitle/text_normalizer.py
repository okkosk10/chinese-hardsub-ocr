from __future__ import annotations

import re
import unicodedata

_PUNCT = r"，。！？；：、,.!?;:"


def _safe_overlap(text: str) -> bool:
    """Reject punctuation-only and URL/number boundary overlaps."""
    if not text or not any("\u3400" <= char <= "\u9fff" for char in text):
        return False
    return not bool(re.fullmatch(r"[\dA-Za-z:/._-]+", text))


def merge_ocr_lines(lines: list[str], max_overlap_chars: int = 6) -> tuple[str, str, list[str]]:
    """Join OCR lines while removing the longest safe suffix/prefix overlap."""
    cleaned = [re.sub(r"\s+", " ", line).strip() for line in lines if line and line.strip()]
    if not cleaned:
        return "", "", []
    before = "".join(cleaned)
    merged = cleaned[0]
    removed: list[str] = []
    for next_line in cleaned[1:]:
        overlap = ""
        limit = min(max_overlap_chars, len(merged), len(next_line))
        for size in range(limit, 0, -1):
            candidate = next_line[:size]
            if merged.endswith(candidate) and _safe_overlap(candidate):
                # One-character overlap is useful for Chinese OCR line boundaries,
                # but never delete an entire one-character line.
                if size == 1 and (len(merged) < 3 or len(next_line) < 2):
                    continue
                overlap = candidate
                break
        if overlap:
            removed.append(overlap)
            merged += next_line[len(overlap):]
        else:
            merged += next_line
    return before, merged, removed


def normalize_text(text: str, deduplicate_lines: bool = False, max_overlap_chars: int = 6) -> str:
    # NFC는 조합 문자만 정리하고 중국어 전각 문장부호/원문 형태는 보존한다.
    text = unicodedata.normalize("NFC", text).replace("\r", "\n").replace("\t", " ")
    lines: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(rf"([{_PUNCT}])\1+", r"\1", line)
        line = re.sub(rf"\s*([{_PUNCT}])\s*", r"\1", line)
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    if deduplicate_lines and len(lines) > 1:
        return merge_ocr_lines(lines, max_overlap_chars)[1]
    return "\n".join(lines)


def comparison_key(text: str) -> str:
    return re.sub(rf"[\s{_PUNCT}]", "", normalize_text(text))
