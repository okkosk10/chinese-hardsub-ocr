from __future__ import annotations

import re
import unicodedata

_PUNCT = r"，。！？；：、,.!?;:"


def normalize_text(text: str) -> str:
    # NFC는 조합 문자만 정리하고 중국어 전각 문장부호/원문 형태는 보존한다.
    text = unicodedata.normalize("NFC", text).replace("\r", "\n").replace("\t", " ")
    lines: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(rf"([{_PUNCT}])\1+", r"\1", line)
        line = re.sub(rf"\s*([{_PUNCT}])\s*", r"\1", line)
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return "\n".join(lines)


def comparison_key(text: str) -> str:
    return re.sub(rf"[\s{_PUNCT}]", "", normalize_text(text))
