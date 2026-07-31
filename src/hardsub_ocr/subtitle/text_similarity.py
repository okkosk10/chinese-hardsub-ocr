from rapidfuzz.fuzz import ratio

from hardsub_ocr.subtitle.text_normalizer import comparison_key
from hardsub_ocr.subtitle.text_normalizer import normalize_text


def line_order_key(text: str) -> tuple[str, ...]:
    lines = [comparison_key(line) for line in normalize_text(text).splitlines()]
    return tuple(sorted(line for line in lines if line))


def line_order_invariant_similarity(left: str, right: str) -> float:
    left_lines, right_lines = line_order_key(left), line_order_key(right)
    if len(left_lines) < 2 and len(right_lines) < 2:
        return 0.0
    return float(ratio("\n".join(left_lines), "\n".join(right_lines)))


def similarity(left: str, right: str) -> float:
    a, b = comparison_key(left), comparison_key(right)
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    normal = float(ratio(a, b))
    return max(normal, line_order_invariant_similarity(left, right))


def is_same_text(left: str, right: str, threshold: float = 82.0, short_threshold: float = 94.0) -> bool:
    shortest = min(len(comparison_key(left)), len(comparison_key(right)))
    return similarity(left, right) >= (short_threshold if shortest <= 3 else threshold)
