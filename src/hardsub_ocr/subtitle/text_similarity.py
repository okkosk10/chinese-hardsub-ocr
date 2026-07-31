from rapidfuzz.fuzz import ratio

from hardsub_ocr.subtitle.text_normalizer import comparison_key


def similarity(left: str, right: str) -> float:
    a, b = comparison_key(left), comparison_key(right)
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return float(ratio(a, b))


def is_same_text(left: str, right: str, threshold: float = 82.0, short_threshold: float = 94.0) -> bool:
    shortest = min(len(comparison_key(left)), len(comparison_key(right)))
    return similarity(left, right) >= (short_threshold if shortest <= 3 else threshold)

