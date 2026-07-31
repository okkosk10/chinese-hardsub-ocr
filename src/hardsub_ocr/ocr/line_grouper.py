from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(slots=True)
class TextBox:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    raw_box: Any = None

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


def _to_text_box(box: Any, text: str, confidence: float) -> TextBox:
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return TextBox(text.strip(), confidence, min(xs), min(ys), max(xs), max(ys), box)


def _join_fragments(parts: list[TextBox]) -> str:
    result = ""
    for part in parts:
        if not result:
            result = part.text
        elif result[-1:].isascii() and result[-1:].isalnum() and part.text[:1].isascii() and part.text[:1].isalnum():
            result += " " + part.text
        else:
            result += part.text
    return result


def group_ocr_boxes(items: list[tuple[Any, str, float]], center_tolerance: float = 0.65) -> list[dict[str, Any]]:
    """Group OCR boxes into visual lines by y, then sort fragments by x."""
    boxes = [_to_text_box(box, text, confidence) for box, text, confidence in items if text.strip()]
    if not boxes:
        return []
    typical_height = median(box.height for box in boxes)
    groups: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda item: (item.center_y, item.x1)):
        best: list[TextBox] | None = None
        best_distance = float("inf")
        for group in groups:
            center = sum(item.center_y for item in group) / len(group)
            distance = abs(box.center_y - center)
            if distance <= max(typical_height, box.height) * center_tolerance and distance < best_distance:
                best, best_distance = group, distance
        if best is None:
            groups.append([box])
        else:
            best.append(box)
    result: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item.x1)
        x1, y1 = min(item.x1 for item in ordered), min(item.y1 for item in ordered)
        x2, y2 = max(item.x2 for item in ordered), max(item.y2 for item in ordered)
        result.append({
            "text": _join_fragments(ordered),
            "confidence": sum(item.confidence for item in ordered) / len(ordered),
            "box": [x1, y1, x2 - x1, y2 - y1],
            "raw_boxes": [item.raw_box for item in ordered],
        })
    return sorted(result, key=lambda line: line["box"][1])


def lines_are_visually_separate(line_boxes: list[list[float]], gap_ratio: float = 0.45) -> bool:
    if len(line_boxes) < 2:
        return False
    ordered = sorted(line_boxes, key=lambda box: box[1])
    for upper, lower in zip(ordered, ordered[1:]):
        gap = lower[1] - (upper[1] + upper[3])
        typical_height = max(1.0, (upper[3] + lower[3]) / 2)
        if gap >= typical_height * gap_ratio:
            return True
    return False
