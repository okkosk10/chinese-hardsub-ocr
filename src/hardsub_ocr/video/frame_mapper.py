from __future__ import annotations

from dataclasses import dataclass

from hardsub_ocr.config import Crop


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


def video_display_rect(widget_width: float, widget_height: float, video_width: int, video_height: int) -> Rect:
    if min(widget_width, widget_height, video_width, video_height) <= 0:
        return Rect(0, 0, 0, 0)
    scale = min(widget_width / video_width, widget_height / video_height)
    width, height = video_width * scale, video_height * scale
    return Rect((widget_width - width) / 2, (widget_height - height) / 2, width, height)


def map_widget_crop(selection: Rect, display: Rect, video_width: int, video_height: int) -> Crop:
    if display.width <= 0 or display.height <= 0:
        raise ValueError("영상 표시 영역이 없습니다.")
    left = max(display.x, min(selection.x, selection.x + selection.width))
    top = max(display.y, min(selection.y, selection.y + selection.height))
    right = min(display.x + display.width, max(selection.x, selection.x + selection.width))
    bottom = min(display.y + display.height, max(selection.y, selection.y + selection.height))
    if right <= left or bottom <= top:
        raise ValueError("선택 영역이 영상 밖에 있습니다.")
    sx, sy = video_width / display.width, video_height / display.height
    x, y = round((left - display.x) * sx), round((top - display.y) * sy)
    r, b = round((right - display.x) * sx), round((bottom - display.y) * sy)
    return Crop(max(0, x), max(0, y), min(video_width, r) - max(0, x), min(video_height, b) - max(0, y))

