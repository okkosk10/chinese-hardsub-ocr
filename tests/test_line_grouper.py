import pytest
import numpy as np
from pathlib import Path

from hardsub_ocr.config import Crop, OcrConfig
from hardsub_ocr.models import OcrResult
from hardsub_ocr.ocr.line_grouper import group_ocr_boxes, lines_are_visually_separate
from hardsub_ocr.pipeline import OcrPipeline


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_groups_by_y_and_sorts_each_line_by_x():
    grouped = group_ocr_boxes([
        (box(100, 10, 150, 30), "世界", .9),
        (box(10, 12, 70, 32), "你好", .8),
        (box(90, 70, 150, 90), "下句", .85),
        (box(10, 68, 70, 88), "这是", .88),
    ])
    assert [line["text"] for line in grouped] == ["你好世界", "这是下句"]
    assert grouped[0]["confidence"] == pytest.approx(.85)


def test_large_vertical_gap_keeps_speaker_lines_separate():
    assert lines_are_visually_separate([[0, 0, 100, 20], [0, 40, 100, 20]])
    assert not lines_are_visually_separate([[0, 0, 100, 20], [0, 22, 100, 20]])


class LineEngine:
    name = "line-fake"

    def __init__(self, lines, boxes):
        self.lines, self.boxes = lines, boxes

    def recognize(self, image):
        return OcrResult("\n".join(self.lines), "\n".join(self.lines), .9,
                         raw_lines=self.lines, line_boxes=self.boxes)


def recognize_lines(tmp_path, lines, boxes):
    pipeline = OcrPipeline(OcrConfig(tmp_path / "unused.mp4", 0, 1, Crop(0, 0, 100, 60)),
                           LineEngine(lines, boxes))
    return pipeline._recognize_frame(0, np.zeros((60, 100, 3), np.uint8), 0)


def test_visually_separate_speaker_lines_keep_newline(tmp_path):
    candidate = recognize_lines(tmp_path, ["你先走", "我随后来"], [[0, 0, 100, 20], [0, 40, 100, 20]])
    assert candidate.normalized_text == "你先走\n我随后来"


def test_close_ocr_fragments_still_use_boundary_dedup(tmp_path):
    candidate = recognize_lines(tmp_path, ["结衣，真", "真不好意思"], [[0, 0, 100, 20], [0, 22, 100, 20]])
    assert candidate.normalized_text == "结衣，真不好意思"
