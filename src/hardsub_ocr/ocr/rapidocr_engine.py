from __future__ import annotations

import time
from typing import Any
import numpy as np

from hardsub_ocr.models import OcrResult
from hardsub_ocr.ocr.line_grouper import group_ocr_boxes
from hardsub_ocr.subtitle.text_normalizer import merge_ocr_lines, normalize_text


class RapidOcrEngine:
    name = "rapidocr-onnxruntime"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR이 설치되지 않았습니다. setup.ps1을 실행하세요.") from exc
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray) -> OcrResult:
        started = time.perf_counter()
        raw, _ = self._engine(image)
        items: list[tuple[Any, str, float]] = []
        for item in raw or []:
            if len(item) < 3:
                continue
            box, text, confidence = item[0], str(item[1]), float(item[2])
            items.append((box, text, confidence))
        grouped = group_ocr_boxes(items)
        raw_lines = [line["text"] for line in grouped]
        line_boxes = [line["box"] for line in grouped]
        before, text, overlaps = merge_ocr_lines(raw_lines)
        confidence = sum(line["confidence"] for line in grouped) / len(grouped) if grouped else 0.0
        return OcrResult(text, normalize_text(text), confidence, raw, [item[0] for item in items],
                         (time.perf_counter() - started) * 1000, raw_lines, before, text, overlaps, line_boxes)
